import argparse
from datetime import datetime, timedelta
from glob import glob
import gc
import json
import matplotlib.pyplot as plt
import numpy as np
import obspy
import os
import pandas as pd
import pypandoc
import re
import shutil
import timeit
import traceback
import warnings

from obspy.io.mseed.util import get_start_and_end_time
from obspy.io.stationxml.core import validate_stationxml
from sklearn.linear_model import LinearRegression

from ioos_qc import qartod

import nfsi_obs as nf
from utilities import config_handler, logger, ReportGenerator

gc.set_debug(gc.DEBUG_UNCOLLECTABLE)
feature_test = False    # set to True to test new features

current_dir = os.path.dirname(os.path.abspath(__file__))
# Ensure resource directory exists
resource_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'resource/OBSDataPipeline')
if not os.path.isdir(resource_dir):
    os.makedirs(resource_dir)


def process(data_dir, obs_log, network_id, config, output_dir=None, metadata=None, channel_map=None, project_meta=None,
            full=True, detrend=False, backup=True, use_existing_plots=False, flags_from_config=False, **kwargs):
    """
    Extra keyword arguments are included as report parameters (must match variables in template file).
    """
    error_count = 0
    proc_start = timeit.default_timer()
    g_log.info("start")

    debug_info = {'timing': {}}

    # Get flags from config if necessary
    if flags_from_config:
        # False fallback value will default to same values as function definition
        full = not config.get('dataset', 'function_check', fallback=False)
        detrend = config.get('dataset', 'detrend_seismic', fallback=False)
        backup = not config.get('dataset', 'skip_backup', fallback=False)
        use_existing_plots = config.get('dataset', 'use_existing_plots', fallback=False)

    # Initialize report parameters dictionary with input keywords
    report_params = {}
    report_params.update(kwargs)
    # Add empty lists for channel-specific information
    for key in ['seismic_channels', 'ocean_channels', 'power_channels', 'health_channels']:
        report_params[key] = []

    # Windowing parameters for seismic data
    win_len = 3600
    overlap = 0.5
    if 'psdWindowsSecs' in report_params:
        win_len = report_params['psdWindowSecs']
    if 'psdOverlapPercent' in report_params:
        overlap = report_params['psdOverlapPercent'] / 100
    spec_win = config.getint('seismic', 'spectrogram_window', fallback=60)
    for key, val in zip(['window_length', 'overlap_percent', 'spectrogram_window'], [win_len, overlap, spec_win]):
        config['seismic'][key] = str(val)

    base_time = timeit.default_timer()
    g_log.debug("Basic processing setup time: {0} seconds".format((base_time - proc_start)))
    debug_info['timing']['base_setup'] = base_time - proc_start

    # Start/end of time period to analyze: (1) on seafloor, (2) off-ship, (3) deployment start/end, (4) project start/end
    data_start, data_end = None, None
    if not pd.isnull(obs_log['Date/Time on Seafloor (UTC)'].values[0]):
        data_start = obspy.UTCDateTime(pd.to_datetime(obs_log['Date/Time on Seafloor (UTC)'].values[0]))
    elif not pd.isnull(obs_log['Launch Date/Time (UTC)'].values[0]):
        data_start = obspy.UTCDateTime(pd.to_datetime(obs_log['Launch Date/Time (UTC)'].values[0]))
    elif 'start_date' in project_meta['this_deployment']:
        data_start = obspy.UTCDateTime(project_meta['this_deployment']['start_date'])
    elif 'start_date' in project_meta:
        data_start = obspy.UTCDateTime(project_meta['start_date'])
    if not pd.isnull(obs_log['Date/Time Released (UTC)'].values[0]):
        data_end = obspy.UTCDateTime(pd.to_datetime(obs_log['Date/Time Released (UTC)'].values[0]))
    elif not pd.isnull(obs_log['Recovery Date/Time (UTC)'].values[0]):
        data_end = obspy.UTCDateTime(pd.to_datetime(obs_log['Recovery Date/Time (UTC)'].values[0]))
    elif 'end_date' in project_meta['this_deployment']:
        data_end = obspy.UTCDateTime(project_meta['this_deployment']['end_date']) + 24 * 60 * 60
    elif 'end_date' in project_meta:
        data_end = obspy.UTCDateTime(project_meta['end_date']) + 24 * 60 * 60

    # Read station metadata file
    station_info = None
    if metadata is not None:
        g_log.info("Reading metadata from file {0}".format(metadata))
        filetype = os.path.splitext(metadata)[-1]
        if filetype in ['.dataless', '.metadata']:
            station_info = nf.metadata.read_dataless(metadata)
        elif filetype == '.xml':
            # read as StationXML format
            station_info = obspy.read_inventory(metadata)
        else:
            g_log.error("Unrecognized file format. Unable to read metadata.")
    else:
        # Search data_dir for suitable metadata file
        seed_files = glob(os.path.join(data_dir, '**/*.dataless'), recursive=True)
        seed_files.extend(glob(os.path.join(data_dir, '**/*.metadata'), recursive=True))
        xml_files = glob(os.path.join(data_dir, '**/*.xml'), recursive=True)
        if len(seed_files) > 0:
            if len(seed_files) > 1:
                g_log.warning("Multiple dataless SEED volumes found in data directory, using {0}.".format(seed_files[0]))
            else:
                g_log.info("Found dataless SEED volume in data directory: {0}".format(seed_files[0]))
            # take first dataless SEED file
            station_info = nf.metadata.read_dataless(seed_files[0])
            config['dataset']['metadata'] = seed_files[0]
        elif len(xml_files) > 0:
            for xf in xml_files:
                is_sxml = validate_stationxml(xf)[0]
                if is_sxml and (station_info is None):
                    g_log.info("Found StationXML file in data directory: {0}".format(xf))
                    station_info = obspy.read_inventory(xf)
                    config['dataset']['metadata'] = xf
        else:
            g_log.warning("No metadata file provided, and none found in data directory.")

    meta_time = timeit.default_timer()
    g_log.debug("Time spent reading station metadata file: {0} seconds".format((meta_time - base_time)))
    debug_info['timing']['station_meta'] = meta_time - base_time

    # Find data files and backup if necessary
    # TODO: Remove file backup here once it has been copied to pre-processing script (QC doesn't change miniSEED files)
    raw_files = glob(os.path.join(data_dir, '**/*.mseed'), recursive=True)
    g_log.info("Found {0} miniSEED file(s) in data directory and sub-folders".format(len(raw_files)))
    debug_info['num_files'] = len(raw_files)
    backup_exists = False
    if output_dir is None:
        if backup:
            # Make a backup copy of as-recorded raw data if no separate output directory is specified (files will be modified in-place)
            raw_dir = os.path.join(data_dir, 'raw_recorded')
            if not os.path.exists(raw_dir):
                g_log.info("Copying raw data to backup directory {0}".format(raw_dir))
                os.makedirs(raw_dir)
                for rf in raw_files:
                    shutil.copy2(rf, raw_dir)
            else:
                backup_exists = True
                g_log.info("Backup of raw data already exists: {0}".format(raw_dir))
        else:
            g_log.info("Skipping backup of raw data")
        output_dir = data_dir

    # Write QC processing configuration to file
    with open(os.path.join(output_dir, 'QC_config.ini'), 'w') as configfile:
        config.write(configfile)

    search_time = timeit.default_timer()
    g_log.debug("Time spent searching for data files and backing up raw data: {0} seconds".format((search_time - meta_time)))
    debug_info['timing']['file_search'] = search_time - meta_time

    # label files by channel name
    labels = []
    for rf in raw_files:
        if backup_exists:
            if re.match(r'.*raw_recorded.*', rf):
                continue
        file_name = re.split(r'/|\\', rf)[-1]
        ch_name = file_name.split('_')[1]
        labels.append({'channel': ch_name, 'path': rf})
    labeled_files = pd.DataFrame(labels)
    g_log.info("Files contain data for {0} unique set(s) of channels".format(len(np.unique(labeled_files['channel'].values))))
    debug_info['num_channels'] = len(np.unique(labeled_files['channel'].values))

    sort_time = timeit.default_timer()
    g_log.debug("Time spent sorting and labeling data files: {0} seconds".format((sort_time - search_time)))
    debug_info['timing']['file_sort'] = sort_time - search_time

    # Initialize arrays for saving stats
    all_gaps = []
    centring = pd.DataFrame()
    power_stats = pd.DataFrame()
    avg_power = []
    voltage_stats = []

    arr_time = timeit.default_timer()
    g_log.debug("Time spent setting up arrays for stats: {0} seconds".format((arr_time - sort_time)))
    debug_info['timing']['array_setup'] = arr_time - sort_time

    # Add keys for running totals
    debug_info['timing'].update({
        'file_read': 0.,
        'time_cut': 0.,
        'apply_meta': 0.,
        'group_setup': 0.,
        'gap_test': 0.,
        'group_assign': 0.,
        'meta_admin': 0.,
        'trace_plot': 0.,
        'psd_calc': 0.,
        'psd_plot': 0.,
        'spec_calc': 0.,
        'spec_plot': 0.,
        'qartod': 0.,
        'power_analysis': 0.,
        'trace_analysis': 0.,
        'long_series_check': 0.,
    })
    # Loop through data files (grouped by channel set name)
    for label, files in labeled_files.groupby('channel'):
        ch_start = timeit.default_timer()
        g_log.info("Begin processing channel set {0}".format(label))
        g_log.info("{0} data file(s) in list".format(len(files.index)))

        try:
            # Data file buffering for long time periods (should only be needed for seismic data)
            if len(files.index) > 3:
                filetimes = []
                for rf in files['path'].values:
                    times = get_start_and_end_time(rf)
                    filetimes.append(times)
                filetimes = np.array(filetimes)
                files_start = min(filetimes[:, 0])
                files_end = max(filetimes[:, 1])

                startend = timeit.default_timer()
                debug_info['timing']['long_series_check'] += startend - ch_start

                trace_info, gaps, buff_time = nf.plotting.buffer_seismic_data(files['path'].values, output_dir, g_log,
                                                                              network_id, station_info, channel_map,
                                                                              project_meta, win_len, spec_win, overlap,
                                                                              use_existing_plots=use_existing_plots)
                for key in buff_time:
                    if key in debug_info['timing']:
                        debug_info['timing'][key] += buff_time[key]
                    else:
                        debug_info['timing'][key] = buff_time[key]

                channel_type = trace_info['channelType']

                all_gaps.extend(gaps)
                report_params[channel_type + '_channels'].append(trace_info)

                g_log.info('{0} | {1} - {2} | {3}'.format(trace_info['seedID'],
                                                          files_start.datetime.strftime('%Y-%m-%d %H:%M:%S.%f'),
                                                          files_end.datetime.strftime('%Y-%m-%d %H:%M:%S.%f'),
                                                          trace_info['channelName']))
            else:
                # Read all files in list
                data = obspy.Stream()
                for rf in files['path'].values:
                    temp = obspy.read(rf)
                    for tr in temp:
                        data.append(tr)
                data.merge()

                read_time = timeit.default_timer()
                g_log.debug("Time spent reading data file(s): {0} seconds".format((read_time - ch_start)))
                debug_info['timing']['file_read'] += read_time - ch_start

                # Cut data to time period of interest (if start/end times provided)
                if (data_start is not None) or (data_end is not None):
                    # This shouldn't change `data` if there is no data to cut out
                    data.trim(data_start, data_end, nearest_sample=False)

                sf_time = timeit.default_timer()
                g_log.debug("Time spent cutting to period of interest: {0} seconds".format((sf_time - read_time)))
                debug_info['timing']['time_cut'] += sf_time - read_time

                # Populate metadata from other files as necessary
                data = nf.metadata.update_metadata(data, network_id, g_log, station_info, channel_map, project_meta)
                data.merge()
                print(data)

                metadata_time = timeit.default_timer()
                g_log.debug("Time spent applying metadata: {0} seconds".format((metadata_time - sf_time)))
                debug_info['timing']['apply_meta'] += metadata_time - sf_time

                # Perform QC
                # TODO: Combine single and multi-channel cases to simplify code (no real reason to separate) -> TEST multi-channel

                # Gap test
                gaps = data.get_gaps()
                all_gaps.extend(gaps)
                if len(gaps) > 0:
                    g_log.info('Found {0} gap(s) or overlap(s) in recorded data'.format(len(gaps)))
                    data.print_gaps()

                gt_time = timeit.default_timer()
                g_log.debug("Time spent for gap test: {0} seconds".format((gt_time - metadata_time)))
                debug_info['timing']['gap_test'] += gt_time - metadata_time

                for tr in data:
                    tr_starttime = timeit.default_timer()

                    # Channel type determines what analysis gets run on this trace
                    channel_type = nf.metadata.get_channel_type(tr.meta.channel)

                    cg_time = timeit.default_timer()
                    g_log.debug("Time spent assigning to channel group: {0} seconds".format((cg_time - tr_starttime)))
                    debug_info['timing']['group_assign'] += cg_time - tr_starttime

                    # Start gathering trace information for report
                    trace_info = {
                        'seedID': tr.id,
                        'channelName': tr.id,
                        'order': 100
                    }
                    if hasattr(tr.meta, 'description'):
                        trace_info['channelName'] = tr.meta.description

                    # Get channel info from project metadata JSON
                    channel_info = None
                    if project_meta is not None:
                        try:
                            channel_info = list(filter(lambda ch: ch['channel_id'] == tr.meta.channel, project_meta['channels']))[0]
                        except (KeyError, IndexError):
                            pass

                    dmin, dmax = None, None
                    qc_config = None
                    if channel_info is not None:
                        if 'hide' in channel_info:
                            if channel_info['hide']:
                                continue
                        if 'max' in channel_info:
                            dmax = float(channel_info['max'])
                        if 'min' in channel_info:
                            dmin = float(channel_info['min'])
                        if 'order' in channel_info:
                            trace_info['order'] = int(channel_info['order'])
                        if 'qc_config' in channel_info:
                            qc_config = channel_info['qc_config']

                    more_meta_time = timeit.default_timer()
                    g_log.debug("Time spent with other metadata admin: {0} seconds".format((more_meta_time - cg_time)))
                    debug_info['timing']['meta_admin'] += more_meta_time - cg_time

                    # Time series plot (applies instrument sensitivity in-place if response present in tr.meta)
                    trace_info['traceLoc'] = nf.plotting.trace_plot(tr, output_dir, dmin, dmax, qc_config, use_existing_plots)

                    plt_time = timeit.default_timer()
                    g_log.debug("Time spent plotting trace: {0} seconds".format((plt_time - more_meta_time)))
                    debug_info['timing']['trace_plot'] += plt_time - more_meta_time

                    # Noise level QC steps (seismic channels and hydrophone) -> if channel code == "CHx" or "HDF"
                    if channel_type == 'seismic':
                        for metaKey, reportKey in zip(['azimuth', 'dip'], ['azimuth', 'dip']):
                            if hasattr(tr.meta, metaKey):
                                trace_info[reportKey] = tr.meta[metaKey]

                        # Detrend seismic data (RMS linear fit)
                        if detrend:
                            tr.detrend('linear')
                            demean_data_plot = os.path.join(output_dir, 'demean_{0}.png'.format(tr.id))
                            if not (use_existing_plots and os.path.isfile(demean_data_plot)):
                                data.plot(outfile=demean_data_plot)

                        start_plots = timeit.default_timer()
                        # Spectrogram
                        trace_info['specLoc'] = nf.plotting.spectrogram(tr, output_dir, spec_win, overlap, use_existing_plots)
                        done_spec = timeit.default_timer()
                        debug_info['timing']['spec_plot'] += done_spec - start_plots

                        # Plot PSDs of data
                        trace_info['psdLoc'] = nf.plotting.psd_plot(tr, output_dir, win_len, overlap, use_existing_plots)
                        done_psd = timeit.default_timer()
                        debug_info['timing']['psd_plot'] += done_psd - done_spec

                        if full:
                            # TODO: Decide if the same operations are appropriate for the hydrophone data or not
                            # TODO: Calculate hourly PSDs
                            # TODO: Average PSD value at 0.2 Hz (save out for comparison with other sensors in the same network)
                            # TODO: Linearity of PSD curves
                            g_log.warning("Full QC of seismic noise not yet implemented")

                    else:
                        # Analysis of auxiliary data
                        timestamps = pd.to_datetime(tr.times(type='timestamp'), unit='s').values
                        # maybe smooth out state-of-health channels? or come up with some way to automatically QC them for anomalous sections
                        start_tran = timeit.default_timer()
                        if qc_config is not None:
                            if 'qartod' in qc_config:
                                if 'gross_range_test' in qc_config['qartod']:
                                    range_check = qartod.gross_range_test(tr.data, **qc_config['qartod']['gross_range_test'])
                                    if np.any(range_check > 1):
                                        num_sus = np.sum(range_check == 3)
                                        num_fail = np.sum(range_check == 4)
                                        g_log.info('Channel {0} has suspect values at {1} sample(s) ({3:.1%}) and failing values at {2} sample(s) ({4:.1%})'.format(tr.id, num_sus, num_fail, num_sus / len(range_check), num_fail / len(range_check)))
                                        # TODO: Add fail/suspect stats to report as well as log printout
                                    check_trace = obspy.Trace(range_check, header=tr.stats)
                                    #trace_info['qcPlotLoc'] = nf.plotting.qartod_plot(check_trace, output_dir, 'gross_range_check', use_existing_plots)

                                if feature_test:
                                    if re.match(r'[A-Z]M[1-3ENZ]', tr.meta.channel) and ('flat_line_test' in qc_config['qartod']):
                                        # centring channels only, must have flat-line test criteria specified
                                        # TODO: Need to re-visit this, not sure it's doing what we want even...
                                        flt_params = qc_config['qartod']['flat_line_test'].copy()
                                        flatline = qartod.flat_line_test(tr.data, timestamps,
                                                                         int(flt_params.pop('suspect_threshold')),
                                                                         int(flt_params.pop('fail_threshold')))
                                        centring[tr.id] = pd.Series(flatline, index=timestamps)
                        done_qartod = timeit.default_timer()
                        debug_info['timing']['qartod'] += done_qartod - start_tran

                        if channel_type == 'power':
                            # Voltage and power consumption channels
                            trace_start = tr.meta.starttime
                            trace_end = tr.meta.endtime
                            first_window = obspy.UTCDateTime(trace_start.year, trace_start.month, trace_start.day)
                            last_window = obspy.UTCDateTime(trace_end.year, trace_end.month, trace_end.day - 1)
                            window_length = 3 * 24 * 60 * 60    # 3 days in seconds
                            window_offset = 24 * 60 * 60        # 1 day in seconds
                            num_windows = int(round((last_window - first_window) / window_offset))

                            if tr.meta.channel == 'LE3':
                                # TODO: Show power consumption as positive rather than negative (as recorded)
                                # Power consumption
                                report_params['meanPower'] = '{:.3f}'.format(np.mean(tr.data))

                                # 3-day rolling window of average power consumption
                                window_start = first_window
                                while window_start < last_window:
                                    window = tr.slice(window_start, window_start + window_length)
                                    if (not np.ma.isMaskedArray(window.data) and len(window.data) > 0) or window.data.count() > 0:
                                        avg_power.append([window_start.datetime, window.data.mean()])
                                    window_start += window_offset

                                # TODO: Get times of data writes (spikes 45 minutes apart)
                                if (qc_config is not None) and ('qartod' in qc_config) and ('spike_test' in qc_config['qartod']):
                                    spikes = qartod.spike_test(tr.data, **qc_config['qartod']['spike_test'])
                            if tr.meta.channel == 'ME4':
                                # Battery voltage
                                # 3-day rolling window for stats
                                window_start = first_window
                                while window_start < last_window:
                                    window = tr.slice(window_start, window_start + window_length)
                                    if (not np.ma.isMaskedArray(window.data) and len(window.data) > 0) or window.data.count() > 0:
                                        secs = np.array(window.times(type='relative'))
                                        if isinstance(window.data, np.ma.MaskedArray):
                                            mask = np.ma.getmaskarray(window.data)
                                            secs_valid = secs[mask == False].reshape(-1, 1)
                                            valid_data = window.data[mask == False]
                                            reg = LinearRegression().fit(secs_valid, valid_data)
                                            r2 = reg.score(secs_valid, valid_data)   # R^2 coefficient of linear fit (should be very close to 1)
                                        else:
                                            reg = LinearRegression().fit(secs.reshape(-1, 1), window.data)
                                            r2 = reg.score(secs.reshape(-1, 1), window.data)
                                        gradient = reg.coef_[0] * 1000 * 60 * 60 * 24   # convert V/s to mV/day for voltage gradient
                                        voltage_stats.append([window_start.datetime, window.data.min(), window.max(), window.data.mean(), gradient, r2])
                                    window_start += window_offset
                        done_power = timeit.default_timer()
                        debug_info['timing']['power_analysis'] += done_power - done_qartod

                        # TODO: Analysis of state-of-health variables?
                        # TODO: Down-sample external pressure and temperature data (plot and save as netCDF)

                    tran_time = timeit.default_timer()
                    g_log.debug("Time spent performing trace-specific analysis: {0} seconds".format((tran_time - plt_time)))
                    debug_info['timing']['trace_analysis'] += tran_time - plt_time

                    # Summary statistics
                    if hasattr(tr.meta, 'response'):
                        units = tr.meta.response.instrument_sensitivity.input_units
                    else:
                        units = ''
                    print("{0} | {1} - {2} | {3} | Average {4:.3f} {5}".format(
                        tr.id,
                        tr.meta.starttime.strftime('%Y-%m-%d %H:%M:%S.%f'),
                        tr.meta.endtime.strftime('%Y-%m-%d %H:%M:%S.%f'),
                        trace_info['channelName'],
                        np.mean(tr.data),
                        units
                    ))

                report_params[channel_type + '_channels'].append(trace_info)
        except Exception as e:
            error_count += 1
            msg = str(e)
            g_log.error(traceback.format_exc())
            g_log.error(msg)

    loop_time = timeit.default_timer()
    g_log.debug("Time spent processing data files: {0} seconds".format((loop_time - arr_time)))
    debug_info['timing']['all_proc'] = loop_time - arr_time

    # Check centring behaviour
    if len(centring.columns) > 0:
        is_centred = centring.eq(4).all(axis='columns')
        # List of time periods where is_centred is True -> [start, end, npts]
        centred = nf.get_true_periods(is_centred)
        
        # TODO: Compile text to summarize centring behaviour
        ctx = ''

        centring_plot = os.path.join(output_dir, 'centring_{0}.png'.format(obs_log['OBS ID'].values[0]))
        if not (use_existing_plots and os.path.isfile(centring_plot)):
            fig, ax = plt.subplots(1, 1, figsize=[8, 2.5])
            is_centred.astype(float).plot(kind='line', ax=ax)
            fig.savefig(centring_plot)
            plt.close(fig)

        report_params['centring'] = {
            'plot': centring_plot,
            'text': ctx
        }

    centre_time = timeit.default_timer()
    g_log.debug("Time spent checking centring behaviour: {0} seconds".format((centre_time - loop_time)))
    debug_info['timing']['centring_summary'] = centre_time - loop_time

    # Parse gap information for report
    # TODO: Include check for duplicates (may come out of buffered seismic data)
    if len(all_gaps) > 0:
        report_params['gapList'] = []
        for gap in all_gaps:
            report_params['gapList'].append({
                'id': '.'.join(gap[0:4]),
                'start': gap[4].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                'end': gap[5].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                'sec': '{:.3f}'.format(gap[6]),
                'samp': gap[7]
            })

    gap_time = timeit.default_timer()
    g_log.debug("Time spent formatting gap information: {0} seconds".format((gap_time - centre_time)))
    debug_info['timing']['gap_format'] = gap_time - centre_time

    # Combine voltage/power statistics and make plots
    if len(avg_power) > 0 or len(voltage_stats) > 0:
        pwr = pd.DataFrame(avg_power, columns=['Start', 'Avg_Power'])
        pwr.set_index('Start', drop=False)
        vlt = pd.DataFrame(voltage_stats, columns=['Start', 'Min_Volts', 'Max_Volts', 'Avg_Volts', 'Gradient', 'R2_coef'])
        vlt.set_index('Start', drop=True)

        power_stats['Start'] = pwr['Start']
        power_stats['End'] = power_stats['Start'] + timedelta(days=3)
        power_stats['Voltage_Min'] = vlt['Min_Volts']
        power_stats['Voltage_Max'] = vlt['Max_Volts']
        power_stats['Voltage_Mean'] = vlt['Avg_Volts']
        power_stats['Voltage_gradient'] = vlt['Gradient']
        power_stats['Voltage_fit'] = vlt['R2_coef']
        power_stats['Power_Mean'] = pwr['Avg_Power']
        power_stats = power_stats.assign(Aquarius_ID=report_params['obsId'])
        # TODO: Add deployment ID from Sensor Tracker integration (for combining stats with other deployments)
        power_stats['Plot_Time'] = power_stats['Start'] + (power_stats['End'] - power_stats['Start']) / 2

        # Save statistics to CSV for further analysis
        csv_name = 'voltage_power_stats_{0}_{1}_{2}.csv'.format(report_params['obsId'],
                                                                pd.to_datetime(power_stats['Start'].min()).strftime('%Y-%m-%d'),
                                                                pd.to_datetime(power_stats['End'].max()).strftime('%Y-%m-%d'))
        power_stats.to_csv(os.path.join(output_dir, csv_name))

        # Average power vs time
        avgpow_plot = os.path.join(output_dir, 'power_mean_{0}.png'.format(obs_log['OBS ID'].values[0]))
        if not (use_existing_plots and os.path.isfile(avgpow_plot)):
            fig, ax = plt.subplots(1, 1, figsize=[8, 2.5])
            power_stats.plot(x='Plot_Time', y='Power_Mean', kind='line', ax=ax, xlabel='Date/Time', ylabel='Average Power Consumption (W)', legend=False)
            fig.tight_layout()
            fig.savefig(avgpow_plot)
            plt.close(fig)

        # Average voltage vs time
        avgvlt_plot = os.path.join(output_dir, 'voltage_mean_{0}.png'.format(obs_log['OBS ID'].values[0]))
        if not (use_existing_plots and os.path.isfile(avgvlt_plot)):
            fig, ax = plt.subplots(1, 1, figsize=[8, 2.5])
            power_stats.plot(x='Plot_Time', y='Voltage_Mean', kind='line', ax=ax, xlabel='Date/Time', ylabel='Average Voltage (V)', legend=False)
            fig.tight_layout()
            fig.savefig(avgvlt_plot)
            plt.close(fig)

        # Voltage gradient vs time
        vltgrd_plot = os.path.join(output_dir, 'voltage_gradient_{0}.png'.format(obs_log['OBS ID'].values[0]))
        if not (use_existing_plots and os.path.isfile(vltgrd_plot)):
            fig, ax = plt.subplots(1, 1, figsize=[8, 2.5])
            power_stats.plot(x='Plot_Time', y='Voltage_gradient', kind='line', ax=ax, xlabel='Date/Time', ylabel='Voltage Gradient (mV/day)', legend=False)
            fig.tight_layout()
            fig.savefig(vltgrd_plot)
            plt.close(fig)

        report_params['batteryStats'] = {}
        report_params['batteryStats']['meanPowerPlot'] = avgpow_plot
        report_params['batteryStats']['meanVoltPlot'] = avgvlt_plot
        report_params['batteryStats']['gradVoltPlot'] = vltgrd_plot

        # TODO: Calculate expected hibernate date/time (6500mV)
        hib_thres = 6500
        latest_V = power_stats['Voltage_Min'].values[-1] * 1000
        latest_win = pd.to_datetime(power_stats['End'].values[-1])
        days_to_hibernate = -(latest_V - hib_thres) / power_stats['Voltage_gradient'].values[-1]
        const_grad = timedelta(days=days_to_hibernate) + latest_win
        const_acc = pd.NaT
        lookup = pd.NaT
        min_hib = pd.Series([const_grad, const_acc, lookup]).min()
        report_params['batteryStats']['HibernateEstimate'] = min_hib.strftime('%Y-%m-%d')

    battery_time = timeit.default_timer()
    g_log.debug("Time spent checking battery stats: {0} seconds".format((battery_time - gap_time)))
    debug_info['timing']['battery_summary'] = battery_time - gap_time

    # Sort channel information by specified order
    for ch_type in ['seismic', 'ocean', 'power', 'health']:
        sorted_channels = sorted(report_params[ch_type + '_channels'], key=lambda d: d['order'])
        report_params[ch_type + '_channels'] = sorted_channels

    # Save report to *.md and *.pdf formats
    report_md = os.path.join(output_dir, 'QC_report_{0}_auto.md'.format(obs_log['OBS ID'].values[0]))
    qcReport = ReportGenerator(type='qc')
    md_out, report_buffer = qcReport.write_report(report_params, report_md)

    pandoc_args = []
    tex_path = os.path.abspath(os.path.expanduser(os.path.expandvars(config.get('common', 'pdftex_path', fallback=None))))
    if tex_path is not None:
        pandoc_args.append('--pdf-engine={0}'.format(tex_path))
    pandoc_args.extend(['--toc'])

    report_pdf = os.path.join(output_dir, 'QC_report_{0}_auto.pdf'.format(obs_log['OBS ID'].values[0]))
    report_converted = pypandoc.convert_text(report_buffer, to='pdf', format='md', outputfile=report_pdf, extra_args=pandoc_args)
    g_log.info("Report saved as {0}".format(report_pdf))

    report_time = timeit.default_timer()
    g_log.debug("Time spent creating report: {0} seconds".format((report_time - battery_time)))
    debug_info['timing']['create_report'] = report_time - battery_time

    g_log.info("end")
    print(debug_info)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform basic QC for OBS data. Will correct channel identifiers if '
                                                 'optional --channelmap argument is provided. Does not require clock '
                                                 'drift correction to have been applied. For most parameters, command '
                                                 'line arguments will override values in config.ini file.')
    parser.add_argument('--base_dir', dest='base_dir', help="Base directory where all files are stored (or will be "
                                                            "specified relative to).")
    parser.add_argument('--relative_paths', dest="relative_paths", action="store_true",
                        help="If true, all other path arguments are specified relative to the base directory.")
    parser.add_argument('--data_dir', dest="data_dir", help="Directory where OBS data is stored.")
    parser.add_argument('--datalog', dest="datalog",
                        help="Log file from deployment/recovery. Must include station identifiers and clock drift "
                             "measurements. If not specified, assumed to be a file called 'log.xlsx' in the data "
                             "directory. Preferred format is XLSX (or similar spreadsheet) following NFSI template.")
    parser.add_argument('--logdelimiter', dest="log_delim",
                        help="If the OBS log file is delimited text (other than comma-delimited), use this to specify "
                             "the column delimiter.")
    parser.add_argument('--obsid', dest="obs_id",
                        help="OBS identifier: station name or serial number")
    parser.add_argument('--start', dest="startdate", help="Start date of deployment to be analyzed, as YYYYMMDD")
    parser.add_argument('--network', dest="network_id",
                        help="Network identifier assigned by FDSN for this project. Default 'XX' for test data.")
    parser.add_argument('--outdir', dest="outdir",
                        help="Output directory, if different from data directory")
    parser.add_argument('--channelmap', dest="channel_map",
                        help="File mapping as-recorded channel codes to their correct values.")
    parser.add_argument('--metadata', dest="metadata_file",
                        help="Path to metadata file (dataless SEED or StationXML). If not specified, will search "
                             "data_dir for a suitable file.")
    parser.add_argument('--extra_meta', dest="extra_meta",
                        help='Optional JSON file with extra description and QC information')
    parser.add_argument('--function_check', dest="function_check", action="store_true",
                        help="Perform basic QC to check Aquarius functionality only. False by default to perform full "
                             "QC.")
    parser.add_argument('--detrend_seismic', dest="detrend_seis", action="store_true",
                        help="Detrend seismic data (RMS linear fit). False by default.")
    parser.add_argument('--skip_backup', dest="skip_backup", action="store_true",
                        help="Skip creating a backup copy of the raw data files. False by default.")
    parser.add_argument('--use_existing_plots', dest='use_existing_plots', action='store_true',
                        help='Do not re-create plots which already exist in output directory. False by default.')
    # TODO: When using ST, project name will come from there instead
    parser.add_argument('--projectname', dest="project_name",
                        help="Project name to be displayed in reports. If not specified, code looks in file extra_meta "
                             "instead.")
    parser.add_argument('--config', dest='config_path', help="Path to config file (if not using default).")
    parser.add_argument('--debug', dest='debug', action='store_true',
                        help="Activate debug mode (more verbose logging). Command-line only.")
    # TODO: Implement a config file to replace most of these arguments (simplify terminal command to make it more user-friendly to run)

    try:
        start_time = datetime.now()
        t0 = timeit.default_timer()
        args = parser.parse_args()
        # TODO: Handle when obs_id is only given in config...
        obs_id = args.obs_id

        g_log = logger.get_general_logger(start_time, obs_id, debug=args.debug)
        g_log.info("\n\n=====================================================================")
        g_log.info("Starting job: {0}".format(str(args)))

        if args.base_dir:
            base_dir = os.path.abspath(os.path.expanduser(os.path.expandvars(args.base_dir)))
        else:
            base_dir = None

        if args.config_path:
            config_path = args.config_path
            if args.relative_paths:
                if base_dir is not None:
                    config = config_handler.get_config(os.path.join(base_dir, args.config_path))
                else:
                    raise RuntimeError('Missing command-line argument: Cannot use relative paths if no base_dir specified.')
            else:
                config = config_handler.get_config(os.path.abspath(os.path.expanduser(os.path.expandvars(args.config_path))))
        else:
            config = config_handler.get_config()

        # Copy existing config info and add/update from command line arguments
        full_config = config_handler.copy_config(config)

        if base_dir is None:
            base_dir = os.path.abspath(os.path.expanduser(os.path.expandvars(config.get('dataset', 'base_dir', fallback=os.path.join(resource_dir, 'test_data')))))
        full_config['dataset']['base_dir'] = base_dir

        if args.relative_paths:
            relpath = True
        else:
            relpath = config.getboolean('dataset', 'relative_paths', fallback=False)
        full_config['dataset']['relative_paths'] = str(relpath)

        if args.obs_id:
            obs_identifier = args.obs_id
        else:
            obs_identifier = config.get('dataset', 'obsid', fallback=None)
        if obs_identifier is None:
            warnings.warn('No valid OBS identifier given, using default AQU-0000.')
            g_log.warn('No valid OBS identifier given, using default AQU-0000.')
            obs_identifier = 'AQU-0000'
        full_config['dataset']['obsid'] = obs_identifier
        id_type = 'unknown'
        if re.match(r'AQU-[0-9a-fA-F]{4}', obs_identifier):
            id_type = 'serial'
        elif re.match(r'D[aA][lL][_\-][0-9]{2,3}', obs_identifier):
            id_type = 'obs_name'

        if args.data_dir:
            data_path = args.data_dir
        else:
            data_path = config.get('dataset', 'data_dir', fallback=None)
        if data_path is not None:
            full_config['dataset']['data_dir'] = data_path
            if relpath:
                data_dir = os.path.join(base_dir, data_path)
            else:
                data_dir = os.path.abspath(os.path.expanduser(os.path.expandvars(data_path)))
        else:
            data_dir = os.path.join(base_dir, 'AQU-0260')
            if relpath:
                full_config['dataset']['data_dir'] = 'AQU-0260'
            else:
                full_config['dataset']['data_dir'] = data_dir

        output_dir, out_path = None, None
        if args.outdir:
            out_path = args.outdir
        else:
            out_path = config.get('dataset', 'outdir', fallback=None)
        if out_path is not None:
            full_config['dataset']['outdir'] = out_path
            if relpath:
                output_dir = os.path.join(base_dir, out_path)
            else:
                output_dir = os.path.abspath(os.path.expanduser(os.path.expandvars(out_path)))

        if args.datalog:
            datalog = args.datalog
        else:
            datalog = config.get('dataset', 'datalog', fallback=None)
        if datalog is not None:
            full_config['dataset']['datalog'] = datalog
            if relpath:
                data_log_file = os.path.join(base_dir, datalog)
            else:
                data_log_file = os.path.abspath(os.path.expanduser(os.path.expandvars(datalog)))
        else:
            data_log_file = os.path.join(base_dir, 'log.xlsx')
            if relpath:
                full_config['dataset']['datalog'] = 'log.xlsx'
            else:
                full_config['dataset']['datalog'] = data_log_file

        if args.log_delim:
            log_delim = args.log_delim
        else:
            log_delim = config.get('dataset', 'logdelimiter', fallback=',')
        full_config['dataset']['logdelimiter'] = log_delim

        deploy_start = None
        if args.startdate:
            deploy_start = datetime.strptime(args.startdate, "%Y%m%d")
        elif config.get('dataset', 'start'):
            deploy_start = datetime.strptime(config.get('dataset', 'start'), "%Y%m%d")

        g_log.info('Reading project metadata from {0}...'.format(data_log_file))
        obs_log_info = nf.io.parse_obs_log(data_log_file, log_delim)
        # Find this OBS in the basic, deployment, and recovery metadata tables
        base_meta, dep, rec = None, None, None
        if id_type == 'serial':
            base_meta = obs_log_info['basic'].loc[obs_log_info['basic']['OBS ID'] == obs_identifier]
            dep = obs_log_info['deployment'].loc[obs_log_info['deployment']['OBS ID'] == obs_identifier]
            rec = obs_log_info['recovery'].loc[obs_log_info['recovery']['OBS ID'] == obs_identifier]
        elif id_type == 'obs_name':
            base_meta = obs_log_info['basic'].loc[obs_log_info['basic']['OBS Name'] == obs_identifier]
            dep = obs_log_info['deployment'].loc[obs_log_info['deployment']['OBS Name'] == obs_identifier]
            rec = obs_log_info['recovery'].loc[obs_log_info['recovery']['OBS Name'] == obs_identifier]
        else:
            id_columns = ['Station', 'OBS Name', 'OBS ID']
            for col in id_columns:
                if obs_identifier in obs_log_info['basic'][col].values:
                    base_meta = obs_log_info['basic'].loc[obs_log_info['basic'][col] == obs_identifier]
                    dep = obs_log_info['deployment'].loc[obs_log_info['deployment'][col] == obs_identifier]
                    rec = obs_log_info['recovery'].loc[obs_log_info['recovery'][col] == obs_identifier]
                    break

        if base_meta is None or base_meta.empty:
            raise IndexError('OBS {0} not found in provided metadata.'.format(obs_identifier))
        if deploy_start is not None:
            base_meta = base_meta.loc[(base_meta['Launch Date/Time (UTC)'] >= deploy_start) & (base_meta['Launch Date/Time (UTC)'] < deploy_start + timedelta(days=1))]
        if base_meta.shape[0] > 1:
            raise IndexError('Multiple entries found for OBS {0} in provided metadata. Please use a unique identifier or provide start date.'.format(obs_identifier))

        if deploy_start is not None:
            meta_start = min(base_meta['Launch Date/Time (UTC)'].values[0],
                             base_meta['Date/Time on Seafloor (UTC)'].values[0])
            if meta_start.date() != deploy_start.date():
                g_log.warning(
                    "Start time in metadata file ({0}) is different from runtime/config argument ({1}).".format(
                        meta_start.strftime('%Y-%m-%d'), deploy_start.strftime('%Y-%m-%d')))

        channel_map = None
        if args.channel_map:
            ch_map = args.channel_map
        else:
            ch_map = config.get('dataset', 'channelmap', fallback=None)
        if ch_map is not None:
            full_config['dataset']['channelmap'] = ch_map
            if relpath:
                channel_map = nf.io.read_channel_map(os.path.join(base_dir, ch_map))
            else:
                channel_map = nf.io.read_channel_map(ch_map)
        else:
            g_log.info("No channel map provided. Channel IDs will be processed as they appear in the raw data files.")

        metadata_file = None
        if args.metadata_file:
            meta_file = args.metadata_file
        else:
            meta_file = config.get('dataset', 'metadata', fallback=None)
        if meta_file is not None:
            full_config['dataset']['metadata'] = meta_file
            if relpath:
                metadata_file = os.path.join(base_dir, meta_file)
            else:
                metadata_file = os.path.abspath(os.path.expanduser(os.path.expandvars(meta_file)))

        # Read project metadata JSON file
        project_meta = None
        # TODO: Replace with ST integration once we have an instance running
        # Search data_dir for project JSON (should have channel descriptions)
        if args.extra_meta:
            json_file = args.extra_meta
        else:
            json_file = config.get('dataset', 'extra_meta', fallback=None)
        if json_file is not None:
            full_config['dataset']['extra_meta'] = json_file
            if relpath:
                project_json = os.path.join(base_dir, json_file)
            else:
                project_json = os.path.abspath(os.path.expanduser(os.path.expandvars(json_file)))
        else:
            project_json = os.path.join(base_dir, 'project_info.json')
            if relpath:
                full_config['dataset']['extra_meta'] = 'project_info.json'
            else:
                full_config['dataset']['extra_meta'] = project_json
        if os.path.isfile(project_json):
            g_log.info("Reading project metadata from {0}...".format(project_json))
            pj = open(project_json)
            project_meta = json.load(pj)
        else:
            g_log.info("No project metadata JSON found at {0}".format(project_json))
            full_config.remove_option('dataset', 'extra_meta')

        station_meta = None
        if project_meta is not None:
            try:
                station_meta = list(filter(lambda x: x['name'] == base_meta['Station'].values[0], project_meta['stations']))[0]
            except (KeyError, IndexError):
                g_log.info("No matching station information found in project metadata JSON.")
            finally:
                project_meta['this_deployment'] = {}
        if station_meta is not None:
            if 'deployments' in station_meta:
                if len(station_meta['deployments']) > 1:
                    if deploy_start is not None:
                        deployment = list(filter(lambda x: x['start_date'] == deploy_start.strftime('%Y-%m-%d'), station_meta['deployments']))
                        if len(deployment) > 0:
                            project_meta['this_deployment'] = deployment[0]
                        else:
                            g_log.info("No deployment found for station {0} with start date {1}.".format(obs_identifier, deploy_start.strftime('%Y-%m-%d')))
                    else:
                        g_log.error("Multiple matching deployments found. Please specify start date.")
                else:
                    project_meta['this_deployment'] = station_meta['deployments'][0]
            else:
                deployment = station_meta
                project_meta['this_deployment'] = deployment

        for flag, key in zip([args.function_check, args.detrend_seis, args.skip_backup, args.use_existing_plots, args.debug], ['function_check', 'detrend_seismic', 'skip_backup', 'use_existing_plots', 'debug']):
            config_flag = config.getboolean('dataset', key, fallback=False)
            # only overwrite existing flags if CL arguments are present and different from config
            if flag and not config_flag:
                full_config['dataset'][key] = str(flag)

        if args.network_id:
            network = args.network_id
        else:
            network = config.get('dataset', 'network', fallback='XX')
        full_config['dataset']['network'] = network

        # Gather some basic information for report
        report_kwargs = {
            'today': datetime.now().strftime('%Y-%m-%d'),
        }
        if args.project_name:
            report_kwargs['projectName'] = args.project_name
        elif config.get('dataset', 'projectname', fallback=None) is not None:
            report_kwargs['projectName'] = config.get('dataset', 'projectname')
        elif project_meta is not None:
            report_kwargs['projectName'] = project_meta['project']
        else:
            report_kwargs['projectName'] = 'Test Recording'
        full_config['dataset']['projectname'] = report_kwargs['projectName']
        if project_meta['common_intro']:
            report_kwargs['intro_pt1'] = project_meta['common_intro']
        report_kwargs['stationName'] = base_meta['Station'].values[0]
        report_kwargs['obsName'] = base_meta['OBS Name'].values[0]
        report_kwargs['obsId'] = base_meta['OBS ID'].values[0]
        report_kwargs['latitude'] = base_meta['Deployed Latitude'].values[0]
        report_kwargs['longitude'] = base_meta['Deployed Longitude'].values[0]
        report_kwargs['waterDepth'] = base_meta['Water Depth (m)'].values[0]
        report_kwargs['deployed'] = pd.to_datetime(base_meta['Launch Date/Time (UTC)'].values[0])
        report_kwargs['deployComments'] = dep['Comments'].values[0]
        report_kwargs['recovered'] = pd.to_datetime(base_meta['Recovery Date/Time (UTC)'].values[0])
        report_kwargs['recoverComments'] = rec['Comments'].values[0]
        report_kwargs['deploymentDays'] = (report_kwargs['recovered'] - report_kwargs['deployed']) / timedelta(days=1)
        report_kwargs['clockDrift'] = base_meta['Clock Offset on Deck (ms)'].values[0]
        report_kwargs['batteryLevel'] = rec['Battery SOC (%)'].values[0]
        if 'qc_intro' in project_meta['this_deployment']:
            report_kwargs['introText'] = project_meta['this_deployment']['qc_intro']
        report_kwargs['psdWindowSecs'] = config.getint('seismic', 'window_length')
        report_kwargs['psdOverlapPercent'] = config.getint('seismic', 'overlap_percent')

        setup_time = timeit.default_timer()
        g_log.info("Time spent parsing arguments and preparing to process data: {0} seconds".format(setup_time - t0))

        # Process data files to apply clock drift correction and update metadata
        process(data_dir, base_meta, network, full_config, output_dir=output_dir, metadata=metadata_file,
                channel_map=channel_map, project_meta=project_meta, flags_from_config=True, **report_kwargs)

        proc_time = timeit.default_timer()
        g_log.info("Time spent processing data: {0} seconds".format(proc_time - setup_time))

        g_log.info("Processing complete!")
        end_time = datetime.now()
        run_time = timeit.default_timer() - t0
        if run_time < 60:
            g_log.info("Total run time: {0} seconds".format(run_time))
        elif run_time < 3600:
            g_log.info("Total run time: {0} seconds ({1} minutes)".format(run_time, run_time/60))
        elif run_time < 3600*24:
            g_log.info("Total run time: {0} seconds ({1} hours)".format(run_time, run_time/3600))
        else:
            g_log.info("Total run time: {0} seconds ({1} days)".format(run_time, run_time/3600/24))

        logger.close_logs()

        print(gc.get_stats())
    except Exception as e:
        print(traceback.print_exc())
        parser.print_help()
        print(gc.get_stats())
        exit(1)
