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
import psutil
import pypandoc
import re
import timeit
import traceback
import warnings

from obspy.io.mseed.util import get_start_and_end_time
from obspy.io.stationxml.core import validate_stationxml
from obspy.signal.trigger import trigger_onset, plot_trigger

from ioos_qc import qartod

import nfsi_obs as nf
from utilities import config_handler, logger, ReportGenerator, time_period_string

gc.set_debug(gc.DEBUG_UNCOLLECTABLE)
feature_test = False    # set to True to test new features

current_dir = os.path.dirname(os.path.abspath(__file__))
# Ensure resource directory exists
resource_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'resource/OBSDataPipeline')
if not os.path.isdir(resource_dir):
    os.makedirs(resource_dir)


def process(data_dir, obs_log, network_id, config, output_dir=None, metadata=None, channel_map=None, project_meta=None,
            full=True, detrend=False, cmap=None, use_existing_plots=False, parallel=False, max_proc=None,
            flags_from_config=False, **kwargs):
    """
    Extra keyword arguments are included as report parameters (must match variables in template file).
    """
    timing_points = []
    error_count = 0
    timing_points.append(timeit.default_timer())
    g_log.info("start")

    debug_info = {'timing': {}}

    # Get flags from config if necessary
    if flags_from_config:
        # False fallback value will default to same values as function definition
        full = not config.getboolean('dataset', 'function_check', fallback=False)
        detrend = config.getboolean('dataset', 'detrend_seismic', fallback=False)
        use_existing_plots = config.getboolean('dataset', 'use_existing_plots', fallback=False)
        parallel = config.getboolean('dataset', 'parallel', fallback=False)

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
    if not config.has_section('seismic'):
        config.add_section('seismic')
    spec_win = config.getint('seismic', 'spectrogram_window', fallback=60)
    if max_proc is None:
        max_proc = config.getint('seismic', 'max_processes', fallback=None)
    for key, val in zip(['window_length', 'overlap_percent', 'spectrogram_window', 'max_processes'],
                        [win_len, overlap, spec_win, max_proc]):
        config['seismic'][key] = str(val)

    timing_points.append(timeit.default_timer())
    g_log.debug("Basic processing setup time: {0} seconds".format((timing_points[-1] - timing_points[-2])))
    debug_info['timing']['base_setup'] = timing_points[-1] - timing_points[-2]

    # Start/end of time period to analyze: (1) on seafloor, (2) off-ship, (3) deployment start/end, (4) project start/end
    # TODO: Only apply this to external or seismic channels? Analyze full battery/power, for example.
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

    if data_start > data_end:
        # TODO: Better fallback handling here, step out until good window found.
        g_log.warning("Start time {} greater than end time {} for data. Ignoring start/end times.".format(
            data_start.strftime('%Y-%m-%d %H:%M:%S'), data_end.strftime('%Y-%m-%d %H:%M:%S')))
        data_start, data_end = None, None

    report_params['seafloorDays'] = '{:.3f}'.format((data_end - data_start) / 60 / 60 / 24)
    g_log.debug('Time at seafloor: {} days'.format(report_params['seafloorDays']))

    # Read station metadata file (dataless SEED or StationXML)
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

    timing_points.append(timeit.default_timer())
    g_log.debug("Time spent reading station metadata file: {0} seconds".format((timing_points[-1] - timing_points[-2])))
    debug_info['timing']['station_meta'] = timing_points[-1] - timing_points[-2]

    # Find data files
    raw_files = glob(os.path.join(data_dir, '**/*.mseed'), recursive=True)
    try:
        raw_files.remove(os.path.join(data_dir, 'calculated_current.mseed'))
    except ValueError:
        pass

    g_log.info("Found {0} miniSEED file(s) in data directory and sub-folders".format(len(raw_files)))
    debug_info['num_files'] = len(raw_files)
    if output_dir is None:
        output_dir = data_dir

    # Write QC processing configuration to file
    with open(os.path.join(output_dir, 'QC_config.ini'), 'w') as configfile:
        config.write(configfile)

    timing_points.append(timeit.default_timer())
    g_log.debug("Time spent searching for data files: {0} seconds".format((timing_points[-1] - timing_points[-2])))
    debug_info['timing']['file_search'] = timing_points[-1] - timing_points[-2]

    # label files by channel name
    labels = []
    for rf in raw_files:
        file_name = re.split(r'[/\\]', rf)[-1]
        ch_name = file_name.split('_')[1]
        labels.append({'channel': ch_name, 'path': rf})
    labeled_files = pd.DataFrame(labels)
    g_log.info("Files contain data for {0} unique set(s) of channels".format(len(np.unique(labeled_files['channel'].values))))
    debug_info['num_channels'] = len(np.unique(labeled_files['channel'].values))

    timing_points.append(timeit.default_timer())
    g_log.debug("Time spent sorting and labeling data files: {0} seconds".format((timing_points[-1] - timing_points[-2])))
    debug_info['timing']['file_sort'] = timing_points[-1] - timing_points[-2]

    # Initialize arrays for saving stats
    all_channels = []
    all_gaps = []
    power_stats = pd.DataFrame()
    avg_power = []
    voltage_stats = []
    power_data = {}

    timing_points.append(timeit.default_timer())
    g_log.debug("Time spent setting up arrays for stats: {0} seconds".format((timing_points[-1] - timing_points[-2])))
    debug_info['timing']['array_setup'] = timing_points[-1] - timing_points[-2]

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
        proc_timing = [timeit.default_timer()]
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
                if np.any(filetimes < datetime(2021,9,1)):
                    g_log.warning('Some data timestamps prior to 2021-09-01 (invalid). Using start/end times from OBS log instead.')
                    files_start = data_start
                    files_end = data_end
                else:
                    files_start = min(filetimes[:, 0])
                    files_end = max(filetimes[:, 1])

                startend = timeit.default_timer()
                debug_info['timing']['long_series_check'] += startend - proc_timing[-1]

                plot_len = None
                """
                if (data_end - data_start) < timedelta(days=29).total_seconds():
                    # 4 weeks or less to analyze, make 10-day plots
                    plot_len = 10
                elif (files_end - files_start) < timedelta(days=31).total_seconds():
                """
                if (data_end - data_start) < timedelta(days=31).total_seconds() or (files_end - files_start) < timedelta(days=31).total_seconds():
                    # Less than 1 month of data to analyze, just make one plot
                    plot_end = min(files_end, data_end) + 24 * 60 * 60
                    plot_start = max(files_start, data_start)
                    plot_days = plot_end.date - plot_start.date
                    plot_len = np.round(plot_days.total_seconds() / 60 / 60 / 24)

                # Ensure files are sorted alphabetically (should be same as chronological order)
                data_files = sorted(files['path'].values)
                trace_info, gaps, buff_time = nf.plotting.buffer_seismic_data(data_files, output_dir, g_log,
                                                                              network_id, station_info, channel_map,
                                                                              project_meta, win_len, spec_win, overlap,
                                                                              plot_length=plot_len, start=data_start,
                                                                              end=data_end, spec_cmap=cmap, detrend=detrend,
                                                                              use_existing_plots=use_existing_plots,
                                                                              parallel=parallel, max_processes=max_proc)
                for key in buff_time:
                    if key in debug_info['timing']:
                        debug_info['timing'][key] += buff_time[key]
                    else:
                        debug_info['timing'][key] = buff_time[key]

                channel_type = trace_info['channelType']

                # TODO: Move string formatting into ReportGenerator class
                ch_start = max(files_start, data_start)
                ch_end = min(files_end, data_end)
                all_channels.append({
                    'id': trace_info['seedID'],
                    'start': ch_start.datetime.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                    'end': ch_end.datetime.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                    'sampling': '{:.1f}'.format(trace_info['samplingRate'])
                })
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

                proc_timing.append(timeit.default_timer())
                g_log.debug("Time spent reading data file(s): {0} seconds".format((proc_timing[-1] - proc_timing[-2])))
                debug_info['timing']['file_read'] += proc_timing[-1] - proc_timing[-2]

                # Cut data to time period of interest (if start/end times provided)
                if (data_start is not None) or (data_end is not None):
                    # This shouldn't change `data` if there is no data to cut out
                    data.trim(data_start, data_end, nearest_sample=False)

                proc_timing.append(timeit.default_timer())
                g_log.debug("Time spent cutting to period of interest: {0} seconds".format((proc_timing[-1] - proc_timing[-2])))
                debug_info['timing']['time_cut'] += proc_timing[-1] - proc_timing[-2]

                # Populate metadata from other files as necessary
                data = nf.metadata.update_metadata(data, network_id, g_log, station_info, channel_map, project_meta)
                data.merge()
                print(data)

                proc_timing.append(timeit.default_timer())
                g_log.debug("Time spent applying metadata: {0} seconds".format((proc_timing[-1] - proc_timing[-2])))
                debug_info['timing']['apply_meta'] += proc_timing[-1] - proc_timing[-2]

                # Perform QC
                # TODO: Combine single and multi-channel cases to simplify code (no real reason to separate) -> TEST multi-channel

                # Gap test
                gaps = data.get_gaps()
                all_gaps.extend(gaps)
                if len(gaps) > 0:
                    g_log.info('Found {0} gap(s) or overlap(s) in recorded data'.format(len(gaps)))
                    data.print_gaps()

                proc_timing.append(timeit.default_timer())
                g_log.debug("Time spent for gap test: {0} seconds".format((proc_timing[-1] - proc_timing[-2])))
                debug_info['timing']['gap_test'] += proc_timing[-1] - proc_timing[-2]

                for tr in data:
                    tr_timing = [timeit.default_timer()]

                    # Channel type determines what analysis gets run on this trace
                    channel_type = nf.metadata.get_channel_type(tr.meta.channel)

                    tr_timing.append(timeit.default_timer())
                    g_log.debug("Time spent assigning to channel group: {0} seconds".format((tr_timing[-1] - tr_timing[-2])))
                    debug_info['timing']['group_assign'] += tr_timing[-1] - tr_timing[-2]

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
                    spec_lim = [None, None]
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
                        if 'spec_min' in channel_info:
                            spec_lim[0] = float(channel_info['spec_min'])
                        if 'spec_max' in channel_info:
                            spec_lim[1] = float(channel_info['spec_max'])

                    tr_timing.append(timeit.default_timer())
                    g_log.debug("Time spent with other metadata admin: {0} seconds".format((tr_timing[-1] - tr_timing[-2])))
                    debug_info['timing']['meta_admin'] += tr_timing[-1] - tr_timing[-2]

                    # Noise level QC steps (seismic channels and hydrophone) -> if channel code == "CHx" or "HDF"
                    if channel_type == 'seismic':
                        # Time series plot (applies instrument sensitivity in-place if response present in tr.meta)
                        trace_info['traceLoc'] = nf.plotting.trace_plot(tr, output_dir, dmin, dmax, qc_config,
                                                                        use_existing_plots)

                        tr_timing.append(timeit.default_timer())
                        g_log.debug("Time spent plotting trace: {0} seconds".format((tr_timing[-1] - tr_timing[-2])))
                        debug_info['timing']['trace_plot'] += tr_timing[-1] - tr_timing[-2]

                        for metaKey, reportKey in zip(['azimuth', 'dip'], ['azimuth', 'dip']):
                            if hasattr(tr.meta, metaKey):
                                trace_info[reportKey] = tr.meta[metaKey]

                        # Detrend seismic data (RMS linear fit)
                        if detrend:
                            tr.detrend('linear')
                            demean_data_plot = os.path.join(output_dir, 'demean_{0}.png'.format(tr.id))
                            if not (use_existing_plots and os.path.isfile(demean_data_plot)):
                                data.plot(outfile=demean_data_plot)

                        start_plots = timeit.default_timer()    # TODO
                        # TODO: Combine spectrogram and PSD creation to save runtime and memory (like when buffering)
                        # Spectrogram
                        trace_info['specLoc'] = [{
                            'image': nf.plotting.spectrogram(tr, output_dir, spec_win, overlap, use_existing_plots),
                            'start': tr.stats.starttime.strftime('%Y-%m-%d'),
                            'end': tr.stats.endtime.strftime('%Y-%m-%d')
                        }]
                        done_spec = timeit.default_timer()  # TODO
                        debug_info['timing']['spec_plot'] += done_spec - start_plots

                        # Plot PSDs of data
                        trace_info['psdLoc'] = [{
                            'image': nf.plotting.psd_plot(tr, output_dir, win_len, overlap, use_existing_plots),
                            'start': tr.stats.starttime.strftime('%Y-%m-%d'),
                            'end': tr.stats.endtime.strftime('%Y-%m-%d')
                        }]
                        done_psd = timeit.default_timer()   # TODO
                        debug_info['timing']['psd_plot'] += done_psd - done_spec

                        if full:
                            # TODO: Decide if the same operations are appropriate for the hydrophone data or not
                            # TODO: Save hourly PSDs
                            # TODO: Average PSD value at 0.2 Hz (save out for comparison with other sensors in the same network)
                            # TODO: Linearity of PSD curves
                            g_log.warning("Full QC of seismic noise not yet implemented")

                    else:
                        if channel_type == 'ocean':
                            # Thresholds determined based on raw counts, so needs to happen before sensitivity is removed by plotting function
                            # TODO: Check if this will work with APG data if we ever collect any
                            if re.match(r'[A-Z]DO', tr.meta.channel):
                                averaging_window = 15 * 60 * tr.stats.sampling_rate
                                outlier_cutoff = 1
                            elif re.match(r'[A-Z]KO', tr.meta.channel):
                                averaging_window = [7 * 60 * tr.stats.sampling_rate, 30 * 60 * tr.stats.sampling_rate]
                                outlier_cutoff = 2.05
                            else:
                                g_log.info('Unrecognized channel type {}. Using default despiking thresholds.'.format(
                                    tr.meta.channel))
                                averaging_window = 10 * 60 * tr.stats.sampling_rate
                                outlier_cutoff = 1

                            despiked = nf.remove_write_spikes(tr, delta=outlier_cutoff, span=averaging_window,
                                                              savedf=False, dfpath=os.path.join(output_dir,
                                                                                  '{}_despiking_info.csv'.format(tr.id)))
                            g_log.debug('Despiked data type: {}'.format(despiked.data.dtype.type))
                            try:
                                despiked.write(os.path.join(output_dir, '{}_despiked.mseed'.format(tr.id)), format='MSEED', encoding='STEIM2')
                            except Exception as e:
                                g_log.error('Error writing despiked data to file!')
                                g_log.error(traceback.format_exc())

                            trace_info['despikedPlot'] = nf.plotting.trace_plot(despiked, output_dir, dmin, dmax, qc_config, use_existing_plots)

                            # Calculate rolling average of despiked data
                            trace_length = tr.meta.endtime - tr.meta.starttime
                            units = nf.metadata.get_units(tr)
                            stat_window = 1
                            vert_label = 'Average Value'
                            if re.match(r'[A-Z]DO', tr.meta.channel):
                                # External pressure
                                g_log.info('Seafloor pressure ({}): mean {:.3f}, min {:.3f}, max {:.3f}, stdev {:.3f}'.format(units, np.mean(despiked.data), np.min(despiked.data), np.max(despiked.data), np.std(despiked.data)))
                                vert_label = 'Average Seafloor Pressure ({})'.format(units)
                                # 3-day rolling window of average seafloor pressure (uses 3 lunar days: 24 hours, 50 minutes)
                                if trace_length > 5 * 24 * 60 * 60:
                                    stat_window = 3 * (24 * 60 + 50) * 60
                                else:
                                    stat_window = trace_length * 0.6
                            if re.match(r'[A-Z]KO', tr.meta.channel):
                                # External temperature
                                g_log.info('Seafloor temperature ({}): mean {:.3f}, min {:.3f}, max {:.3f}, stdev {:.3f}'.format(units, np.mean(despiked.data), np.min(despiked.data), np.max(despiked.data), np.std(despiked.data)))
                                vert_label = 'Average Seafloor Temperature ({})'.format(units)
                                # 3-day rolling window of average seafloor pressure (uses 3 lunar days: 24 hours, 50 minutes)
                                if trace_length > 5 * 24 * 60 * 60:
                                    stat_window = 3 * 24 * 60 * 60
                                else:
                                    stat_window = trace_length * 0.6

                            trace_info['window_str'] = time_period_string(stat_window)
                            roll_stats = nf.rolling_window_stats(despiked, window_length=stat_window, window_offset=stat_window/3, full=True)

                            # Save rolling window statistics to CSV file
                            ch_stats = pd.DataFrame(roll_stats, columns=['Start', 'End', 'Center', 'Min', 'Max', 'Avg', 'Gradient', 'R2_coef', 'Days_Deployed'])
                            ch_stats.to_csv(os.path.join(output_dir, '{}_rolling_stats_{}_{}.csv'.format(tr.id, pd.to_datetime(ch_stats['Start'].min()).strftime('%Y-%m-%d'), pd.to_datetime(ch_stats['End'].max()).strftime('%Y-%m-%d'))))

                            # Plot rolling mean and add to report
                            # TODO: Make x-lims start and end dates of data
                            roll_plot = os.path.join(output_dir, '{}_mean.png'.format(tr.id))
                            fig, ax = plt.subplots(1, 1, figsize=[8, 2.5])
                            ch_stats.plot(x='Center', y='Avg', kind='line', ax=ax, xlabel='Date/Time', ylabel=vert_label, legend=False)
                            ax.grid(True, ls=':')
                            fig.tight_layout()
                            fig.savefig(roll_plot)
                            plt.close(fig)
                            trace_info['rollPlot'] = roll_plot

                            # Add average seafloor readings to report
                            if re.match(r'[A-Z]DO', tr.meta.channel):
                                report_params['meanPressure'] = '{:.0f}'.format(np.mean(despiked.data))
                            elif re.match(r'[A-Z]KO', tr.meta.channel):
                                report_params['meanTemperature'] = '{:.3f}'.format(np.mean(despiked.data))

                            tr_timing.append(timeit.default_timer())
                            g_log.debug("Time spent despiking trace: {} seconds".format((tr_timing[-1] - tr_timing[-2])))

                        # Time series plot (applies instrument sensitivity in-place if response present in tr.meta)
                        trace_info['traceLoc'] = nf.plotting.trace_plot(tr, output_dir, dmin, dmax, qc_config, use_existing_plots)

                        tr_timing.append(timeit.default_timer())
                        g_log.debug("Time spent plotting trace: {0} seconds".format((tr_timing[-1] - tr_timing[-2])))
                        debug_info['timing']['trace_plot'] += tr_timing[-1] - tr_timing[-2]

                        # Analysis of auxiliary data
                        # TODO: maybe smooth out state-of-health channels? or come up with some way to automatically QC them for anomalous sections
                        start_tran = timeit.default_timer()     # TODO
                        if qc_config is not None:
                            if 'qartod' in qc_config:
                                if 'gross_range_test' in qc_config['qartod']:
                                    range_check = qartod.gross_range_test(tr.data, **qc_config['qartod']['gross_range_test'])
                                    if np.any(range_check > 1):
                                        num_sus = np.sum(range_check == 3)
                                        num_fail = np.sum(range_check == 4)
                                        g_log.info('Channel {0} has suspect values at {1} sample(s) ({3:.1%}) and failing values at {2} sample(s) ({4:.1%})'.format(tr.id, num_sus, num_fail, num_sus / len(range_check), num_fail / len(range_check)))
                                        # TODO: Add fail/suspect stats to report as well as log printout
                                    #check_trace = obspy.Trace(range_check, header=tr.stats)
                                    #trace_info['qcPlotLoc'] = nf.plotting.qartod_plot(check_trace, output_dir, 'gross_range_check', use_existing_plots)

                                if feature_test:
                                    # Implement new features to be tested here
                                    boo = True

                        done_qartod = timeit.default_timer()    # TODO
                        debug_info['timing']['qartod'] += done_qartod - start_tran

                        if channel_type == 'power':
                            # Voltage and power consumption channels
                            trace_length = tr.meta.endtime - tr.meta.starttime

                            if tr.meta.channel == 'LE3':
                                power_data[tr.meta.channel] = tr.copy()
                                # TODO: Show power consumption as positive rather than negative (as recorded)
                                # Power consumption
                                report_params['meanPower'] = '{:.3f}'.format(np.mean(tr.data))

                                # 3-day rolling window of average power consumption
                                if trace_length > 5*24*60*60:
                                    avg_power.extend(nf.rolling_window_stats(tr, full=False))
                                else:
                                    stat_window = trace_length * 0.6
                                    if 'battery_stats_window' not in report_params:
                                        report_params['battery_stats_window'] = stat_window
                                    avg_power.extend(nf.rolling_window_stats(tr, window_length=stat_window, window_offset=stat_window/3, full=False))

                                # TODO: Get times of data writes (spikes 45 minutes apart)
                                if feature_test and (qc_config is not None) and ('qartod' in qc_config) and ('spike_test' in qc_config['qartod']):
                                    # This may or may not work and/or be useful
                                    spikes = qartod.spike_test(tr.data, **qc_config['qartod']['spike_test'])
                            if tr.meta.channel == 'ME4':
                                power_data[tr.meta.channel] = tr.copy()
                                # Battery voltage
                                # 3-day rolling window for stats
                                if trace_length > 5*24*60*60:
                                    voltage_stats.extend(nf.rolling_window_stats(tr, full=True))
                                else:
                                    stat_window = trace_length * 0.6
                                    if 'battery_stats_window' not in report_params:
                                        report_params['battery_stats_window'] = stat_window
                                    voltage_stats.extend(nf.rolling_window_stats(tr, window_length=stat_window, window_offset=stat_window/3, full=True))

                        done_power = timeit.default_timer()     # TODO
                        debug_info['timing']['power_analysis'] += done_power - done_qartod

                        # Check humidity data for blips (tested for Ischia 2023 deployment)
                        if re.match(r'[A-Z]I[IO]', tr.meta.channel):
                            hum = tr.copy()
                            try:
                                hum.detrend('demean')
                                hum.detrend('linear')
                            except Exception:
                                pass

                            try:
                                hum.trigger('classicstalta', sta=60*60*3, lta=60*60*24)
                                triggers = trigger_onset(hum.data, 3, 1.5)
                            except Exception as ex:
                                # Raised if humidity data shorter than LTA window (1 day)
                                print(ex)
                                triggers = []

                            if len(triggers) > 0:
                                plot_trigger(tr, hum.data, 3, 1.5, show=False)
                                fig = plt.gcf()
                                fig.savefig(os.path.join(output_dir, 'triggered_{0}.png'.format(tr.id)))

                                trig_secs = triggers * hum.stats.delta
                                trig_times = [[hum.stats.starttime + float(y) for y in x] for x in trig_secs]

                                hum_filt = tr.copy()
                                hfs = hum_filt.split()
                                hfs.filter('lowpass', freq=1./(60*30))
                                hfs.merge()
                                hum_filt = hfs.traces[0]
                                humidity_blips = []
                                for tt in trig_times:
                                    ht = hum_filt.slice(tt[0], tt[1], nearest_sample=False)
                                    back = hum_filt.slice(tt[0] - 24 * 60 * 60, tt[0], nearest_sample=False)
                                    dev = np.nan
                                    try:
                                        bm = back.data.mean()
                                        hx = ht.max()
                                        hn = ht.data.min()
                                        if abs(hx - bm) > abs(hn - bm):
                                            dev = hx - bm
                                        else:
                                            dev = hn - bm
                                    except ValueError:
                                        g_log.warning('Error encountered determining stats for trigger {}'.format(tt[0].strftime('%Y-%m-%d %H:%M:%S')))
                                        g_log.error(traceback.format_exc())

                                    # TODO: Move string formatting into ReportGenerator class
                                    humidity_blips.append({
                                        'start': tt[0].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                                        'end': tt[1].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                                        'sec': '{:.1f}'.format(tt[1] - tt[0]),
                                        'dev': '{:.3f}'.format(dev)
                                    })

                                if 'humid' in report_params:
                                    g_log.warn('Multiple humidity channels processed for instrument {}. Only first '
                                               'plot will be included in report.'.format(obs_identifier))
                                    report_params['humid']['triggers'].extend(humidity_blips)
                                else:
                                    report_params['humid'] = {
                                        'ch': tr.id,
                                        'plot': os.path.join(output_dir, 'triggered_{0}.png'.format(tr.id)),
                                        'triggers': humidity_blips
                                    }

                        # TODO: Analysis of other state-of-health variables?
                        # TODO: Down-sample external pressure and temperature data (plot and save as netCDF)

                    tr_timing.append(timeit.default_timer())
                    g_log.debug("Time spent performing trace-specific analysis: {0} seconds".format((tr_timing[-1] - tr_timing[-2])))
                    debug_info['timing']['trace_analysis'] += tr_timing[-1] - tr_timing[-2]

                    # Summary statistics
                    units = nf.metadata.get_units(tr)
                    print("{0} | {1} - {2} | {3} | Average {4:.3f} {5}".format(
                        tr.id,
                        tr.meta.starttime.strftime('%Y-%m-%d %H:%M:%S.%f'),
                        tr.meta.endtime.strftime('%Y-%m-%d %H:%M:%S.%f'),
                        trace_info['channelName'],
                        np.mean(tr.data),
                        units
                    ))

                    # TODO: Move string formatting into ReportGenerator class
                    all_channels.append({
                        'id': tr.id,
                        'start': tr.meta.starttime.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                        'end': tr.meta.endtime.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                        'sampling': '{:.1f}'.format(tr.meta.sampling_rate)
                    })

                    report_params[channel_type + '_channels'].append(trace_info)
        except Exception as e:
            error_count += 1
            msg = str(e)
            g_log.error(traceback.format_exc())
            g_log.error(msg)

    timing_points.append(timeit.default_timer())
    g_log.debug("Time spent processing data files: {0} seconds".format((timing_points[-1] - timing_points[-2])))
    debug_info['timing']['all_proc'] = timing_points[-1] - timing_points[-2]

    # TODO: Add expected hibernation date (once calculated properly) to report summary
    # TODO: Column formatting for report summary page (easier to read?)

    # Add list of all channels for report
    if len(all_channels) > 0:
        report_params['channelList'] = sorted(all_channels, key=lambda p: p['id'])

    # Parse gap information for report
    if len(all_gaps) > 0:
        unique_gaps = {}
        for gap in all_gaps:
            gap_key = '_'.join(['.'.join(gap[0:4]), gap[4].strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3], gap[5].strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3]])
            unique_gaps.update({gap_key: gap})

        gap_list = []
        # TODO: Move string formatting into ReportGenerator class
        for gap in unique_gaps.values():
            gap_list.append({
                'id': '.'.join(gap[0:4]),
                'start': gap[4].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                'end': gap[5].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                'sec': '{:.3f}'.format(gap[6]),
                'samp': gap[7]
            })

        report_params['gapList'] = sorted(gap_list, key=lambda p: p['start'])
        gaps_df = pd.DataFrame(gap_list)
        gaps_df.to_csv(os.path.join(output_dir, 'gaps_{}.csv'.format(report_params['stationName'])))

    timing_points.append(timeit.default_timer())
    g_log.debug("Time spent formatting gap information: {0} seconds".format((timing_points[-1] - timing_points[-2])))
    debug_info['timing']['gap_format'] = timing_points[-1] - timing_points[-2]

    # Combine voltage/power statistics and make plots
    if len(avg_power) > 0 or len(voltage_stats) > 0:
        pwr = pd.DataFrame(avg_power, columns=['Start', 'End', 'Center', 'Min_Power', 'Max_Power', 'Avg_Power'])
        pwr.set_index('Start', drop=False)
        vlt = pd.DataFrame(voltage_stats, columns=['Start', 'End', 'Center', 'Min_Volts', 'Max_Volts', 'Avg_Volts', 'Gradient', 'R2_coef', 'Days_Deployed'])
        vlt.set_index('Start', drop=True)

        power_stats['Start'] = pwr['Start']
        power_stats['End'] = pwr['End']
        power_stats['Voltage_Min'] = vlt['Min_Volts']
        power_stats['Voltage_Max'] = vlt['Max_Volts']
        power_stats['Voltage_Mean'] = vlt['Avg_Volts']
        power_stats['Voltage_gradient'] = vlt['Gradient']
        power_stats['Voltage_fit'] = vlt['R2_coef']
        power_stats['Power_Mean'] = pwr['Avg_Power']
        power_stats = power_stats.assign(Aquarius_ID=report_params['obsId'])
        # TODO: Add deployment ID from Sensor Tracker integration (for combining stats with other deployments)
        power_stats['Plot_Time'] = pwr['Center']
        power_stats['Days_Deployed'] = vlt['Days_Deployed']

        # Save statistics to CSV for further analysis
        csv_name = 'voltage_power_stats_{0}_{1}_{2}_{3}.csv'.format(report_params['stationName'], report_params['obsId'],
                                                                pd.to_datetime(power_stats['Start'].min()).strftime('%Y-%m-%d'),
                                                                pd.to_datetime(power_stats['End'].max()).strftime('%Y-%m-%d'))
        power_stats.to_csv(os.path.join(output_dir, csv_name))

        # TODO: Make x-lims start and end dates of data
        # Average power vs time
        avgpow_plot = os.path.join(output_dir, 'power_mean_{0}.png'.format(obs_log['OBS ID'].values[0]))
        if not (use_existing_plots and os.path.isfile(avgpow_plot)):
            fig, ax = plt.subplots(1, 1, figsize=[8, 2.5])
            power_stats.plot(x='Plot_Time', y='Power_Mean', kind='line', ax=ax, xlabel='Date/Time', ylabel='Average Power Consumption (W)', legend=False)
            ax.grid(True, ls=':')
            fig.tight_layout()
            fig.savefig(avgpow_plot)
            plt.close(fig)

        # Average voltage vs time
        avgvlt_plot = os.path.join(output_dir, 'voltage_mean_{0}.png'.format(obs_log['OBS ID'].values[0]))
        if not (use_existing_plots and os.path.isfile(avgvlt_plot)):
            fig, ax = plt.subplots(1, 1, figsize=[8, 2.5])
            power_stats.plot(x='Plot_Time', y='Voltage_Mean', kind='line', ax=ax, xlabel='Date/Time', ylabel='Average Voltage (V)', legend=False)
            ax.grid(True, ls=':')
            fig.tight_layout()
            fig.savefig(avgvlt_plot)
            plt.close(fig)

        # Voltage gradient vs time
        vltgrd_plot = os.path.join(output_dir, 'voltage_gradient_{0}.png'.format(obs_log['OBS ID'].values[0]))
        if not (use_existing_plots and os.path.isfile(vltgrd_plot)):
            fig, ax = plt.subplots(1, 1, figsize=[8, 2.5])
            power_stats.plot(x='Plot_Time', y='Voltage_gradient', kind='line', ax=ax, xlabel='Date/Time', ylabel='Voltage Gradient (mV/day)', legend=False)
            ax.set_ylim(ymax=0)
            ax.grid(True, ls=':')
            fig.tight_layout()
            fig.savefig(vltgrd_plot)
            plt.close(fig)

        report_params['batteryStats'] = {}
        report_params['batteryStats']['meanPowerPlot'] = avgpow_plot
        report_params['batteryStats']['meanVoltPlot'] = avgvlt_plot
        report_params['batteryStats']['gradVoltPlot'] = vltgrd_plot

        # TODO: Calculate expected hibernate date/time (6500mV)
        hib_thres = 6500
        valid_gradient = power_stats.loc[power_stats['Voltage_gradient'] < 0]
        latest_vlt = valid_gradient['Voltage_Min'].values[-1] * 1000
        latest_win = pd.to_datetime(valid_gradient['End'].values[-1])
        if latest_vlt > hib_thres:
            days_to_hibernate = -(latest_vlt - hib_thres) / valid_gradient['Voltage_gradient'].values[-1]
            const_grad = timedelta(days=days_to_hibernate) + latest_win
            const_acc = pd.NaT
            lookup = pd.NaT
            min_hib = pd.Series([const_grad, const_acc, lookup]).min()
        else:
            # TODO: Return actual hibernation date/time if instrument is already below 6.5V
            min_hib = latest_win
        report_params['batteryStats']['HibernateEstimate'] = min_hib.strftime('%Y-%m-%d')

    # Calculate current draw, if both power and voltage data present
    if ('LE3' in power_data) and ('ME4' in power_data):
        curr_stats = power_data['LE3'].stats.copy()
        curr_stats.channel = 'LZ9'
        curr_stats.description = 'Current Draw (A)'
        if 'response' in curr_stats:
            curr_stats.__delitem__('response')

        # Get equal sample rates for voltage and power
        if power_data['ME4'].stats.sampling_rate != power_data['LE3'].stats.sampling_rate:
            factor = power_data['ME4'].stats.sampling_rate / power_data['LE3'].stats.sampling_rate
            if (factor % 1) > 1e-5:
                g_log.warning('Cannot resample voltage data to match power data, non-integer factor {}.'.format(factor))
            else:
                power_data['ME4'].trim(starttime=power_data['LE3'].stats.starttime,
                                       endtime=power_data['LE3'].stats.endtime, nearest_sample=True)
                factor = int(factor)
                power_data['ME4'].decimate(factor, no_filter=True, strict_length=False)

        curr_data = -power_data['LE3'].data / power_data['ME4'].data
        tr_curr = obspy.Trace(curr_data, curr_stats)
        st_curr = tr_curr.split()
        st_curr.write(os.path.join(output_dir, 'calculated_current.mseed'), format='MSEED')
        if (tr_curr.stats.endtime - tr_curr.stats.starttime) > 5*24*60*60:
            curr_windowed = nf.rolling_window_stats(tr_curr, full=False)
        else:
            stat_window = (tr_curr.stats.endtime - tr_curr.stats.starttime) * 0.6
            if 'battery_stats_window' not in report_params:
                report_params['battery_stats_window'] = stat_window
            curr_windowed = nf.rolling_window_stats(tr_curr, window_length=stat_window, window_offset=stat_window/3,
                                                    full=False)
        crnt = pd.DataFrame(curr_windowed, columns=['Start', 'End', 'Center', 'Min_Amps', 'Max_Amps', 'Avg_Amps'])
        # Time series plot (applies instrument sensitivity in-place if response present in tr.meta)
        current_plot = os.path.join(output_dir, 'current_{0}.png'.format(obs_log['OBS ID'].values[0]))
        if not (use_existing_plots and os.path.isfile(current_plot)):
            fig, ax = plt.subplots(1, 1, figsize=[8, 2.5])
            crnt.plot(x='Center', y='Avg_Amps', kind='line', ax=ax, xlabel='Date/Time', ylabel='Current Draw (A)',
                      legend=False)
            ax.grid(True, ls=':')
            fig.tight_layout()
            fig.savefig(current_plot)
            plt.close(fig)
        report_params['batteryStats']['currentPlot'] = current_plot

    timing_points.append(timeit.default_timer())
    g_log.debug("Time spent checking battery stats: {0} seconds".format((timing_points[-1] - timing_points[-2])))
    debug_info['timing']['battery_summary'] = timing_points[-1] - timing_points[-2]

    # Sort channel information by specified order
    for ch_type in ['seismic', 'ocean', 'power', 'health']:
        sorted_channels = sorted(report_params[ch_type + '_channels'], key=lambda d: d['order'])
        report_params[ch_type + '_channels'] = sorted_channels

    if obs_log['Station'].values[0] != obs_log['OBS ID'].values[0]:
        id_str = '_'.join([obs_log['Station'].values[0], obs_log['OBS ID'].values[0]])
    else:
        id_str = obs_log['OBS ID'].values[0]

    # Save report to *.md and *.pdf formats
    report_md = os.path.join(output_dir, 'QC_report_{0}_auto.md'.format(id_str))
    qcReport = ReportGenerator(type='qc')
    md_out, report_buffer = qcReport.write_report(report_params, report_md)

    pandoc_args = []
    tex_path = os.path.abspath(os.path.expanduser(os.path.expandvars(config.get('common', 'pdftex_path', fallback=None))))
    if tex_path is not None:
        pandoc_args.append('--pdf-engine={0}'.format(tex_path))
    pandoc_args.extend(['--toc'])

    report_pdf = os.path.join(output_dir, 'QC_report_{0}_auto.pdf'.format(id_str))
    report_converted = pypandoc.convert_text(report_buffer, to='pdf', format='md', outputfile=report_pdf, extra_args=pandoc_args)
    g_log.info("Report saved as {0}".format(report_pdf))

    timing_points.append(timeit.default_timer())
    g_log.debug("Time spent creating report: {0} seconds".format((timing_points[-1] - timing_points[-2])))
    debug_info['timing']['create_report'] = timing_points[-1] - timing_points[-2]

    g_log.info("end")
    print(debug_info)
    g_log.debug(str(debug_info))


if __name__ == '__main__':
    num_vcpu = psutil.cpu_count(logical=True)
    num_cores = psutil.cpu_count(logical=False)

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
    parser.add_argument('--logcolnames', dest="obslog_column_names", action="store_true",
                        help="Use column names from OBS deployment log file.")
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
    parser.add_argument('--use_existing_plots', dest='use_existing_plots', action='store_true',
                        help='Do not re-create plots which already exist in output directory. False by default.')
    # TODO: When using ST, project name will come from there instead
    parser.add_argument('--projectname', dest="project_name",
                        help="Project name to be displayed in reports. If not specified, code looks in file extra_meta "
                             "instead.")
    parser.add_argument('--colormap', dest="colormap", default=None,
                        help="Name of matplotlib colormap to use for spectrogram plots.")
    parser.add_argument('--config', dest='config_path', help="Path to config file (if not using default).")
    parser.add_argument('--parallel', dest='parallel', action='store_true',
                        help="Run with multiprocessing parallelization for PSD calculations. Only implemented for "
                             "buffered seismic data.")
    parser.add_argument('--max_processes', dest='max_proc', type=int, default=0,
                        help="Maximum number of processes/threads to be used in parallelized analysis.")
    parser.add_argument('--debug', dest='debug', action='store_true',
                        help="Activate debug mode (more verbose logging). Command-line only.")

    try:
        start_time = datetime.now()
        t0 = timeit.default_timer()
        args = parser.parse_args()

        if args.base_dir:
            base_dir = os.path.abspath(os.path.expanduser(os.path.expandvars(args.base_dir)))
        else:
            base_dir = None

        # Read config file
        if args.config_path:
            config_path = args.config_path
            if args.relative_paths:
                if base_dir is not None:
                    config = config_handler.get_config(os.path.join(base_dir, args.config_path))
                else:
                    raise RuntimeError('Missing command-line argument: Cannot use relative path for config file if no '
                                       'base_dir specified.')
            else:
                config = config_handler.get_config(os.path.abspath(os.path.expanduser(os.path.expandvars(args.config_path))))
        else:
            config = config_handler.get_config()

        # Copy existing config info and add/update from command line arguments
        full_config = config_handler.copy_config(config)
        # Key folders
        if base_dir is None:
            base_dir = os.path.abspath(os.path.expanduser(os.path.expandvars(config.get('dataset', 'base_dir', fallback=os.path.join(resource_dir, 'test_data')))))
        full_config['dataset']['base_dir'] = base_dir

        if args.relative_paths:
            relpath = True
        else:
            relpath = config.getboolean('dataset', 'relative_paths', fallback=False)
        full_config['dataset']['relative_paths'] = str(relpath)

        # Location of OBS data
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
        data_dir = os.path.normpath(data_dir)

        # OBS identifier
        default_obs = False
        if args.obs_id:
            obs_identifier = args.obs_id
        else:
            obs_identifier = config.get('dataset', 'obsid', fallback=None)
        if obs_identifier is None:
            warnings.warn('No valid OBS identifier given, using default AQU-0000.')
            default_obs = True
            obs_identifier = 'AQU-0000'
        full_config['dataset']['obsid'] = obs_identifier

        # Runtime flags
        for flag, key in zip([args.function_check, args.detrend_seis, args.use_existing_plots, args.debug, args.obslog_column_names, args.parallel], ['function_check', 'detrend_seismic', 'use_existing_plots', 'debug', 'logcolnames', 'parallel']):
            config_flag = config.getboolean('dataset', key, fallback=False)
            # only overwrite existing flags if CL arguments are present and different from config
            if flag and not config_flag:
                full_config['dataset'][key] = str(flag)

        # Get log directory if in config
        log_dir = config.get('common', 'log_dir', fallback=None)
        debug_logging = full_config.getboolean('dataset', 'debug', fallback=False)
        if log_dir is not None:
            if log_dir == ':base':
                logs_dir = os.path.join(base_dir, 'logs')
            elif log_dir == ':data':
                logs_dir = os.path.join(data_dir, 'logs')
            else:
                logs_dir = os.path.abspath(os.path.expanduser(os.path.expandvars(log_dir)))
            if not os.path.exists(logs_dir):
                os.makedirs(logs_dir)
            g_log = logger.get_general_logger(start_time, obs_identifier, debug=debug_logging, logs_dir=logs_dir)
        else:
            g_log = logger.get_general_logger(start_time, obs_identifier, debug=debug_logging)

        g_log.info("\n\n=====================================================================")
        g_log.info("Starting job: {0}".format(str(args)))
        g_log.info("{} CPU cores available on this machine ({} logical processors)".format(num_cores, num_vcpu))

        if default_obs:
            g_log.warn('No valid OBS identifier given, using default AQU-0000.')

        id_type = 'unknown'
        if re.match(r'AQU-[0-9a-fA-F]{4}', obs_identifier):
            id_type = 'serial'
        elif re.match(r'D[aA][lL][_\-][0-9]{2,3}', obs_identifier):
            id_type = 'obs_name'

        # Output directory for report files, if specified
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
            output_dir = os.path.normpath(output_dir)

        if output_dir is not None:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

        # Path to deployment summary spreadsheet
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
            data_log_file = os.path.normpath(data_log_file)
        else:
            data_log_file = os.path.join(base_dir, 'log.xlsx')
            if relpath:
                full_config['dataset']['datalog'] = 'log.xlsx'
            else:
                full_config['dataset']['datalog'] = os.path.normpath(data_log_file)

        if args.log_delim:
            log_delim = args.log_delim
        else:
            log_delim = config.get('dataset', 'logdelimiter', fallback=',')
        full_config['dataset']['logdelimiter'] = log_delim

        deploy_start = None
        if args.startdate:
            deploy_start = datetime.strptime(args.startdate, "%Y%m%d")
        elif config.get('dataset', 'start', fallback=False):
            deploy_start = datetime.strptime(config.get('dataset', 'start'), "%Y%m%d")

        g_log.info('Reading project metadata from {0}...'.format(data_log_file))
        log_column_names = full_config.getboolean('dataset', 'logcolnames', fallback=False)
        obs_log_info = nf.io.parse_obs_log(data_log_file, log_delim, names_in_file=log_column_names)
        # Find this OBS in the metadata tables
        base_meta, rec_meta = None, None
        if id_type == 'serial':
            base_meta = obs_log_info['basic'].loc[obs_log_info['basic']['OBS ID'] == obs_identifier]
            rec_meta = obs_log_info['recovery'].loc[obs_log_info['recovery']['OBS ID'] == obs_identifier]
        elif id_type == 'obs_name':
            base_meta = obs_log_info['basic'].loc[obs_log_info['basic']['OBS Name'] == obs_identifier]
            rec_meta = obs_log_info['recovery'].loc[obs_log_info['recovery']['OBS Name'] == obs_identifier]
        else:
            id_columns = ['Station', 'OBS Name', 'OBS ID']
            for col in id_columns:
                if obs_identifier in obs_log_info['basic'][col].values:
                    base_meta = obs_log_info['basic'].loc[obs_log_info['basic'][col] == obs_identifier]
                    rec_meta = obs_log_info['recovery'].loc[obs_log_info['recovery'][col] == obs_identifier]
                    break

        if base_meta is None or base_meta.empty:
            raise IndexError('OBS {0} not found in provided metadata.'.format(obs_identifier))
        if deploy_start is not None:
            base_meta = base_meta.loc[(base_meta['Launch Date/Time (UTC)'] >= deploy_start) &
                                      (base_meta['Launch Date/Time (UTC)'] < deploy_start + timedelta(days=1))]
            rec_meta = rec_meta.loc[(rec_meta['On-Deck Date/Time (UTC)'] == base_meta['Recovery Date/Time (UTC)'].values[0])]
        if base_meta.shape[0] > 1:
            raise IndexError('Multiple entries found for OBS {0} in provided metadata. Please use a unique identifier '
                             'or provide start date.'.format(obs_identifier))

        # Check that deployment start date matches between metadata and CLI/config argument
        if deploy_start is not None:
            meta_start = min(base_meta['Launch Date/Time (UTC)'].values[0],
                             base_meta['Date/Time on Seafloor (UTC)'].values[0])
            if pd.Timestamp(meta_start).to_pydatetime().date() != deploy_start.date():
                g_log.warning(
                    "Start time in metadata file ({0}) is different from runtime/config argument ({1}).".format(
                        meta_start.strftime('%Y-%m-%d'), deploy_start.strftime('%Y-%m-%d')))

        # Path to channel map file, if specified
        channel_map = None
        if args.channel_map:
            ch_map = args.channel_map
        else:
            ch_map = config.get('dataset', 'channelmap', fallback=None)
        if ch_map is not None:
            full_config['dataset']['channelmap'] = os.path.normpath(ch_map)
            if relpath:
                channel_map = nf.io.read_channel_map(os.path.join(base_dir, ch_map))
            else:
                channel_map = nf.io.read_channel_map(ch_map)
        else:
            g_log.info("No channel map provided. Channel IDs will be processed as they appear in the raw data files.")

        # Path to metadata file (dataless SEED or StationXML), if specified
        metadata_file = None
        if args.metadata_file:
            meta_file = args.metadata_file
        else:
            meta_file = config.get('dataset', 'metadata', fallback=None)
        if meta_file is not None:
            full_config['dataset']['metadata'] = os.path.normpath(meta_file)
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
            full_config['dataset']['extra_meta'] = os.path.normpath(json_file)
            if relpath:
                project_json = os.path.join(base_dir, json_file)
            else:
                project_json = os.path.abspath(os.path.expanduser(os.path.expandvars(json_file)))
        else:
            project_json = os.path.join(base_dir, 'project_info.json')
            if relpath:
                full_config['dataset']['extra_meta'] = 'project_info.json'
            else:
                full_config['dataset']['extra_meta'] = os.path.normpath(project_json)
        if os.path.isfile(project_json):
            g_log.info("Reading project metadata from {0}...".format(os.path.normpath(project_json)))
            pj = open(project_json)
            project_meta = json.load(pj)
        else:
            g_log.info("No project metadata JSON found at {0}".format(os.path.normpath(project_json)))
            full_config.remove_option('dataset', 'extra_meta')

        # Pull extra metadata for this station/deployment only (from JSON)
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

        if args.network_id:
            network = args.network_id
        else:
            network = config.get('dataset', 'network', fallback='XX')
        full_config['dataset']['network'] = network

        if args.colormap is not None:
            colormap = args.colormap
        else:
            colormap = config.get('dataset', 'colormap', fallback='viridis')
        full_config['dataset']['colormap'] = colormap

        max_processes = None
        if args.max_proc > 0:
            max_processes = args.max_proc

        # Gather some basic information for report
        report_kwargs = {
            'today': datetime.now().strftime('%Y-%m-%d'),
        }
        if args.project_name:
            report_kwargs['projectName'] = args.project_name
        elif config.get('dataset', 'projectname', fallback=False):
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
        report_kwargs['deployComments'] = base_meta['Deployment Comments'].values[0]
        # TODO: Handle case of intermediate download (no "recovery" time yet)
        report_kwargs['recovered'] = pd.to_datetime(base_meta['Recovery Date/Time (UTC)'].values[0])
        report_kwargs['recoverComments'] = base_meta['Recovery Comments'].values[0]
        deployed_days = (report_kwargs['recovered'] - report_kwargs['deployed']) / timedelta(days=1)
        report_kwargs['deploymentDays'] = '{:.3f}'.format(deployed_days)
        report_kwargs['clockDrift'] = '{:.0f}'.format(base_meta['Clock Offset on Deck (ms)'].values[0])
        report_kwargs['clockDriftPerDay'] = '{:.3f}'.format(base_meta['Clock Offset on Deck (ms)'].values[0] / deployed_days)
        report_kwargs['batteryLevel'] = {'start': '{:.0f}'.format(base_meta['Battery SOC at Deployment (%)'].values[0]),
                                         'end': '{:.0f}'.format(base_meta['Battery SOC at Recovery (%)'].values[0])}
        if 'qc_intro' in project_meta['this_deployment']:
            report_kwargs['introText'] = project_meta['this_deployment']['qc_intro']
        report_kwargs['psdWindowSecs'] = config.getint('seismic', 'window_length')
        report_kwargs['psdOverlapPercent'] = config.getint('seismic', 'overlap_percent')

        # Check for tilt info
        if rec_meta is not None:
            try:
                if {'AccZ', 'AccN', 'AccE'}.issubset(rec_meta.columns):
                    mems_acc = [rec_meta[c].values[0] for c in ['AccZ', 'AccN', 'AccE']]
                    if abs(mems_acc[0]) > 0:
                        tilt_deg = np.degrees(np.arctan(np.sqrt(mems_acc[1]**2 + mems_acc[2]**2) / mems_acc[0]))
                        report_kwargs['tiltAtRecovery'] = '{:.3f}'.format(tilt_deg)
            except TypeError as e:
                # TypeError if AccZ is None (from abs(None))
                pass

        setup_time = timeit.default_timer()
        g_log.info("Time spent parsing arguments and preparing to process data: {0} seconds".format(setup_time - t0))
        g_log.debug('Logging level: {}'.format(g_log.getEffectiveLevel()))

        # Process data files to apply clock drift correction and update metadata
        process(data_dir, base_meta, network, full_config, output_dir=output_dir, metadata=metadata_file,
                channel_map=channel_map, project_meta=project_meta, cmap=colormap, max_proc=max_processes,
                flags_from_config=True, **report_kwargs)

        # Final post-process logging
        proc_time = timeit.default_timer()
        g_log.info("Time spent processing data: {0} seconds".format(proc_time - setup_time))
        g_log.debug('Logging level: {}'.format(g_log.getEffectiveLevel()))

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
