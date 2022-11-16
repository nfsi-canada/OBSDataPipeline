import argparse
from glob import glob
from ioos_qc import utils as iq_utils
from ioos_qc import qartod
import json
import matplotlib.pyplot as plt
import numpy as np
import obspy
import os
import pandas as pd
import pypandoc
import re
from scipy import signal
import shutil
import traceback
import warnings
from datetime import datetime, timedelta
from obspy.io.stationxml.core import validate_stationxml
from obspy.signal import PPSD
from sklearn.linear_model import LinearRegression
import timeit
from obspy.io.mseed.util import get_start_and_end_time

import nfsi_obs as nf
from utilities import config_handler, logger, check_nan, ReportGenerator

current_dir = os.path.dirname(os.path.abspath(__file__))
# Ensure resource directory exists
resource_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'resource/OBSDataPipeline')
if not os.path.isdir(resource_dir):
    os.makedirs(resource_dir)


def process(data_dir, obs_log, network_id, config, output_dir=None, metadata=None, channel_map=None, project_meta=None, full=True, detrend=False, backup=True, **kwargs):
    """
    Extra keyword arguments are included as report parameters (must match variables in template file).
    """
    proc_start = timeit.default_timer()
    g_log.info("start")

    # Initialize report parameters dictionary with input keywords
    report_params = {}
    report_params.update(kwargs)
    # Add empty lists for channel-specific information
    for key in ['seismic_channels', 'ocean_channels', 'power_channels', 'health_channels']:
        report_params[key] = []

    # Windowing parameters for seismic data
    win_len = 3600
    overlap = 0.75
    if 'psdWindowsSecs' in report_params:
        win_len = report_params['psdWindowSecs']
    if 'psdOverlapPercent' in report_params:
        overlap = report_params['psdOverlapPercent'] / 100
    spec_win = int(config.get('seismic', 'spectrogram_window', fallback=60))

    base_time = timeit.default_timer()
    g_log.debug("Basic processing setup time: {0} seconds".format((base_time - proc_start)))

    # Start/end of time on seafloor (if provided)
    sf_start, sf_end = None, None
    if not pd.isnull(obs_log['Date/Time on Seafloor (UTC)'].values[0]):
        sf_start = obspy.UTCDateTime(pd.to_datetime(obs_log['Date/Time on Seafloor (UTC)'].values[0]))
    if not pd.isnull(obs_log['Date/Time Released (UTC)'].values[0]):
        sf_end = obspy.UTCDateTime(pd.to_datetime(obs_log['Date/Time Released (UTC)'].values[0]))

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
        elif len(xml_files) > 0:
            for xf in xml_files:
                is_sxml = validate_stationxml(xf)[0]
                if is_sxml and (station_info is None):
                    g_log.info("Found StationXML file in data directory: {0}".format(xf))
                    station_info = obspy.read_inventory(xf)
        else:
            g_log.warning("No metadata file provided, and none found in data directory.")

    meta_time = timeit.default_timer()
    g_log.debug("Time spent reading station metadata file: {0} seconds".format((meta_time - base_time)))

    # Find data files and backup if necessary
    # TODO: Remove file backup here once it has been copied to pre-processing script (QC doesn't change miniSEED files)
    raw_files = glob(os.path.join(data_dir, '**/*.mseed'), recursive=True)
    g_log.info("Found {0} miniSEED file(s) in data directory and sub-folders".format(len(raw_files)))
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

    search_time = timeit.default_timer()
    g_log.debug("Time spent searching for data files and backing up raw data: {0} seconds".format((search_time - meta_time)))

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

    sort_time = timeit.default_timer()
    g_log.debug("Time spent sorting and labeling data files: {0} seconds".format((sort_time - search_time)))

    # Initialize arrays for saving stats
    all_gaps = []
    centring = pd.DataFrame()
    power_stats = pd.DataFrame()
    avg_power = []
    voltage_stats = []

    arr_time = timeit.default_timer()
    g_log.debug("Time spent setting up arrays for stats: {0} seconds".format((arr_time - sort_time)))

    # Loop through data files (grouped by channel set name)
    for label, files in labeled_files.groupby('channel'):
        ch_start = timeit.default_timer()
        g_log.info("Begin processing channel set {0}".format(label))
        g_log.info("{0} data file(s) in list".format(len(files.index)))

        # Ignore channels with lots of data files (long time periods of seismic data) for now
        # TODO: Implement data file buffering for long time periods
        if len(files.index) > 3:
            filetimes = []
            for rf in files['path'].values:
                times = get_start_and_end_time(rf)
                filetimes.append(times)
            filetimes = np.array(filetimes)
            data_start = min(filetimes[:, 0])
            data_end = max(filetimes[:, 1])

            trace_info = nf.plotting.buffer_seismic_data(files['path'].values, output_dir, g_log, network_id,
                                                         station_info, channel_map, project_meta, win_len, spec_win,
                                                         overlap, plot_length=7)
            channel_type = trace_info['channelType']

            report_params[channel_type + '_channels'].append(trace_info)

            g_log.info('{0} | {1} - {2} | {3}'.format(trace_info['seedID'],
                                                      data_start.datetime.strftime('%Y-%m-%d %H:%M:%S.%f'),
                                                      data_end.datetime.strftime('%Y-%m-%d %H:%M:%S.%f'),
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

            # Cut data to time on seafloor (if start/end times provided)
            if (sf_start is not None) or (sf_end is not None):
                # This shouldn't change `data` if there is no data to cut out
                data.trim(sf_start, sf_end, nearest_sample=False)

            sf_time = timeit.default_timer()
            g_log.debug("Time spent cutting to on-seafloor: {0} seconds".format((sf_time - read_time)))

            # Populate metadata from other files as necessary
            data = nf.metadata.update_metadata(data, network_id, g_log, station_info, channel_map, project_meta)
            data.merge()
            print(data)

            metadata_time = timeit.default_timer()
            g_log.debug("Time spent applying metadata: {0} seconds".format((metadata_time - sf_time)))

            # Perform QC
            # TODO: Combine single and multi-channel cases to simplify code (no real reason to separate) -> TEST multi-channel
            seismic = obspy.Stream()
            ocean = obspy.Stream()
            power = obspy.Stream()
            health = obspy.Stream()

            str_time = timeit.default_timer()
            g_log.debug("Time spent creating empty streams: {0} seconds".format((str_time - sf_time)))

            # Gap test
            gaps = data.get_gaps()
            all_gaps.extend(gaps)
            if len(gaps) > 0:
                g_log.info('Found {0} gap(s) or overlap(s) in recorded data'.format(len(gaps)))
                data.print_gaps()

            gt_time = timeit.default_timer()
            g_log.debug("Time spent for gap test: {0} seconds".format((gt_time - str_time)))

            for tr in data:
                tr_starttime = timeit.default_timer()

                # Assign to relevant group of channels
                channel_type = nf.metadata.get_channel_type(tr.meta.channel)
                if channel_type == 'seismic':
                    # seismic data and hydrophone
                    seismic.append(tr)
                elif channel_type == 'ocean':
                    # oceanographic data (external P/T, include APG if present)
                    ocean.append(tr)
                elif channel_type == 'power':
                    # battery voltage and power consumption
                    power.append(tr)
                else:
                    health.append(tr)

                cg_time = timeit.default_timer()
                g_log.debug("Time spent assigning to channel group: {0} seconds".format((cg_time - tr_starttime)))

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

                # Time series plot
                trace_info['traceLoc'] = nf.plotting.trace_plot(tr, output_dir, dmin, dmax, qc_config)

                plt_time = timeit.default_timer()
                g_log.debug("Time spent plotting trace: {0} seconds".format((plt_time - more_meta_time)))

                # Noise level QC steps (seismic channels and hydrophone) -> if channel code == "CHx" or "HDF"
                if channel_type == 'seismic':
                    for metaKey, reportKey in zip(['azimuth', 'dip'], ['azimuth', 'dip']):
                        if hasattr(tr.meta, metaKey):
                            trace_info[reportKey] = tr.meta[metaKey]

                    # Detrend seismic data (RMS linear fit)
                    if detrend:
                        tr.detrend('linear')
                        demean_data_plot = os.path.join(output_dir, 'demean_{0}.png'.format(tr.id))
                        data.plot(outfile=demean_data_plot)

                    # Spectrogram
                    trace_info['specLoc'] = nf.plotting.spectrogram(tr, output_dir, spec_win, overlap)

                    # Plot PSDs of data
                    trace_info['psdLoc'] = nf.plotting.psd_plot(tr, output_dir, win_len, overlap)

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
                    if qc_config is not None:
                        if 'qartod' in qc_config:
                            if 'gross_range_test' in qc_config['qartod']:
                                range_check = qartod.gross_range_test(tr.data, **qc_config['qartod']['gross_range_test'])
                                if np.any(range_check > 1):
                                    g_log.info('Channel {0} has suspect values at {1} sample(s) and failing values at {2} sample(s)'.format(tr.id, np.sum(range_check==3), np.sum(range_check==4)))
                                check_trace = obspy.Trace(range_check, header=tr.stats)
                                trace_info['qcPlotLoc'] = nf.plotting.qartod_plot(check_trace, output_dir, 'gross_range_check')

                            if re.match(r'[A-Z]M[1-3ENZ]', tr.meta.channel) and ('flat_line_test' in qc_config['qartod']):
                                # centring channels only, must have flat-line test criteria specified
                                flt_params = qc_config['qartod']['flat_line_test'].copy()
                                flatline = qartod.flat_line_test(tr.data, timestamps,
                                                                 int(flt_params.pop('suspect_threshold')),
                                                                 int(flt_params.pop('fail_threshold')))
                                centring[tr.id] = pd.Series(flatline, index=timestamps)

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
                            # Power consumption
                            report_params['meanPower'] = '{:.3f}'.format(np.mean(tr.data))

                            # 3-day rolling window of average power consumption
                            window_start = first_window
                            while window_start < last_window:
                                window = tr.slice(window_start, window_start + window_length)
                                if window.data.count() > 0:
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
                                if window.data.count() > 0:
                                    secs = np.array(window.times(type='relative'))
                                    secs_valid = secs[window.data.mask == False].reshape(-1, 1)
                                    valid_data = window.data[window.data.mask == False]
                                    reg = LinearRegression().fit(secs_valid, valid_data)
                                    gradient = reg.coef_[0] * 1000 * 60 * 60 * 24   # convert V/s to mV/day for voltage gradient
                                    r2 = reg.score(secs_valid, valid_data)   # R^2 coefficient of linear fit (should be very close to 1)
                                    voltage_stats.append([window_start.datetime, window.data.min(), window.max(), window.data.mean(), gradient, r2])
                                window_start += window_offset
                    # TODO: Analysis of state-of-health variables?
                    # TODO: Down-sample external pressure and temperature data (plot and save as netCDF)

                tran_time = timeit.default_timer()
                g_log.debug("Time spent performing trace-specific analysis: {0} seconds".format((tran_time - plt_time)))

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

    loop_time = timeit.default_timer()
    g_log.debug("Time spent processing data files: {0} seconds".format((loop_time - arr_time)))

    # Check centring behaviour
    if len(centring.columns) > 0:
        is_centred = centring.eq(4).all(axis='columns')
        # List of time periods where is_centred is True -> [start, end, npts]
        centred = nf.get_true_periods(is_centred)
        
        # TODO: Compile text to summarize centring behaviour
        ctx = ''

        centring_plot = os.path.join(output_dir, 'centring_{0}.png'.format(obs_log['OBS ID'].values[0]))
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

    # Parse gap information for report
    if len(all_gaps) > 0:
        report_params['gapList'] = []
        for gap in all_gaps:
            report_params['gapList'].append({
                'id': '.'.join(gap[0:4]),
                'start': gap[4].strftime('%Y-%m-%d %H:%M:%S.%f'),
                'end': gap[5].strftime('%Y-%m-%d %H:%M:%S.%f'),
                'sec': gap[6],
                'samp': gap[7]
            })

    gap_time = timeit.default_timer()
    g_log.debug("Time spent formatting gap information: {0} seconds".format((gap_time - centre_time)))

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
        fig, ax = plt.subplots(1, 1, figsize=[8, 2.5])
        power_stats.plot(x='Plot_Time', y='Power_Mean', kind='line', ax=ax, xlabel='Date/Time', ylabel='Average Power Consumption (W)')
        fig.tight_layout()
        fig.savefig(avgpow_plot)
        plt.close(fig)

        # Average voltage vs time
        avgvlt_plot = os.path.join(output_dir, 'voltage_mean_{0}.png'.format(obs_log['OBS ID'].values[0]))
        fig, ax = plt.subplots(1, 1, figsize=[8, 2.5])
        power_stats.plot(x='Plot_Time', y='Voltage_Mean', kind='line', ax=ax, xlabel='Date/Time', ylabel='Average Voltage (V)')
        fig.tight_layout()
        fig.savefig(avgvlt_plot)
        plt.close(fig)

        # Voltage gradient vs time
        vltgrd_plot = os.path.join(output_dir, 'voltage_gradient_{0}.png'.format(obs_log['OBS ID'].values[0]))
        fig, ax = plt.subplots(1, 1, figsize=[8, 2.5])
        power_stats.plot(x='Plot_Time', y='Voltage_gradient', kind='line', ax=ax, xlabel='Date/Time', ylabel='Voltage Gradient (mV/day)')
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

    g_log.info("end")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform basic QC for OBS data. Will correct channel identifiers if '
                                                 'optional --channelmap argument is provided. Does not require clock '
                                                 'drift correction to have been applied.')
    parser.add_argument('--data_dir', dest="data_dir", help="Directory where OBS data is stored.")
    parser.add_argument('--relative_paths', dest="relative_paths", action="store_true",
                        help="If true, all other path arguments are specified relative to the data directory.")
    parser.add_argument('--datalog', dest="datalog",
                        help="Log file from deployment/recovery. Must include station identifiers and clock drift "
                             "measurements. If not specified, assumed to be a file called 'log.xlsx' in the data "
                             "directory. Preferred format is XLSX (or similar spreadsheet) following NFSI template.")
    parser.add_argument('--logdelimiter', dest="log_delim", default=",",
                        help="If the OBS log file is delimited text (other than comma-delimited), use this to specify "
                             "the column delimiter.")
    parser.add_argument('--obsid', dest="obs_id", default="AQU-0000",
                        help="OBS identifier: station name or serial number")
    parser.add_argument('--network', dest="network_id", default='XX',
                        help="Network identifier assigned by FDSN for this project. Default 'XX' for test data.")
    parser.add_argument('--outdir', dest="outdir", default=None,
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
                        help="If true, will skip creating a backup copy of the raw data files.")
    # TODO: When using ST, project name will come from there instead
    parser.add_argument('--projectname', dest="project_name", help="Project name to be displayed in reports")
    parser.add_argument('--config', dest='config_path', help="Path to config file (if not using default).")

    try:
        start_time = datetime.now()
        t0 = timeit.default_timer()
        args = parser.parse_args()
        obs_id = args.obs_id

        g_log = logger.get_general_logger(start_time, obs_id)
        g_log.info("\n\n=====================================================================")
        g_log.info("Starting job: {0}".format(str(args)))

        obs_identifier = args.obs_id
        id_type = 'unknown'
        if re.match(r'AQU-[0-9a-fA-F]{4}', obs_identifier):
            id_type = 'serial'
        elif re.match(r'D[aA][lL][_\-][0-9]{2,3}', obs_identifier):
            id_type = 'obs_name'

        if args.data_dir:
            data_dir = os.path.abspath(os.path.expanduser(os.path.expandvars(args.data_dir)))
        else:
            data_dir = os.path.join(resource_dir, 'test_data')

        output_dir = None
        if args.outdir is not None:
            if args.relative_paths:
                output_dir = os.path.join(data_dir, args.outdir)
            else:
                output_dir = os.path.abspath(os.path.expanduser(os.path.expandvars(args.outdir)))

        if args.datalog:
            if args.relative_paths:
                data_log_file = os.path.join(data_dir, args.datalog)
            else:
                data_log_file = os.path.abspath(os.path.expanduser(os.path.expandvars(args.datalog)))
        else:
            data_log_file = os.path.join(data_dir, 'log.xlsx')

        g_log.info('Reading project metadata from {0}...'.format(data_log_file))
        obs_log_info = nf.io.parse_obs_log(data_log_file, args.log_delim)
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
        if base_meta.shape[0] > 1:
            raise IndexError('Multiple entries found for OBS {0} in provided metadata. Please use a unique identifier.'.format(obs_identifier))

        channel_map = None
        if args.channel_map:
            if args.relative_paths:
                channel_map = nf.io.read_channel_map(os.path.join(data_dir, args.channel_map))
            else:
                channel_map = nf.io.read_channel_map(args.channel_map)

        metadata_file = None
        if args.metadata_file:
            if args.relative_paths:
                metadata_file = os.path.join(data_dir, args.metadata_file)
            else:
                metadata_file = os.path.abspath(os.path.expanduser(os.path.expandvars(args.metadata_file)))

        if args.config_path:
            if args.relative_paths:
                config = config_handler.get_config(os.path.join(data_dir, args.config_path))
            else:
                config = config_handler.get_config(os.path.abspath(os.path.expanduser(os.path.expandvars(args.config_path))))
        else:
            config = config_handler.get_config()

        # Read project metadata JSON file
        project_meta = None
        # TODO: Replace with ST integration once we have an instance running
        if channel_map is None:
            g_log.info("No channel map provided. Checking data directory for project_info.json...")
        else:
            # TODO: This seems a bit weird... make these log statements more sensical
            g_log.info("Reading project metadata from [data_dir]/project_info.json...")
        # Search data_dir for project JSON (should have channel descriptions)
        if args.extra_meta:
            if args.relative_paths:
                project_json = os.path.join(data_dir, args.extra_meta)
            else:
                project_json = os.path.abspath(os.path.expanduser(os.path.expandvars(args.extra_meta)))
        else:
            project_json = os.path.join(data_dir, 'project_info.json')
        if os.path.isfile(project_json):
            pj = open(project_json)
            project_meta = json.load(pj)
        else:
            g_log.info("No project metadata JSON found at {0}".format(project_json))

        station_meta = None
        if project_meta is not None:
            try:
                station_meta = list(filter(lambda x: x['name'] == base_meta['Station'].values[0], project_meta['stations']))[0]
            except (KeyError, IndexError):
                g_log.info("No matching station information found in project metadata JSON.")

        # Gather some basic information for report
        report_kwargs = {
            'today': datetime.now().strftime('%Y-%m-%d'),
        }
        if args.project_name:
            report_kwargs['projectName'] = args.project_name
        elif project_meta is not None:
            report_kwargs['projectName'] = project_meta['project']
        else:
            report_kwargs['projectName'] = 'Test Recording'
        if project_meta['common_intro']:
            report_kwargs['intro_pt1'] = project_meta['common_intro']
        report_kwargs['stationName'] = base_meta['Station'].values[0]
        report_kwargs['obsName'] = base_meta['OBS Name'].values[0]
        report_kwargs['obsId'] = base_meta['OBS ID'].values[0]
        report_kwargs['latitude'] = base_meta['Surveyed Latitude'].values[0]
        report_kwargs['longitude'] = base_meta['Surveyed Longitude'].values[0]
        report_kwargs['waterDepth'] = base_meta['Water Depth (m)'].values[0]
        report_kwargs['deployed'] = pd.to_datetime(base_meta['Launch Date/Time (UTC)'].values[0])
        report_kwargs['deployComments'] = dep['Comments'].values[0]
        report_kwargs['recovered'] = pd.to_datetime(base_meta['Recovery Date/Time (UTC)'].values[0])
        report_kwargs['recoverComments'] = rec['Comments'].values[0]
        report_kwargs['deploymentDays'] = (report_kwargs['recovered'] - report_kwargs['deployed']) / timedelta(days=1)
        report_kwargs['clockDrift'] = base_meta['Clock Offset on Deck (ms)'].values[0]
        report_kwargs['batteryLevel'] = rec['Battery SOC (%)'].values[0]
        if station_meta is not None:
            if 'qc_intro' in station_meta:
                report_kwargs['introText'] = station_meta['qc_intro']
        report_kwargs['psdWindowSecs'] = int(config.get('seismic', 'window_length'))
        report_kwargs['psdOverlapPercent'] = int(config.get('seismic', 'overlap_percent'))

        setup_time = timeit.default_timer()
        g_log.info("Time spent parsing arguments and preparing to process data: {0} seconds".format(setup_time - t0))

        # Process data files to apply clock drift correction and update metadata
        process(data_dir, base_meta, args.network_id, config, output_dir, metadata_file, channel_map, project_meta, not args.function_check, args.detrend_seis, not args.skip_backup, **report_kwargs)

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
    except Exception as e:
        print(traceback.print_exc())
        parser.print_help()
        exit(1)
