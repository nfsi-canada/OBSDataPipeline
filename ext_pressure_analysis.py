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
import re
import seaborn as sns
import timeit
import traceback
import warnings

from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

from obspy.io.mseed.util import get_record_information
from obspy.io.stationxml.core import validate_stationxml

import nfsi_obs as nf
from utilities import config_handler, logger

gc.set_debug(gc.DEBUG_UNCOLLECTABLE)
feature_test = False  # set to True to test new features

current_dir = os.path.dirname(os.path.abspath(__file__))
# Ensure resource directory exists
resource_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                            'resource/OBSDataPipeline')
if not os.path.isdir(resource_dir):
    os.makedirs(resource_dir)


def exp_curve(x, m, t, b):
    return m * np.exp(t * x) + b


def process(data_dir, obs_log, network_id, config, output_dir=None, metadata=None, channel_map=None, project_meta=None,
            **kwargs):
    """
    Extra keyword arguments are included as report parameters (must match variables in template file).
    """
    timing_points = []
    error_count = 0
    timing_points.append(timeit.default_timer())
    g_log.info("start")

    debug_info = {'timing': {}}

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
                g_log.warning(
                    "Multiple dataless SEED volumes found in data directory, using {0}.".format(seed_files[0]))
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
    backup_exists = False
    if output_dir is None:
        output_dir = data_dir

    # Write QC processing configuration to file
    with open(os.path.join(output_dir, 'QC_config.ini'), 'w') as configfile:
        config.write(configfile)

    timing_points.append(timeit.default_timer())
    g_log.debug("Time spent searching for data files: {0} seconds".format(
        (timing_points[-1] - timing_points[-2])))
    debug_info['timing']['file_search'] = timing_points[-1] - timing_points[-2]

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
    g_log.info(
        "Files contain data for {0} unique set(s) of channels".format(len(np.unique(labeled_files['channel'].values))))
    debug_info['num_channels'] = len(np.unique(labeled_files['channel'].values))

    timing_points.append(timeit.default_timer())
    g_log.debug(
        "Time spent sorting and labeling data files: {0} seconds".format((timing_points[-1] - timing_points[-2])))
    debug_info['timing']['file_sort'] = timing_points[-1] - timing_points[-2]

    # Initialize arrays for saving stats
    mean_sea_level = []
    sea_temp = []

    timing_points.append(timeit.default_timer())
    g_log.debug("Time spent setting up arrays for stats: {0} seconds".format((timing_points[-1] - timing_points[-2])))
    debug_info['timing']['array_setup'] = timing_points[-1] - timing_points[-2]

    # Add keys for running totals
    debug_info['timing'].update({
        'file_read': 0.,
        'time_cut': 0.,
        'apply_meta': 0.,
        'gap_test': 0.,
        'meta_admin': 0.,
        'trace_plot': 0.,
        'trend_analysis': 0.,
    })
    # Loop through data files (grouped by channel set name)
    for label, files in labeled_files.groupby('channel'):
        proc_timing = []
        proc_timing.append(timeit.default_timer())
        g_log.info("Begin processing channel set {0}".format(label))
        g_log.info("{0} data file(s) in list".format(len(files.index)))

        try:
            # Data file buffering for long time periods (should only be needed for seismic data)
            if len(files.index) > 3:
                g_log.error('This processing is not implemented for channels spread across more than 3 miniSEED files. '
                            'Ensure you are using the raw data as recorded by the Aquarius.')
                raise ValueError('Too many data files to process.')
            else:
                # Get record information from files, exclude channels we aren't interested in
                # Read all files of interest in list
                data = obspy.Stream()
                for rf in files['path'].values:
                    ri = get_record_information(rf)
                    channel_type = nf.metadata.get_channel_type(ri['channel'])
                    # Only considering "oceanographic" data (i.e. external pressure and temperature)
                    if channel_type == 'ocean':
                        temp = obspy.read(rf)
                        for tr in temp:
                            data.append(tr)
                    else:
                        g_log.info('Processing only implemented for oceanographic data. Skipping channel {}.'.format(ri['channel']))
                data.merge()

                proc_timing.append(timeit.default_timer())
                g_log.debug("Time spent reading data file(s): {0} seconds".format((proc_timing[-1] - proc_timing[-2])))
                debug_info['timing']['file_read'] += proc_timing[-1] - proc_timing[-2]

                # Check for empty data stream
                if len(data) < 1:
                    g_log.info('No data read for channel set {}.'.format(label))
                    continue

                # Cut data to time period of interest (if start/end times provided)
                if (data_start is not None) or (data_end is not None):
                    # This shouldn't change `data` if there is no data to cut out
                    data.trim(data_start, data_end, nearest_sample=False)

                proc_timing.append(timeit.default_timer())
                g_log.debug(
                    "Time spent cutting to period of interest: {0} seconds".format((proc_timing[-1] - proc_timing[-2])))
                debug_info['timing']['time_cut'] += proc_timing[-1] - proc_timing[-2]

                # Populate metadata from other files as necessary
                data = nf.metadata.update_metadata(data, network_id, g_log, station_info, channel_map, project_meta)
                data.merge()
                print(data)

                proc_timing.append(timeit.default_timer())
                g_log.debug("Time spent applying metadata: {0} seconds".format((proc_timing[-1] - proc_timing[-2])))
                debug_info['timing']['apply_meta'] += proc_timing[-1] - proc_timing[-2]

                # Perform QC

                # Gap test
                gaps = data.get_gaps()
                if len(gaps) > 0:
                    g_log.info('Found {0} gap(s) or overlap(s) in recorded data'.format(len(gaps)))
                    data.print_gaps()

                proc_timing.append(timeit.default_timer())
                g_log.debug("Time spent for gap test: {0} seconds".format((proc_timing[-1] - proc_timing[-2])))
                debug_info['timing']['gap_test'] += proc_timing[-1] - proc_timing[-2]

                for tr in data:
                    tr_timing = []
                    tr_timing.append(timeit.default_timer())

                    # Get channel name
                    channel_name = tr.id
                    if hasattr(tr.meta, 'description'):
                        channel_name = tr.meta.description

                    # Get channel info from project metadata JSON
                    channel_info = None
                    if project_meta is not None:
                        try:
                            channel_info = \
                            list(filter(lambda ch: ch['channel_id'] == tr.meta.channel, project_meta['channels']))[0]
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
                        if 'qc_config' in channel_info:
                            qc_config = channel_info['qc_config']

                    tr_timing.append(timeit.default_timer())
                    g_log.debug(
                        "Time spent with other metadata admin: {0} seconds".format((tr_timing[-1] - tr_timing[-2])))
                    debug_info['timing']['meta_admin'] += tr_timing[-1] - tr_timing[-2]

                    # Remove data write spikes from raw data
                    # Thresholds determined based on raw counts, so needs to happen before sensitivity is removed by plotting function
                    # TODO: Check if this will work with APG data if we ever collect any
                    if re.match(r'[A-Z]DO', tr.meta.channel):
                        averaging_window = 15 * 60 * tr.stats.sampling_rate
                        outlier_cutoff = 1
                    elif re.match(r'[A-Z]KO', tr.meta.channel):
                        averaging_window = [7 * 60 * tr.stats.sampling_rate, 30 * 60 * tr.stats.sampling_rate]
                        outlier_cutoff = 2.05
                    else:
                        g_log.info('Unrecognized channel type {}. Using default despiking thresholds.'.format(tr.meta.channel))
                        averaging_window = 10 * 60 * tr.stats.sampling_rate
                        outlier_cutoff = 1

                    despiked = nf.remove_write_spikes(tr, delta=outlier_cutoff, span=averaging_window,
                                                      savedf=False, dfpath=os.path.join(output_dir,
                                                                                        '{}_despiking_info.csv'.format(
                                                                                            tr.id)))
                    g_log.debug('Despiked data type: {}'.format(despiked.data.dtype.type))
                    try:
                        despiked.write(os.path.join(output_dir, '{}_despiked.mseed'.format(tr.id)),
                                       format='MSEED', encoding='STEIM2')
                    except Exception as e:
                        g_log.error('Error writing despiked data to file!')
                        g_log.error(traceback.format_exc())

                    despiked_plot = nf.plotting.trace_plot(despiked, output_dir, dmin, dmax, qc_config)

                    tr_timing.append(timeit.default_timer())
                    g_log.debug("Time spent despiking trace: {} seconds".format((tr_timing[-1] - tr_timing[-2])))

                    # Time series plot (applies instrument sensitivity in-place if response present in tr.meta)
                    full_trace_plot = nf.plotting.trace_plot(tr, output_dir, dmin, dmax, qc_config)

                    tr_timing.append(timeit.default_timer())
                    g_log.debug("Time spent plotting trace: {0} seconds".format((tr_timing[-1] - tr_timing[-2])))
                    debug_info['timing']['trace_plot'] += tr_timing[-1] - tr_timing[-2]

                    # Calculate rolling average of oceanographic data (despiked)
                    trace_length = tr.meta.endtime - tr.meta.starttime
                    if hasattr(tr.meta, 'response'):
                        units = tr.meta.response.instrument_sensitivity.input_units
                    else:
                        units = ''

                    if tr.meta.channel == 'MDO':
                        # External pressure
                        g_log.info('Seafloor pressure ({}): mean {:.3f}, min {:.3f}, max {:.3f}, stdev {:.3f}'.format(units, np.mean(despiked.data), np.min(despiked.data), np.max(despiked.data), np.std(despiked.data)))

                        # 3-day rolling window of average seafloor pressure
                        if trace_length > 5 * 24 * 60 * 60:
                            # 3 lunar days (24 hours, 50 minutes)
                            stat_window = 3 * (24 * 60 + 50) * 60
                        else:
                            stat_window = trace_length * 0.6
                        mean_sea_level.extend(nf.rolling_window_stats(despiked, window_length=stat_window,
                                                                 window_offset=stat_window / 3, full=True))

                    if tr.meta.channel == 'LKO':
                        # External temperature
                        g_log.info('Seafloor temperature {}: mean {:.3f}, min {:.3f}, max {:.3f}, stdev {:.3f}'.format(units, np.mean(despiked.data), np.min(despiked.data), np.max(despiked.data), np.std(despiked.data)))
                        # 3-day (function default) rolling window for average seafloor temperature
                        if trace_length > 5 * 24 * 60 * 60:
                            sea_temp.extend(nf.rolling_window_stats(despiked, full=True))
                        else:
                            stat_window = trace_length * 0.6
                            sea_temp.extend(nf.rolling_window_stats(despiked, window_length=stat_window,
                                                                         window_offset=stat_window / 3, full=True))

                    tr_timing.append(timeit.default_timer())
                    debug_info['timing']['trend_analysis'] += tr_timing[-1] - tr_timing[-2]
                    g_log.debug("Time spent calculating rolling window stats: {0} seconds".format(
                        (tr_timing[-1] - tr_timing[-2])))

                    # Summary statistics
                    print("{0} | {1} - {2} | {3} | Average {4:.3f} {5} | Average despiked {6:.3f} {5}".format(
                        tr.id,
                        tr.meta.starttime.strftime('%Y-%m-%d %H:%M:%S.%f'),
                        tr.meta.endtime.strftime('%Y-%m-%d %H:%M:%S.%f'),
                        channel_name,
                        np.mean(tr.data),
                        units,
                        np.mean(despiked.data)
                    ))

        except Exception as e:
            error_count += 1
            msg = str(e)
            g_log.error(traceback.format_exc())
            g_log.error(msg)

    timing_points.append(timeit.default_timer())
    g_log.debug("Time spent processing data files: {0} seconds".format((timing_points[-1] - timing_points[-2])))
    debug_info['timing']['all_proc'] = timing_points[-1] - timing_points[-2]

    # Combine pressure/temperature statistics and make plots
    if len(mean_sea_level) > 0 or len(sea_temp) > 0:
        prs = pd.DataFrame(mean_sea_level, columns=['Start', 'End', 'Center', 'Min_Pres', 'Max_Pres', 'Avg_Pres', 'Gradient', 'R2_coef', 'Days_Deployed'])
        prs.set_index('Start', drop=False)
        tmp = pd.DataFrame(sea_temp, columns=['Start', 'End', 'Center', 'Min_Temp', 'Max_Temp', 'Avg_Temp', 'Gradient', 'R2_coef', 'Days_Deployed'])
        tmp.set_index('Start', drop=True)
        prs['Plot_Time'] = prs['Center']
        tmp['Plot_Time'] = tmp['Center']

        prs = prs.assign(Aquarius_ID=kwargs['obsId'])
        tmp = tmp.assign(Aquarius_ID=kwargs['obsId'])

        # Save statistics to CSV for further analysis
        prs_csv_name = 'pressure_stats_{0}_{1}_{2}.csv'.format(kwargs['obsId'],
                                                                pd.to_datetime(prs['Start'].min()).strftime(
                                                                    '%Y-%m-%d'),
                                                                pd.to_datetime(prs['End'].max()).strftime(
                                                                    '%Y-%m-%d'))
        prs.to_csv(os.path.join(output_dir, prs_csv_name))

        tmp_csv_name = 'temperature_stats_{0}_{1}_{2}.csv'.format(kwargs['obsId'],
                                                                pd.to_datetime(tmp['Start'].min()).strftime(
                                                                    '%Y-%m-%d'),
                                                                pd.to_datetime(tmp['End'].max()).strftime(
                                                                    '%Y-%m-%d'))
        tmp.to_csv(os.path.join(output_dir, tmp_csv_name))

        # Average pressure vs time
        avgprs_plot = os.path.join(output_dir, 'pressure_mean_{0}.png'.format(obs_log['OBS ID'].values[0]))
        fig, ax = plt.subplots(1, 1, figsize=[8, 2.5])
        prs.plot(x='Plot_Time', y='Avg_Pres', kind='line', ax=ax, xlabel='Date/Time',
                         ylabel='Average Seafloor Pressure (Pa)', legend=False)
        ax.grid(True, ls=':')
        fig.tight_layout()
        fig.savefig(avgprs_plot)
        plt.close(fig)

        # Average temperature vs time
        avgtmp_plot = os.path.join(output_dir, 'temperature_mean_{0}.png'.format(obs_log['OBS ID'].values[0]))
        fig, ax = plt.subplots(1, 1, figsize=[8, 2.5])
        tmp.plot(x='Plot_Time', y='Avg_Temp', kind='line', ax=ax, xlabel='Date/Time',
                         ylabel='Average Seafloor Temperature (C)', legend=False)
        ax.grid(True, ls=':')
        fig.tight_layout()
        fig.savefig(avgtmp_plot)
        plt.close(fig)

        # Regression calculations/plots
        # Pressure linear regression
        prs['Timestamp'] = prs['Plot_Time'].values.astype(np.int64) // 1e9    # timestamps in seconds
        reg = LinearRegression().fit(prs['Timestamp'].values.reshape(-1, 1), prs['Avg_Pres'].values)
        r2 = reg.score(prs['Timestamp'].values.reshape(-1, 1), prs['Avg_Pres'].values)
        gradient = reg.coef_[0] * 60 * 60 * 24  # convert Pa/s to Pa/day for pressure gradient
        g_log.info('Average pressure linear fit: Slope {} Pa/day, R2 {}'.format(gradient, r2))

        prs_linplot = os.path.join(output_dir, 'pressure_linearfit_{0}.png'.format(obs_log['OBS ID'].values[0]))
        prs_fit = sns.lmplot(prs, x='Days_Deployed', y='Avg_Pres', aspect=3)
        prs_fit.savefig(prs_linplot)

        """
        # Pressure exponential curve
        init_guess = np.polyfit(prs['Timestamp'].values, np.log(prs['Avg_Pres'].values), 1)
        p0 = (np.exp(init_guess[1]), init_guess[0], 0)
        params, cv = curve_fit(exp_curve, prs['Timestamp'].values, prs['Avg_Pres'].values, p0)
        # fit quality
        squaredDiffs = np.square(prs['Avg_Pres'].values - exp_curve(prs['Timestamp'].values, *params))
        squaredDiffsFromMean = np.square(prs['Avg_Pres'].values - np.mean(prs['Avg_Pres'].values))
        rSq = 1 - np.sum(squaredDiffs) / np.sum(squaredDiffsFromMean)
        g_log.info('Exponential fit equation: P = {} * e^({} * t) + {}'.format(*params))
        g_log.info('Exponential fit quality R^2 = {}'.format(rSq))

        fig, ax = plt.subplots(figsize=[12, 4])
        prs.plot('Timestamp', 'Avg_Pres', kind='scatter', ax=ax)
        ax.plot(prs['Timestamp'].values, exp_curve(prs['Timestamp'].values, *params), '--')
        fig.savefig(os.path.join(output_dir, 'pressure_expfit_{0}.png'.format(obs_log['OBS ID'].values[0])))
        """

        # Temperature
        tmp['Timestamp'] = tmp['Plot_Time'].values.astype(np.int64) // 1e9    # timestamps in seconds
        reg = LinearRegression().fit(tmp['Timestamp'].values.reshape(-1, 1), tmp['Avg_Temp'].values)
        r2 = reg.score(tmp['Timestamp'].values.reshape(-1, 1), tmp['Avg_Temp'].values)
        gradient = reg.coef_[0] * 60 * 60 * 24  # convert degC/s to degC/day for pressure gradient
        g_log.info('Average temperature linear fit: Slope {} degC/day, R2 {}'.format(gradient, r2))

        tmp_linplot = os.path.join(output_dir, 'temperature_linearfit_{0}.png'.format(obs_log['OBS ID'].values[0]))
        tmp_fit = sns.lmplot(tmp, x='Days_Deployed', y='Avg_Temp', aspect=3)
        tmp_fit.savefig(tmp_linplot)

    timing_points.append(timeit.default_timer())
    g_log.debug("Time spent checking seafloor environment stats: {0} seconds".format((timing_points[-1] - timing_points[-2])))
    debug_info['timing']['env_summary'] = timing_points[-1] - timing_points[-2]

    g_log.info("end")
    print(debug_info)
    g_log.debug(str(debug_info))


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
    parser.add_argument('--config', dest='config_path', help="Path to config file (if not using default).")
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
                config = config_handler.get_config(
                    os.path.abspath(os.path.expanduser(os.path.expandvars(args.config_path))))
        else:
            config = config_handler.get_config()

        # Copy existing config info and add/update from command line arguments
        full_config = config_handler.copy_config(config)
        # Key folders
        if base_dir is None:
            base_dir = os.path.abspath(os.path.expanduser(os.path.expandvars(
                config.get('dataset', 'base_dir', fallback=os.path.join(resource_dir, 'test_data')))))
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
        for flag, key in zip([args.debug, args.obslog_column_names], ['debug', 'logcolnames']):
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
            rec_meta = rec_meta.loc[
                (rec_meta['On-Deck Date/Time (UTC)'] == base_meta['Recovery Date/Time (UTC)'].values[0])]
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
                station_meta = \
                list(filter(lambda x: x['name'] == base_meta['Station'].values[0], project_meta['stations']))[0]
            except (KeyError, IndexError):
                g_log.info("No matching station information found in project metadata JSON.")
            finally:
                project_meta['this_deployment'] = {}
        if station_meta is not None:
            if 'deployments' in station_meta:
                if len(station_meta['deployments']) > 1:
                    if deploy_start is not None:
                        deployment = list(filter(lambda x: x['start_date'] == deploy_start.strftime('%Y-%m-%d'),
                                                 station_meta['deployments']))
                        if len(deployment) > 0:
                            project_meta['this_deployment'] = deployment[0]
                        else:
                            g_log.info("No deployment found for station {0} with start date {1}.".format(obs_identifier,
                                                                                                         deploy_start.strftime(
                                                                                                             '%Y-%m-%d')))
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

        # Gather some basic information for report
        proc_kwargs = {
            'today': datetime.now().strftime('%Y-%m-%d'),
        }
        proc_kwargs['stationName'] = base_meta['Station'].values[0]
        proc_kwargs['obsName'] = base_meta['OBS Name'].values[0]
        proc_kwargs['obsId'] = base_meta['OBS ID'].values[0]
        proc_kwargs['latitude'] = base_meta['Deployed Latitude'].values[0]
        proc_kwargs['longitude'] = base_meta['Deployed Longitude'].values[0]
        proc_kwargs['waterDepth'] = base_meta['Water Depth (m)'].values[0]

        setup_time = timeit.default_timer()
        g_log.info("Time spent parsing arguments and preparing to process data: {0} seconds".format(setup_time - t0))
        g_log.debug('Logging level: {}'.format(g_log.getEffectiveLevel()))

        # Process data files to apply clock drift correction and update metadata
        process(data_dir, base_meta, network, full_config, output_dir=output_dir, metadata=metadata_file,
                channel_map=channel_map, project_meta=project_meta, **proc_kwargs)

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
            g_log.info("Total run time: {0} seconds ({1} minutes)".format(run_time, run_time / 60))
        elif run_time < 3600 * 24:
            g_log.info("Total run time: {0} seconds ({1} hours)".format(run_time, run_time / 3600))
        else:
            g_log.info("Total run time: {0} seconds ({1} days)".format(run_time, run_time / 3600 / 24))

        logger.close_logs()

        print(gc.get_stats())
    except Exception as e:
        print(traceback.print_exc())
        parser.print_help()
        print(gc.get_stats())
        exit(1)
