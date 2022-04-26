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

import nfsi_obs as nf
from utilities import config_handler, logger, check_nan, ReportGenerator

current_dir = os.path.dirname(os.path.abspath(__file__))
# Ensure resource directory exists
resource_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'resource/OBSDataPipeline')
if not os.path.isdir(resource_dir):
    os.makedirs(resource_dir)


def process(data_dir, obs_log, network_id, config, output_dir=None, metadata=None, channel_map=None, project_meta=None, full=True, detrend=False, **kwargs):
    """
    Extra keyword arguments are included as report parameters (must match variables in template file).
    """
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

    # Find data files and backup if necessary
    # TODO: Remove file backup here once it has been copied to pre-processing script (QC doesn't change miniSEED files)
    raw_files = glob(os.path.join(data_dir, '**/*.mseed'), recursive=True)
    g_log.info("Found {0} miniSEED file(s) in data directory and sub-folders".format(len(raw_files)))
    backup_exists = False
    if output_dir is None:
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
        output_dir = data_dir

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

    all_gaps = []
    centring = pd.DataFrame()

    # Loop through data files (grouped by channel set)
    for label, files in labeled_files.groupby('channel'):
        g_log.info("Begin processing channel set {0}".format(label))

        # Read all files in list
        data = obspy.Stream()
        for rf in files['path'].values:
            temp = obspy.read(rf)
            for tr in temp:
                data.append(tr)
        data.merge()

        # Populate metadata from other files as necessary
        for tr in data:
            # Get response info from metadata
            if station_info is not None:
                try:
                    tr.attach_response(station_info)
                except ValueError:
                    warnings.warn("No matching response information found")

                # Get orientations of seismic channels
                if re.match(r'[A-Z]H[1-3ABCENRTUVWZ]', tr.meta.channel):
                    orient = station_info.get_orientation(tr.id)
                    for key in ['azimuth', 'dip']:
                        tr.stats[key] = orient[key]

            # Fix channel/station/network codes if necessary (N/E/Z vs 1/2/3)
            if channel_map is not None:
                ch_info = channel_map.loc[tr.id]
                for code in ['Network', 'Station', 'Location', 'Channel', 'Description']:
                    if ch_info[code] is not None and ~check_nan(ch_info[code]):
                        tr.meta[code.lower()] = ch_info[code]
            if tr.meta.network != network_id:
                raise AssertionError('Channel {0} is not in network {1}'.format(tr.id, network_id))

            # Get channel info from project metadata JSON
            if project_meta is not None:
                try:
                    channel_info = list(filter(lambda ch: ch['channel_id'] == tr.meta.channel, project_meta['channels']))[0]
                    if 'description' in channel_info:
                        tr.meta.description = channel_info['description']
                except (KeyError, IndexError):
                    g_log.warning("No matching information found in project metadata for channel {0}".format(tr.id))

        data.merge()
        print(data)

        # Cut data to time on seafloor (if start/end times provided)
        start, end = None, None
        if not pd.isnull(obs_log['Date/Time on Seafloor (UTC)'].values[0]):
            start = obspy.UTCDateTime(pd.to_datetime(obs_log['Date/Time on Seafloor (UTC)'].values[0]))
        if not pd.isnull(obs_log['Date/Time Released (UTC)'].values[0]):
            end = obspy.UTCDateTime(pd.to_datetime(obs_log['Date/Time Released (UTC)'].values[0]))

        data = data.slice(start, end, nearest_sample=False)

        # Perform QC
        # TODO: Combine single and multi-channel cases to simplify code (no real reason to separate) -> TEST
        seismic = obspy.Stream()
        ocean = obspy.Stream()
        power = obspy.Stream()
        health = obspy.Stream()

        # Gap test
        gaps = data.get_gaps()
        all_gaps.extend(gaps)
        if len(gaps) > 0:
            g_log.info('Found {0} gaps or overlaps in recorded data'.format(len(gaps)))
            data.print_gaps()

        for tr in data:
            # Timing check
            if not iq_utils.check_timestamps(tr.times()):
                g_log.warning("One or more timestamps are not in chronological order.")

            # Assign to relevant group of channels
            channel_type = 'health'
            if (re.match(r'[A-Z]H[1-3ABCENRTUVWZ]', tr.meta.channel)) or (re.match(r'[A-Z]D[HF]', tr.meta.channel)):
                # seismic data and hydrophone
                channel_type = 'seismic'
                seismic.append(tr)
            elif tr.meta.channel in ['LKO', 'MDO', 'MDU']:
                # oceanographic data (external P/T, include APG if present)
                # TODO: Would like this to be more general, but internal temperature is also labeled with "KO" source/subsource code by default
                channel_type = 'ocean'
                ocean.append(tr)
            elif tr.meta.channel in ['LE3', 'ME4']:
                # battery voltage and power consumption
                channel_type = 'power'
                power.append(tr)
            else:
                health.append(tr)

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

            # Time series plot
            trace_info['traceLoc'] = nf.plotting.trace_plot(tr, output_dir, dmin, dmax, qc_config)

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

                        # TODO: Check how often instrument centres (save flat-line test results for all 3 and compare later)
                        if re.match(r'[A-Z]M[1-3ENZ]', tr.meta.channel) and ('flat_line_test' in qc_config['qartod']):
                            # centring channels only, must have flat-line test criteria specified
                            flatline = qartod.flat_line_test(tr.data, **qc_config['qartod']['flat_line_test'])
                            centring[tr.id] = pd.Series(flatline, index=timestamps)

                if channel_type == 'power':
                    if tr.meta.channel == 'LE3':
                        report_params['meanPower'] = '{:.3f}'.format(np.mean(tr.data))

                        # TODO: Get times of data writes (spikes 45 minutes apart)
                        if (qc_config is not None) and ('qartod' in qc_config) and ('spike_test' in qc_config['qartod']):
                            spikes = qartod.spike_test(tr.data, **qc_config['qartod']['spike_test'])

                # TODO: Analysis of state-of-health variables?
                # TODO: Down-sample external pressure and temperature data (plot and save as netCDF)

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

        # TODO: Decide whether to keep this section. Don't actually use these plots.
        for data, description in zip([seismic, ocean, power, health], ['seismic', 'ocean', 'power', 'health']):
            # Noise level QC steps (seismic channels and hydrophone) -> if channel code == "CHx" or "HDF"
            raw_data_plot = os.path.join(output_dir, 'raw_{0}_{1}.png'.format(description, network_id))
            data.plot(outfile=raw_data_plot)

            # Apply instrument sensitivity
            sens_applied = False
            for tr in data:
                if hasattr(tr.meta, 'response'):
                    tr.remove_sensitivity()
                    sens_applied = True
            if sens_applied:
                # TODO: Make vertical scales for each channel appropriate
                # TODO: Replace with custom plotting routine
                full_data_plot = os.path.join(output_dir, 'full_{0}_{1}.png'.format(description, network_id))
                fig = data.plot(show=False, handle=True)
                for i in range(len(data.traces)):
                    if hasattr(data.traces[i].meta, 'description'):
                        ax = fig.axes[i]
                        ax.set_ylabel("{0} ({1})".format(data.traces[i].meta.description, data.traces[
                            i].meta.response.instrument_sensitivity.input_units))
                plt.grid(True, ls=':')
                fig.savefig(full_data_plot)
                plt.close(fig)

            if detrend and (description == 'seismic'):
                # detrend seismic data (RMS linear fit)
                data.detrend('linear')
                demean_data_plot = os.path.join(output_dir, 'demean_seismic_{0}.png'.format(network_id))
                data.plot(outfile=demean_data_plot)

    # Sort channel information by specified order
    for ch_type in ['seismic', 'ocean', 'power', 'health']:
        sorted_channels = sorted(report_params[ch_type + '_channels'], key=lambda d: d['order'])
        report_params[ch_type + '_channels'] = sorted_channels

    # Save report to *.md and *.pdf formats
    # TODO: Add gap information to report (all_gaps list should cover all traces)
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

    g_log.info("end")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform basic QC for OBS data. Will correct channel identifiers if '
                                                 'optional --channelmap argument is provided. Does not require clock '
                                                 'drift correction to have been applied.')
    parser.add_argument('--data_dir', dest="data_dir", help="Directory where OBS data is stored.")
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
    parser.add_argument('--function_check', dest="function_check", action="store_true",
                        help="Perform basic QC to check Aquarius functionality only. False by default to perform full "
                             "QC.")
    parser.add_argument('--detrend_seismic', dest="detrend_seis", action="store_true",
                        help="Detrend seismic data (RMS linear fit). False by default.")
    # TODO: When using ST, project name will come from there instead
    parser.add_argument('--projectname', dest="project_name", help="Project name to be displayed in reports")
    parser.add_argument('--config', dest='config_path', help="Path to config file (if not using default).")

    try:
        start_time = datetime.now()
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
            output_dir = os.path.abspath(os.path.expanduser(os.path.expandvars(args.outdir)))

        if args.datalog:
            data_log_file = os.path.abspath(os.path.expanduser(os.path.expandvars(args.datalog)))
        else:
            data_log_file = os.path.join(data_dir, 'log.xlsx')

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
            channel_map = nf.io.read_channel_map(args.channel_map)

        metadata_file = None
        if args.metadata_file:
            metadata_file = os.path.abspath(os.path.expanduser(os.path.expandvars(args.metadata_file)))

        if args.config_path:
            config = config_handler.get_config(os.path.abspath(os.path.expanduser(os.path.expandvars(args.config_path))))
        else:
            config = config_handler.get_config()

        # Read project metadata JSON file
        project_meta = None
        # TODO: Replace with ST integration once we have an instance running
        if channel_map is None:
            g_log.info("No channel map provided. Checking data directory for project_info.json...")
        else:
            g_log.info("Reading project metadata from [data_dir]/project_info.json...")
        # Search data_dir for project JSON (should have channel descriptions)
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
            report_kwargs['introText'] = station_meta['qc_intro']
        report_kwargs['psdWindowSecs'] = int(config.get('seismic', 'window_length'))
        report_kwargs['psdOverlapPercent'] = int(config.get('seismic', 'overlap_percent'))

        # Process data files to apply clock drift correction and update metadata
        process(data_dir, base_meta, args.network_id, config, output_dir, metadata_file, channel_map, project_meta, ~args.function_check, args.detrend_seis, **report_kwargs)

        g_log.info("Processing complete!")
        end_time = datetime.now()
        run_time = end_time - start_time
        g_log.info("Total run time: {0} seconds".format(run_time.total_seconds()))

        logger.close_logs()
    except Exception as e:
        print(traceback.print_exc())
        parser.print_help()
        exit(1)
