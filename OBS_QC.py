import argparse
from glob import glob
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

import nfsi_obs as nf
from utilities import config_handler, logger, check_nan, ReportGenerator

# Ensure resource directory exists
resource_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'resource/OBSDataPipeline')
if not os.path.isdir(resource_dir):
    os.makedirs(resource_dir)


def process(data_dir, obs_log, network_id, output_dir=None, metadata=None, channel_map=None, full=True, detrend=False, **kwargs):
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
    g_log.info("Files contain data for {0} unique set of channels".format(len(np.unique(labeled_files['channel'].values))))

    # Loop through data files
    for label, files in labeled_files.groupby('channel'):
        g_log.info("Begin processing channel set {0}".format(label))

        data = obspy.Stream()
        for rf in files['path'].values:
            temp = obspy.read(rf)
            for tr in temp:
                data.append(tr)
        data.merge()
        # print(data)

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
            if channel_map is None:
                g_log.info("No channel map provided. Checking data directory for project_info.json...")
                # Search data_dir for project JSON (should have channel descriptions)
                # TODO: Replace with ST integration once we have an instance running
                project_json = os.path.join(data_dir, 'project_info.json')
                if os.path.isfile(project_json):
                    project_info = json.load(project_json)
                    try:
                        channel_info = list(filter(lambda ch: ch['channel_id'] == tr.meta.channel, project_info['channels']))[0]
                        tr.meta.description = channel_info['description']
                    except (KeyError, IndexError):
                        g_log.warn("No matching description found in project metadata for channel {0}".format(tr.id))
                else:
                    g_log.info("No project metadata JSON found at {0}".format(project_json))
            else:
                ch_info = channel_map.loc[tr.id]
                for code in ['Network', 'Station', 'Location', 'Channel', 'Description']:
                    if ch_info[code] is not None and ~check_nan(ch_info[code]):
                        tr.meta[code.lower()] = ch_info[code]
            if tr.meta.network != network_id:
                raise AssertionError('Channel {0} is not in network {1}'.format(tr.id, network_id))
        data.merge()
        print(data)

        # Cut data to time on seafloor (if start/end times provided)
        start, end = None, None
        if ~pd.isnull(obs_log['Date/Time on Seafloor (UTC)'].values[0]):
            start = obspy.UTCDateTime(pd.to_datetime(obs_log['Date/Time on Seafloor (UTC)'].values[0]))
        if ~pd.isnull(obs_log['Date/Time Released (UTC)'].values[0]):
            end = obspy.UTCDateTime(pd.to_datetime(obs_log['Date/Time Released (UTC)'].values[0]))

        data = data.slice(start, end, nearest_sample=False)

        # Perform QC
        if len(data.traces) > 1:
            # Multiple channels in one miniSEED file
            seismic = obspy.Stream()
            ocean = obspy.Stream()
            power = obspy.Stream()
            health = obspy.Stream()
            for tr in data:
                # Assign to relevant group of channels (there should only be one channel in the Stream object)
                if (re.match(r'[A-Z]H[1-3ABCENRTUVWZ]', tr.meta.channel)) or (re.match(r'[A-Z]D[HF]', data[0].meta.channel)):
                    # seismic data and hydrophone
                    seismic.append(tr)
                elif tr.meta.channel in ['LKO', 'MDO', 'MDU']:
                    # oceanographic data (external P/T, include APG if present)
                    # TODO: Would like this to be more general, but internal temperature is also labeled with "KO" source/subsource code by default
                    ocean.append(tr)
                elif tr.meta.channel in ['LE3', 'ME4']:
                    # battery voltage and power consumption
                    power.append(tr)
                else:
                    health.append(tr)

            # Noise level QC steps (seismic channels and hydrophone) -> if channel code == "CHx" or "HDF"
            raw_data_plot = os.path.join(output_dir, 'raw_seismic_{0}.png'.format(network_id))
            seismic.plot(outfile=raw_data_plot)

            for tr in seismic:
                if hasattr(tr.meta, 'response'):
                    tr.remove_sensitivity()

            full_data_plot = os.path.join(output_dir, 'full_seismic_{0}.png'.format(network_id))
            seismic.plot(outfile=full_data_plot)

            if detrend:
                # detrend seismic data (RMS linear fit)
                seismic.detrend('linear')
                demean_data_plot = os.path.join(output_dir, 'demean_seismic_{0}.png'.format(network_id))
                seismic.plot(outfile=demean_data_plot)

            for tr in seismic:
                trace_info = {
                    'seedID': tr.id,
                    'channelName': tr.id,
                    'azimuth': tr.meta.azimuth,
                    'dip': tr.meta.dip,
                    'windowSecs': 3600,
                    'overlapPercent': 75
                }
                if hasattr(tr.meta, 'description'):
                    trace_info['channelName'] = tr.meta.description

                # Plot each trace individually for QC report
                trace_plot = os.path.join(output_dir, 'full_seismic_{0}.png'.format(tr.id))
                tr.plot(outfile=trace_plot)
                trace_info['traceLoc'] = trace_plot

                # Plot spectrogram of full time period
                # TODO: Have window length chosen automatically based on length of time period
                # TODO: Deal with RuntimeWarning for divide by zero (due to dbscale?)
                spectrogram_plot = os.path.join(output_dir, 'spec_seismic_{0}.png'.format(tr.id))
                tr.spectrogram(per_lap=0.75, wlen=60, dbscale=True, log=True, outfile=spectrogram_plot)
                trace_info['specLoc'] = spectrogram_plot

                # Plot PSDs of data
                # TODO: Have window length chosen automatically based on length of time period
                psd_v_plot = os.path.join(output_dir, 'psd_seismic_vel_{0}.png'.format(tr.id))
                psd_a_plot = os.path.join(output_dir, 'psd_seismic_acc_{0}.png'.format(tr.id))
                freqs, psds = [], []
                psd_v_fig, vax = plt.subplots(1, 1)
                for sect in tr.slide(3600, 900):
                    seg_len = pow(2, 17)
                    psd, frq = plt.psd(sect.data, NFFT=seg_len, Fs=tr.meta.sampling_rate, window=signal.get_window('hamming', seg_len, False), detrend='linear', color='0.7', linewidth=0.5)
                    freqs.append(frq)
                    psds.append(psd)
                vax.set_xscale('log')
                psd_v_fig.savefig(psd_v_plot)

                # Convert PSDs to acceleration and plot
                psd_a_fig, aax = plt.subplots(1, 1)
                for f, p in zip(freqs, psds):
                    apsd = p * (2 * np.pi * f) * (2 * np.pi * f)
                    aax.plot(f, 10 * np.log10(apsd), c='0.8', lw=0.5, marker=None)
                aax.set_xscale('log')
                plt.grid(True, ls=':')
                psd_a_fig.savefig(psd_a_plot)
                trace_info['psdLoc'] = psd_a_plot

                report_params['seismic_channels'].append(trace_info)

            if full:
                # TODO: Decide if these operations are appropriate for the hydrophone data or not
                # TODO: Calculate hourly PSDs
                # TODO: Average PSD value at 0.2 Hz (save out for comparison with other sensors in the same network)
                # TODO: Linearity of PSD curves
                g_log.warning("Full QC of seismic noise not yet implemented")

            for data, description in zip([ocean, power, health], ['ocean', 'power', 'health']):
                # TODO: Make vertical scales for each channel appropriate
                # Analysis of auxiliary data
                raw_data_plot = os.path.join(output_dir, 'raw_{0}_{1}.png'.format(description, network_id))
                data.plot(outfile=raw_data_plot)

                # Apply instrument sensitivity
                sens_applied = False
                for tr in data:
                    if hasattr(tr.meta, 'response'):
                        tr.remove_sensitivity()
                        sens_applied = True
                if sens_applied:
                    # TODO: Replace with custom plotting routine
                    full_data_plot = os.path.join(output_dir, 'full_{0}_{1}.png'.format(description, network_id))
                    fig = data.plot(show=False, handle=True)
                    for i in range(len(data.traces)):
                        if hasattr(data.traces[i].meta, 'description'):
                            ax = fig.axes[i]
                            ax.set_ylabel("{0} ({1})".format(data.traces[i].meta.description, data.traces[i].meta.response.instrument_sensitivity.input_units))
                    plt.grid(True, ls=':')
                    fig.savefig(full_data_plot)
                    plt.close(fig)

                # maybe smooth out state-of-health channels? or come up with some way to automatically QC them for anomalous sections

                # Summary statistics
                for tr in data:
                    trace_info = {
                        'seedID': tr.id,
                        'channelName': tr.id,
                    }
                    if hasattr(tr.meta, 'description'):
                        trace_info['channelName'] = tr.meta.description

                    # Plot each trace individually for QC report
                    trace_plot = os.path.join(output_dir, 'full_{0}_{1}.png'.format(description, tr.id))
                    tr.plot(outfile=trace_plot)
                    trace_info['traceLoc'] = trace_plot

                    if hasattr(tr.meta, 'response'):
                        units = tr.meta.response.instrument_sensitivity.input_units
                    else:
                        units = ''
                    print("{0} | {1} - {2} | Average {3:.3f} {4}".format(
                        tr.id,
                        tr.meta.starttime.strftime('%Y-%m-%d %H:%M:%S.%f'),
                        tr.meta.endtime.strftime('%Y-%m-%d %H:%M:%S.%f'),
                        np.mean(tr.data),
                        units
                    ))

                    report_params[description + '_channels'].append(trace_info)
                # print(data[0].stats)

                if description == 'power':
                    for tr in data:
                        if tr.meta.channel == 'LE3':
                            report_params['meanPower'] = np.mean(tr.data)

                # TODO: Analysis of state-of-health variables?
                # TODO: Down-sample external pressure and temperature data (plot and save as netCDF)
        else:
            # Single channel per miniSEED file
            channel_type = 'health'
            # Assign to relevant group of channels (there should only be one channel in the Stream object)
            if (re.match(r'[A-Z]H[1-3ABCENRTUVWZ]', data[0].meta.channel)) or (re.match(r'[A-Z]D[HF]', data[0].meta.channel)):
                # seismic data and hydrophone
                channel_type = 'seismic'
            elif data[0].meta.channel in ['LKO', 'MDO', 'MDU']:
                # oceanographic data (external P/T, include APG if present)
                # TODO: Would like this to be more general, but internal temperature is also labeled with "KO" source/subsource code by default
                channel_type = 'ocean'
            elif data[0].meta.channel in ['LE3', 'ME4']:
                # battery voltage and power consumption
                channel_type = 'power'

            trace_info = {
                'seedID': data[0].id,
                'channelName': data[0].id,
            }
            if hasattr(data[0].meta, 'description'):
                trace_info['channelName'] = data[0].meta.description

            # Noise level QC steps (seismic channels and hydrophone) -> if channel code == "CHx" or "HDF"
            if channel_type == 'seismic':
                trace_info.update({
                    'azimuth': data[0].meta.azimuth,
                    'dip': data[0].meta.dip,
                    'windowSecs': 3600,
                    'overlapPercent': 75,
                })

                for tr in data:
                    if hasattr(tr.meta, 'response'):
                        tr.remove_sensitivity()

                full_data_plot = os.path.join(output_dir, 'full_seismic_{0}.png'.format(data[0].id))
                data.plot(outfile=full_data_plot)
                trace_info['traceLoc'] = full_data_plot

                # detrend
                data.detrend('linear')
                demean_data_plot = os.path.join(output_dir, 'demean_{0}.png'.format(data[0].id))
                data.plot(outfile=demean_data_plot)

                spectrogram_plot = os.path.join(output_dir, 'spec_{0}.png'.format(data[0].id))
                data.spectrogram(per_lap=0.5, wlen=60, outfile=spectrogram_plot)
                trace_info['specLoc'] = spectrogram_plot

                # Plot PSDs of data
                # TODO: Have window length chosen automatically based on length of time period
                psd_v_plot = os.path.join(output_dir, 'psd_seismic_vel_{0}.png'.format(data[0].id))
                psd_a_plot = os.path.join(output_dir, 'psd_seismic_acc_{0}.png'.format(data[0].id))
                freqs, psds = [], []
                psd_v_fig, vax = plt.subplots(1, 1)
                for sect in data[0].slide(3600, 900):
                    seg_len = pow(2, 17)
                    psd, frq = plt.psd(sect.data, NFFT=seg_len, Fs=data[0].meta.sampling_rate, window=signal.get_window('hamming', seg_len, False), detrend='linear', color='0.7', linewidth=0.5)
                    freqs.append(frq)
                    psds.append(psd)
                vax.set_xscale('log')
                psd_v_fig.savefig(psd_v_plot)

                # Convert PSDs to acceleration and plot
                psd_a_fig, aax = plt.subplots(1, 1)
                for f, p in zip(freqs, psds):
                    apsd = p * (2 * np.pi * f) * (2 * np.pi * f)
                    aax.plot(f, 10 * np.log10(apsd), c='0.8', lw=0.5, marker=None)
                aax.set_xscale('log')
                plt.grid(True, ls=':')
                psd_a_fig.savefig(psd_a_plot)
                trace_info['psdLoc'] = psd_a_plot

                if full:
                    # TODO: Decide if the same operations are appropriate for the hydrophone data or not
                    # TODO: Calculate hourly PSDs
                    # TODO: Average PSD value at 0.2 Hz (save out for comparison with other sensors in the same network)
                    # TODO: Linearity of PSD curves
                    g_log.warning("Full QC of seismic noise not yet implemented")

            else:
                # Analysis of auxiliary data
                raw_data_plot = os.path.join(output_dir, 'raw_{0}.png'.format(data[0].id))
                data.plot(outfile=raw_data_plot)

                # Apply instrument sensitivity
                sens_applied = False
                for tr in data:
                    if hasattr(tr.meta, 'response'):
                        tr.remove_sensitivity()
                        sens_applied = True
                if sens_applied:
                    # TODO: Replace with custom plotting routine
                    full_data_plot = os.path.join(output_dir, 'full_{0}.png'.format(data[0].id))
                    fig = data.plot(show=False, handle=True)
                    for i in range(len(data.traces)):
                        if hasattr(data.traces[i].meta, 'description'):
                            ax = fig.axes[i]
                            ax.set_ylabel("{0} ({1})".format(data.traces[i].meta.description, data.traces[i].meta.response.instrument_sensitivity.input_units))
                    plt.grid(True, ls=':')
                    fig.savefig(full_data_plot)
                    plt.close(fig)
                    trace_info['traceLoc'] = full_data_plot

                # maybe smooth out state-of-health channels? or come up with some way to automatically QC them for anomalous sections

                # Summary statistics
                for tr in data:
                    if hasattr(tr.meta, 'response'):
                        units = tr.meta.response.instrument_sensitivity.input_units
                    else:
                        units = ''
                    print("{0} | {1} - {2} | Average {3:.3f} {4}".format(
                        tr.id,
                        tr.meta.starttime.strftime('%Y-%m-%d %H:%M:%S.%f'),
                        tr.meta.endtime.strftime('%Y-%m-%d %H:%M:%S.%f'),
                        np.mean(tr.data),
                        units
                    ))
                # print(data[0].stats)

                # TODO: Analysis of state-of-health variables?
                # TODO: Down-sample external pressure and temperature data (plot and save as netCDF)

            report_params[channel_type + '_channels'].append(trace_info)

    # Save report to *.md and *.pdf formats
    report_md = os.path.join(output_dir, 'QC_report_{0}_auto.md'.format(obs_log['OBS ID'].values[0]))
    qcReport = ReportGenerator(type='qc')
    md_out, report_buffer = qcReport.write_report(report_params, report_md)

    report_pdf = os.path.join(output_dir, 'QC_report_{0}_auto.pdf'.format(obs_log['OBS ID'].values[0]))
    report_converted = pypandoc.convert_text(report_buffer, to='pdf', format='md', outputfile=report_pdf)

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

    try:
        args = parser.parse_args()
        start_time = datetime.now()
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

        # Gather some basic information for report
        report_kwargs = {}
        if args.project_name:
            report_kwargs['projectName'] = args.project_name
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
        report_kwargs['introText'] = ''

        # Process data files to apply clock drift correction and update metadata
        process(data_dir, base_meta, args.network_id, output_dir, metadata_file, channel_map, ~args.function_check, args.detrend_seis, **report_kwargs)

        g_log.info("Processing complete!")
        logger.close_logs()
    except Exception as e:
        print(traceback.print_exc())
        parser.print_help()
        exit(1)
