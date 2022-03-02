import argparse
from glob import glob
import numpy as np
import obspy
import os
import pandas as pd
import re
import shutil
import traceback
from datetime import datetime
from obspy.io.stationxml.core import validate_stationxml

import nfsi_obs as nf
from utilities import config_handler, logger, check_nan

# Ensure resource directory exists
resource_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'resource/OBSDataPipeline')
if not os.path.isdir(resource_dir):
    os.makedirs(resource_dir)


def process(data_dir, obs_log, network_id, output_dir=None, metadata=None, channel_map=None, full=True):
    g_log.info("start")

    station_info = None
    if metadata is not None:
        g_log.info("Reading metadata from file {0}".format(metadata))
        filetype = os.path.splitext(metadata)[-1]
        if filetype == '.dataless':
            station_info = nf.metadata.read_dataless(metadata)
        elif filetype == '.xml':
            # read as StationXML format
            station_info = obspy.read_inventory(metadata)
        else:
            g_log.error("Unrecognized file format. Unable to read metadata.")
    else:
        # Search data_dir for suitable metadata file
        seed_files = glob(os.path.join(data_dir, '**/*.dataless'), recursive=True)
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

    raw_files = glob(os.path.join(data_dir, '**/*.mseed'), recursive=True)
    g_log.info("Found {0} miniSEED files in data directory and sub-folders".format(len(raw_files)))
    backup_exists = False
    if output_dir is None:
        # Make a backup copy of as-recorded raw data if no separate output directory is specified (files will be modified in-place)
        raw_dir = os.path.join(data_dir, 'raw_recorded/')
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
    g_log.info("Files contain data for {0} unique channels".format(len(np.unique(labeled_files['channel'].values))))

    for channel, files in labeled_files.groupby('channel'):
        channel_type = 'health'
        g_log.info("Begin processing channel {0}".format(channel))

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
                tr.attach_response(station_info)
                tr.remove_response()
            # Fix channel/station/network codes if necessary (N/E/Z vs 1/2/3)
            if channel_map is not None:
                ch_info = channel_map.loc[tr.id]
                for code in ['Network', 'Station', 'Location', 'Channel']:
                    if ch_info[code] is not None and ~check_nan(ch_info[code]):
                        tr.meta[code.lower()] = ch_info[code]
            if tr.meta.network != network_id:
                raise AssertionError('Channel {0} is not in network {1}'.format(tr.id, network_id))
        data.merge()
        print(data)

        # Assign to relevant group of channels (there should only be one channel in the Stream object)
        if re.match(r'[BCDEGHLMRUVW][H][1-3ABCENRTUVWZ]', data[0].meta.channel):
            # seismic data
            channel_type = 'seismic'
        elif data[0].meta.channel in ['LKO', 'MDO', 'MDU']:
            # oceanographic data (external P/T, include APG if present)
            # TODO: Would like this to be more general, but internal temperature is also labeled with "KO" source/subsource code by default
            channel_type = 'ocean'
        elif data[0].meta.channel in ['LE3', 'ME4']:
            # battery voltage and power consumption
            channel_type = 'power'

        # Noise level QC steps (seismic channels and hydrophone) -> if channel code == "CHx" or "HDF"
        if channel_type == 'seismic':
            if full:
                # TODO: Decide if the same operations are appropriate for the hydrophone data or not
                # TODO: Calculate hourly PSDs
                # TODO: Average PSD value at 0.2 Hz (save out for comparison with other sensors in the same network)
                # TODO: Linearity of PSD curves
                g_log.warning("Full QC of seismic noise not yet implemented")
        else:
            # Analysis of auxiliary data
            full_data_plot = os.path.join(output_dir, '{0}_full.png'.format(data[0].id))
            data.plot(outfile=full_data_plot)
            # maybe smooth out state-of-health channels? or come up with some way to automatically QC them for anomalous sections

            # print(data[0].stats)

            # TODO: Plot battery draw-down and power consumption over full deployment
            # TODO: Plot internal state-of-health variables: pressure, temperature, humidity
            # TODO: Down-sample external pressure and temperature data (plot and save as netCDF)

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
        base_meta = obs_log_info['basic']
        # Find this OBS in the basic metadata table
        row = None
        if id_type == 'serial':
            row = base_meta.loc[base_meta['OBS ID'] == obs_identifier]
        elif id_type == 'obs_name':
            row = base_meta.loc[base_meta['OBS Name'] == obs_identifier]
        else:
            id_columns = ['Station', 'OBS Name', 'OBS ID']
            for col in id_columns:
                if obs_identifier in base_meta[col].values:
                    row = base_meta.loc[base_meta[col] == obs_identifier]
                    break

        if row is None or row.empty:
            raise IndexError('OBS {0} not found in provided metadata.'.format(obs_identifier))
        if row.shape[0] > 1:
            raise IndexError('Multiple entries found for OBS {0} in provided metadata. Please use a unique identifier.'.format(obs_identifier))

        channel_map = None
        if args.channel_map:
            channel_map = nf.io.read_channel_map(args.channel_map)

        metadata_file = None
        if args.metadata_file:
            metadata_file = os.path.abspath(os.path.expanduser(os.path.expandvars(args.metadata_file)))

        # Process data files to apply clock drift correction and update metadata
        process(data_dir, row, args.network_id, output_dir, metadata_file, channel_map, ~args.function_check)

        g_log.info("Processing complete!")
        logger.close_logs()
    except Exception as e:
        print(traceback.print_exc())
        parser.print_help()
        exit(1)
