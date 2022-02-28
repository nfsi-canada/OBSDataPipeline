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

import nfsi_obs as nf
from utilities import config_handler, logger

# Ensure resource directory exists
resource_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'resource/OBSDataPipeline')
if not os.path.isdir(resource_dir):
    os.makedirs(resource_dir)


def process(data_dir, obs_log, network_id, output_dir=None, dataless=None, channel_map=None, full=True):
    g_log.info("start")

    raw_files = glob(os.path.join(data_dir, '**/*.mseed'), recursive=True)
    g_log.info("Found {0} miniSEED files in data directory and sub-folders".format(len(raw_files)))
    if output_dir is None:
        # Make a backup copy of as-recorded raw data if no separate output directory is specified (files will be modified in-place)
        raw_dir = os.path.join(data_dir, 'raw_recorded/')
        os.makedirs(raw_dir)
        for rf in raw_files:
            shutil.copy2(rf, raw_dir)
        output_dir = data_dir

    # label files by channel name
    labels = []
    for rf in raw_files:
        file_name = re.split(r'/|\\', rf)[-1]
        ch_name = file_name.split('_')[1]
        labels.append({'channel': ch_name, 'path': rf})
    labeled_files = pd.DataFrame(labels)

    g_log.info("Reading data files...")
    for channel, files in labeled_files.groupby('channel'):
        channel_type = 'health'
        g_log.info("Begin processing channel {0}".format(channel))

        data = obspy.Stream()
        for rf in files['path'].values:
            temp = obspy.read(rf)
            for tr in temp:
                data.append(tr)
        data.merge()
        print(data)

        for tr in data:
            # Fix channel/station/network codes if necessary (N/E/Z vs 1/2/3)
            if channel_map is not None:
                ch_info = channel_map.loc[tr.id]
                for code in ['Network', 'Station', 'Location', 'Channel']:
                    if ch_info[code] is not None and ~np.isnan(ch_info[code]):
                        tr.meta[code.lower()] = ch_info[code]
            if tr.meta.network != network_id:
                raise (AssertionError, 'Channel {0} is not in network {1}'.format(tr.id, network_id))
        data.merge()
        print(data)

        # Assign to relevant group of channels (there should only be one channel in the Stream object)
        if re.match(r'[BCDEGHLMRUVW][HM][0-9A-F]', data[0].meta.channel):
            # seismic data and mass position channels
            channel_type = 'seismic'
        elif data[0].meta.channel in ['LKO', 'MDO', 'MDU']:
            # oceanographic data (external P/T, include APG if present)
            # TODO: Would like this to be more general, but internal temperature is also labeled with "KO" source/subsource code by default
            channel_type = 'ocean'


    # Basic QC steps (seismic channels and hydrophone) -> if channel code == "CHx" or "HDF"
    # TODO: Decide if the same operations are appropriate for the hydrophone data or not
    # TODO: Calculate hourly PSDs
    # TODO: Average PSD value at 0.2 Hz (save out for comparison with other sensors in the same network)
    # TODO: Linearity of PSD curves

    # Analysis of auxiliary data
    # maybe smooth out state-of-health channels? or come up with some way to automatically QC them for anomalous sections
    for tr in state_of_health:
        if re.match(r'[A-Z]M[1-3A-Z]', tr.meta.channel):
            # mass position channel
            continue

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
    parser.add_argument('--network', "network_id", default='XX',
                        help="Network identifier assigned by FDSN for this project. Default 'XX' for test data.")
    parser.add_argument('--outdir', dest="outdir", default=None,
                        help="Output directory, if different from data directory")
    parser.add_argument('--channelmap', dest="channel_map",
                        help="File mapping as-recorded channel codes to their correct values.")
    parser.add_argument('--metadata', dest="metadata_file", help="Path to metadata file (dataless SEED or StationXML)")
    parser.add_argument('--function_check', dest="function_check", action="store_true",
                        help="Perform basic QC to check Aquarius functionality only. False by default to perform full QC.")

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
            raise(IndexError, 'OBS {0} not found in provided metadata.'.format(obs_identifier))
        if isinstance(row, pd.DataFrame):
            raise(IndexError, 'Multiple entries found for OBS {0} in provided metadata. Please use a unique identifier.'.format(obs_identifier))

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
