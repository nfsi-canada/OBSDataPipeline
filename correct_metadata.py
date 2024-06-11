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

# TODO: Update for StationXML created on Aquarius

# Ensure resource directory exists
resource_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'resource/OBSDataPipeline')
if not os.path.isdir(resource_dir):
    os.makedirs(resource_dir)


def fix_metadata(data_dir, obs_log, network_id, output_dir=None, channel_map=None, dataless=None):
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

    if dataless is None:
        dataless_files = glob(os.path.join(data_dir, '**.*.dataless'), recursive=True)
        if len(dataless_files) < 1:
            g_log.warn("No metadata file found.")
        elif len(dataless_files) > 1:
            raise IOError("Multiple metadata files found. Please specify a file using the --metadata argument.")
        else:
            dataless = dataless_files[0]
    xml_meta = nf.metadata.convert_dataless_to_stationxml(dataless, obs_log, output_dir, channel_map)
    # TODO: Sensor orientation (azimuth and dip in Channel object of StationXML)
    g_log.info("Converted metadata to StationXML format: {0}".format(xml_meta))

    g_log.info("Reading data files...")
    ocean_data = obspy.Stream()
    seismic_data = obspy.Stream()
    state_of_health = obspy.Stream()
    for rf in raw_files:
        g_log.info("Begin processing file {0}".format(rf))

        # TODO: Apply clock drift and identifier correction to raw miniSEED file and copy to output_dir
        mseed_file = nf.mseed.MiniSEED(rf)

        data = obspy.read(rf)
        print(data)

        # Metadata and pre-processing
        for tr in data:
            tr.meta.network = network_id
            # Fix channel/station/network codes if necessary (N/E/Z vs 1/2/3)
            if channel_map is not None:
                ch_info = channel_map.loc[tr.id]
                if ch_info['Network'] != network_id:
                    raise AssertionError('Corrected network ID {0} in channel map does not match input --network argument {1}'.format(ch_info['Network'], network_id))
                for code in ['Station', 'Location', 'Channel']:
                    if ch_info[code] is not None and ~np.isnan(ch_info[code]):
                        tr.meta[code.lower()] = ch_info[code]

            if re.match(r'CH[0-9A-F]', tr.meta.channel):
                # seismic data
                seismic_data.append(tr)
            elif tr.meta.channel in ['LKO', 'MDO']:
                # oceanographic data (external P/T)
                ocean_data.append(tr)
            else:
                # all other channels
                state_of_health.append(tr)

    # Combine traces with the same ID
    for stm in [seismic_data, ocean_data, state_of_health]:
        stm.merge()

    g_log.info("end")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pre-process OBS data and perform basic QC')
    parser.add_argument('--data_dir', dest="data_dir", help="Directory where raw OBS data is stored.")
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
    parser.add_argument('--metadata', dest="metadata_file", help="Path to metadata file (dataless SEED or StationXML)")

    try:
        args = parser.parse_args()
        start_time = datetime.now()
        obs_id = args.obs_id

        # TODO: Allow user-configurable log directory
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
        if isinstance(row, pd.DataFrame):
            raise IndexError('Multiple entries found for OBS {0} in provided metadata. Please use a unique identifier.'.format(obs_identifier))

        channel_map = None
        if args.channel_map:
            channel_map = nf.io.read_channel_map(args.channel_map)

        metadata_file = None
        if args.metadata_file:
            metadata_file = os.path.abspath(os.path.expanduser(os.path.expandvars(args.metadata_file)))

        # Process data files to apply clock drift correction and update metadata
        fix_metadata(data_dir, row, args.network_id, output_dir, channel_map, metadata_file)

        g_log.info("Processing complete!")
        logger.close_logs()
    except Exception as e:
        print(traceback.print_exc())
        parser.print_help()
        exit(1)
