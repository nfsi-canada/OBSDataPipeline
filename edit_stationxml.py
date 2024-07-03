"""
Add missing info to existing StationXML file(s) from Aquarius, and reduce to only channels actually recorded.

Author: K. Bosman
June 7, 2024
"""
import argparse
from datetime import datetime, timedelta
from glob import glob
import json
import numpy as np
import obspy
import os
import pandas as pd
import traceback

from obspy.io.stationxml.core import validate_stationxml
from obspy.core.inventory import Inventory, Network, Station, Channel

import nfsi_obs as nf
from utilities import logger


def filter_xml(sxml_file, channel_list=None, channel_map=None):
    """
    Filter StationXML file to only channels included in list of channels.

    :param sxml_file: Path to StationXML file
    :param channel_list: List of channel IDs
    :param channel_map: pandas.DataFrame mapping existing channel IDs to corrected IDs
    :return: obspy.Inventory
    """
    is_sxml = validate_stationxml(sxml_file)
    if not is_sxml:
        raise TypeError('Input file {} is not a valid StationXML file.'.format(sxml_file))

    input_inv = obspy.read_inventory(sxml_file)

    for net in input_inv.networks:
        out_net = Network()
        for sta in net.stations:
            out_sta = Station()
            for ch in sta.channels:
                out_ch = Channel()
                # Check if channel is in channel_map
                # Correct channel ID if necessary
                # Add channel to output inventory if in channel_list or channel_map

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pre-process OBS data and perform basic QC')
    parser.add_argument('--input_dir', dest="in_dir",
                        help="Directory where input StationXML files are stored, and/or base directory for relative "
                             "paths.")
    parser.add_argument('--output_dir', dest="out_dir", help="Directory where output files are to be stored.")
    parser.add_argument('--network', dest="network_id", default='XX',
                        help="Network identifier assigned by FDSN for this project. Default 'XX' for test data.")
    parser.add_argument('--relative_paths', dest="relative_paths", action="store_true",
                        help="Specify all file paths relative to in_dir.")
    parser.add_argument('--xml', dest="aqu_xml",
                        help="Path to input StationXML file (channel IDs not corrected by channel_map) if only editing "
                             "a single file. Will override directory of files if specified.")
    parser.add_argument('--datalog', dest="datalog",
                        help="Log file from deployment/recovery. Must include station identifiers and locations. "
                             "Preferred format is XLSX (or similar spreadsheet) following NFSI template.")
    parser.add_argument('--legacylogcols', dest="obslog_column_names_legacy", action="store_true",
                        help="Use legacy column names for OBS deployment log file.")
    parser.add_argument('--channelmap', dest="channel_map",
                        help="File mapping as-recorded channel codes to their correct values.")
    parser.add_argument('--out_channels', dest="out_channels", default=None,
                        help="List of channel IDs to include in output XML file. Either comma-separated string or file "
                             "(comma-separated or one channel per line). If not specified, channels included in "
                             "channel_map will be output. If no channel_map is specified, all channels will be output.")

    try:
        args = parser.parse_args()
        run_start = datetime.now()

        # TODO: Allow user-configurable log directory
        g_log = logger.get_general_logger(run_start, 'SDS')
        g_log.info("\n\n=====================================================================")
        g_log.info("Starting job: {0}".format(str(args)))

        # Base directories
        input_dir = None
        if args.in_dir:
            input_dir = os.path.abspath(os.path.expanduser(os.path.expandvars(args.in_dir)))

        if args.aqu_xml:
            xml_files = [os.path.abspath(os.path.expanduser(os.path.expandvars(args.aqu_xml)))]
        elif input_dir is not None:
            xml_files = glob(os.path.join(input_dir, '*.xml'))
        else:
            raise SyntaxError('No input file or directory specified!')

        if args.relative_paths:
            if input_dir is None:
                raise RuntimeError('Missing command-line argument: Cannot use relative paths if in_dir not specified.')

        out_dir = None
        if args.out_dir:
            if args.relative_paths:
                out_dir = os.path.join(input_dir, args.out_dir)
            else:
                out_dir = os.path.abspath(os.path.expanduser(os.path.expandvars(args.out_dir)))

        # Channel list (if specified separately)
        channels = None
        if args.out_channels:
            if args.relative_paths:
                ch_path = os.path.join(input_dir, args.out_channels)
            else:
                ch_path = args.out_channels

            if os.path.isfile(ch_path):
                channel_file = os.path.abspath(os.path.expanduser(os.path.expandvars(ch_path)))
                cf = open(channel_file)
                ch_list = cf.readlines()
                if len(ch_list) > 1:
                    channels = [c for c in ch_list]
                else:
                    channels = ch_list[0].split(',')
            else:
                channels = args.out_channels.split(',')

        # Metadata files
        if args.datalog:
            if args.relative_paths:
                data_log_file = os.path.normpath(os.path.join(input_dir, args.datalog))
            else:
                data_log_file = os.path.normpath(os.path.abspath(os.path.expanduser(os.path.expandvars(args.datalog))))
            g_log.info('Reading project metadata from {0}...'.format(data_log_file))
            obs_log_info = nf.io.parse_obs_log(data_log_file, names_in_file=not args.obslog_column_names_legacy)

        channel_map = None
        if args.channel_map:
            if args.relative_paths:
                ch_map = os.path.normpath(os.path.join(input_dir, args.channel_map))
            else:
                ch_map = os.path.abspath(os.path.expanduser(os.path.expandvars(args.channel_map)))
            channel_map = nf.io.read_channel_map(ch_map)

        for xf in xml_files:
            # TODO: Filter and correct each StationXML file in list `xml_files`
            good_channels = filter_xml(xf, channels, channel_map)

        g_log.info("Processing complete!")
        logger.close_logs()
    except Exception as e:
        print(traceback.print_exc())
        parser.print_help()
        exit(1)
