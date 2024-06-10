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

import nfsi_obs as nf
from utilities import logger


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pre-process OBS data and perform basic QC')
    parser.add_argument('--input_dir', dest="in_dir", help="Directory where input StationXML files are stored.")
    parser.add_argument('--output_dir', dest="out_dir", help="Directory where output files are to be stored.")
    parser.add_argument('--network', dest="network_id", default='XX',
                        help="Network identifier assigned by FDSN for this project. Default 'XX' for test data.")
    parser.add_argument('--relative_paths', dest="relative_paths", action="store_true",
                        help="Specify all file paths relative to data_dir (excluding archive_dir).")
    parser.add_argument('--xml', dest="aqu_xml",
                        help="Path to input StationXML file (channel IDs not corrected by channel_map) if only editing "
                             "a single file.")
    parser.add_argument('--datalog', dest="datalog",
                        help="Log file from deployment/recovery. Must include station identifiers and locations. "
                             "Preferred format is XLSX (or similar spreadsheet) following NFSI template.")
    parser.add_argument('--channelmap', dest="channel_map",
                        help="File mapping as-recorded channel codes to their correct values.")
    parser.add_argument('--out_channels', dest="out_channels", default=None,
                        help="List of channel IDs to include in output XML file. Either comma-separated string or file "
                             "(comma-separated or one channel per line). If not specified, channels included in "
                             "channel_map will be output. If no channel_map is specified, all channels will be output.")

    try:
        args = parser.parse_args()
        run_start = datetime.now()

        g_log = logger.get_general_logger(run_start, 'SDS')
        g_log.info("\n\n=====================================================================")
        g_log.info("Starting job: {0}".format(str(args)))

        # Base directories
        if args.data_dir:
            data_dir = os.path.abspath(os.path.expanduser(os.path.expandvars(args.data_dir)))
            xml_files = glob(os.path.join(data_dir, '*.xml'))
        elif args.aqu_xml:
            xml_files = [os.path.abspath(os.path.expanduser(os.path.expandvars(args.aqu_xml)))]
        else:
            raise SyntaxError('No input file or directory specified!')

        out_dir = None
        if args.out_dir:
            out_dir = os.path.abspath(os.path.expanduser(os.path.expandvars(args.out_dir)))

        # Channel list (if specified separately)
        channels = None
        if args.out_channels:
            if os.path.isfile(args.out_channels):
                channel_file = os.path.abspath(os.path.expanduser(os.path.expandvars(args.out_channels)))
                cf = open(channel_file)
                ch_list = cf.readlines()
                if len(ch_list) > 1:
                    channels = [c for c in ch_list]
                else:
                    channels = ch_list[0].split(',')
            else:
                channels = args.out_channels.split(',')

        # Start and end dates
        startdate, enddate = None, None
        if args.start is not None:
            try:
                startdate = obspy.UTCDateTime(datetime.strptime(args.start, '%Y%m%d'))
            except Exception as e:
                startdate = None
                pass
        if args.end is not None:
            try:
                enddate = obspy.UTCDateTime(datetime.strptime(args.end, '%Y%m%d') + timedelta(days=1))
            except Exception as e:
                enddate = None
                pass

        # Metadata files
        meta_args = None
        if args.correct_metadata:
            channel_map = None
            if args.channel_map:
                if args.relative_paths:
                    channel_map = os.path.abspath(os.path.expanduser(os.path.expandvars(os.path.join(data_dir, args.channel_map))))
                else:
                    channel_map = os.path.abspath(os.path.expanduser(os.path.expandvars(args.channel_map)))

            metadata_file = None
            if args.metadata_file:
                if args.relative_paths:
                    metadata_file = os.path.abspath(os.path.expanduser(os.path.expandvars(os.path.join(data_dir, args.metadata_file))))
                else:
                    metadata_file = os.path.abspath(os.path.expanduser(os.path.expandvars(args.metadata_file)))

            project_meta = None
            # TODO: Replace with ST integration once we have an instance running and populated
            if args.extra_meta:
                if args.relative_paths:
                    project_meta = os.path.abspath(os.path.expanduser(os.path.expandvars(os.path.join(data_dir, args.extra_meta))))
                else:
                    project_meta = os.path.abspath(os.path.expanduser(os.path.expandvars(args.extra_meta)))

            meta_args = {
                'channel_map': channel_map,
                'metadata_file': metadata_file,
                'extra_meta': project_meta,
                'network': args.network_id
            }

        # Split data into day-long miniSEED files saved in archive_dir (SDS folder structure)
        make_daily_miniseed_files(data_dir, arc_dir, subfolders=subfolders, channels=channels, start=startdate,
                                  end=enddate, correct_meta=args.correct_metadata, metadata_args=meta_args)

        g_log.info("Processing complete!")
        logger.close_logs()
    except Exception as e:
        print(traceback.print_exc())
        parser.print_help()
        exit(1)
