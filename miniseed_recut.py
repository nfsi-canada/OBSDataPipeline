"""
Take in arbitrary length miniSEED files for a given channel, and output data as day-long miniSEED files.

Author: K. Bosman
June 6, 2023
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

# TODO: Make script callable with arguments (easier to reuse)
# TODO: Include metadata correction (trace IDs)

# Default SDS archive folder.
DEFAULT_ARCHIVE = 'L:/Data/SDS'
DEFAULT_CHANNELS = ['S1SeisEFR', 'S1SeisNFR', 'S1SeisZFR', 'S1SeisXFR']

def make_daily_miniseed_files(data_dir, archive_dir, subfolders=None, channels=None, correct_meta=False, metadata_args=None):
    if channels is None:
        channels = DEFAULT_CHANNELS

    # Get full list of files to process
    data_files = []
    if subfolders is None:
        if channels == 'all':
            data_files = glob(os.path.join(data_dir, '**', '*.mseed'), recursive=True)
        elif isinstance(channels, list):
            for ch in channels:
                temp_files = glob(os.path.join(data_dir, '**', '*'+ch+'*.mseed'), recursive=True)
                data_files.extend(temp_files)
        else:
            raise TypeError('Unrecognized value of input `channels`: {}'.format(channels))
    elif isinstance(subfolders, list):
        for sf in subfolders:
            if channels == 'all':
                temp_files = glob(os.path.join(data_dir, sf, '**', '*.mseed'), recursive=True)
                data_files.extend(temp_files)
            elif isinstance(channels, list):
                for ch in channels:
                    temp_files = glob(os.path.join(data_dir, sf, '**', '*'+ch+'*.mseed'), recursive=True)
                    data_files.extend(temp_files)
            else:
                raise TypeError('Unrecognized value of input `channels`: {}'.format(channels))
    else:
        raise TypeError('Unrecognized value of input `subfolders`: {}'.format(subfolders))

    # Label data files by channel name
    labels = []
    for df in data_files:
        file_name = os.path.basename(df)
        ch_name = file_name.split('_')[1]
        labels.append({'channel': ch_name, 'path': df})
    labeled_files = pd.DataFrame(labels)
    g_log.info("Files contain data for {0} unique set(s) of channels".format(len(np.unique(labeled_files['channel'].values))))

    # Read metadata files (if necessary)
    station_info, ch_map, proj_meta = None, None, None
    net_id = 'XX'
    if correct_meta:
        if not isinstance(metadata_args, dict):
            raise TypeError('Unrecognized type for metadata arguments (should be dict): {}'.format(type(metadata_args)))

        if 'metadata_file' in metadata_args:
            g_log.info("Reading metadata from file {0}".format(metadata_args['metadata_file']))
            filetype = os.path.splitext(metadata_args['metadata_file'])[-1]
            if filetype in ['.dataless', '.metadata']:
                station_info = nf.metadata.read_dataless(metadata_args['metadata_file'])
            elif filetype == '.xml':
                # read as StationXML format
                station_info = obspy.read_inventory(metadata_args['metadata_file'])
            else:
                g_log.error("Unrecognized file format. Unable to read metadata.")

        if 'channel_map' in metadata_args:
            g_log.info("Reading channel ID mapping from file {0}".format(metadata_args['channel_map']))
            ch_map = nf.io.read_channel_map(metadata_args['channel_map'])

        if 'extra_meta' in metadata_args:
            if os.path.isfile(metadata_args['extra_meta']):
                g_log.info("Reading project metadata from {0}...".format(os.path.normpath(metadata_args['extra_meta'])))
                pj = open(metadata_args['extra_meta'])
                proj_meta = json.load(pj)

        if 'network_id' in metadata_args:
            net_id = metadata_args['network_id']

    # Process data files by channel
    for label, files in labeled_files.groupby('channel'):
        g_log.info('Processing channel {}...'.format(label))

        full_data = obspy.Stream()
        for df in files:
            g_log.info('Reading {}...'.format(df))
            try:
                temp = obspy.read(df, header_byteorder='>')
                for tr in temp:
                    full_data.append(tr)
            except Exception as e:
                #print(traceback.print_exc())
                print(str(e))
                continue

        full_data.merge()
        g_log.info(full_data)
        full_data.print_gaps()

        # Correct metadata (if applicable)
        if correct_meta:
            full_data = nf.metadata.update_metadata(full_data, net_id, g_log, station_info, ch_map, proj_meta)

        # Cut and save day-long miniSEED files in SDS archive structure
        for tr in full_data:
            start = tr.stats.starttime.datetime
            end = tr.stats.endtime.datetime + timedelta(days=1)
            startday = start.date()
            endday = end.date()

            cut = obspy.UTCDateTime(startday)
            while cut < endday:
                temp = tr.slice(cut, cut + 24 * 60 * 60, nearest_sample=False)
                stt = temp.split()  # deal with traces with gaps
                g_log.info(stt)

                output_dir = os.path.join(archive_dir, str(cut.year), tr.stats.network, tr.stats.station, tr.stats.channel)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                outfile = os.path.join(output_dir, '{}.{}.{}.mseed'.format(tr.id, cut.year, cut.julday))
                stt.write(outfile, format="MSEED")

                cut += 24 * 60 * 60


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pre-process OBS data and perform basic QC')
    parser.add_argument('--data_dir', dest="data_dir", help="Directory where raw miniSEED data is stored.")
    parser.add_argument('--archive_dir', dest="arc_dir",
                        help="Directory where corrected day-long miniSEED data files are to be stored.")
    parser.add_argument('--subfolders', dest="subfolders",
                        help="Comma-separated list of subfolders to be processed (optional). These must be immediate "
                             "children of data_dir.")
    parser.add_argument('--channels', dest="channels",
                        help="Comma-separated list of channel names to process (optional).")
    parser.add_argument('--correct_metadata', dest="correct_metadata", action='store_true',
                        help="Flag to correct channel ID(s) in output data.")
    parser.add_argument('--network', dest="network_id", default='XX',
                        help="Network identifier assigned by FDSN for this project. Default 'XX' for test data.")
    parser.add_argument('--relative_paths', dest="relative_paths", action="store_true",
                        help="Specify all file paths relative to data_dir (excluding archive_dir).")
    parser.add_argument('--metadata', dest="metadata_file",
                        help="Path to metadata file (dataless SEED or StationXML). Channel IDs should match the raw "
                             "data (not corrected by channel_map).")
    parser.add_argument('--channelmap', dest="channel_map",
                        help="File mapping as-recorded channel codes to their correct values.")
    parser.add_argument('--extra_meta', dest="extra_meta",
                        help="Optional JSON file with extra description and QC information. Station/channel codes "
                             "should match the corrected trace IDs in channel_map, if applicable.")

    try:
        args = parser.parse_args()
        start_time = datetime.now()
        obs_id = args.obs_id

        g_log = logger.get_general_logger(start_time, obs_id)
        g_log.info("\n\n=====================================================================")
        g_log.info("Starting job: {0}".format(str(args)))

        # Base directories
        if args.data_dir:
            data_dir = os.path.abspath(os.path.expanduser(os.path.expandvars(args.data_dir)))
        else:
            raise SyntaxError('No input data directory specified!')

        arc_dir = DEFAULT_ARCHIVE
        if args.arc_dir:
            arc_dir = os.path.abspath(os.path.expanduser(os.path.expandvars(args.arc_dir)))

        # Filtering by subfolder/channel
        subfolders, channels = None, None
        if args.subfolders:
            subfolders = args.subfolders.split(',')
        if args.channels:
            channels = args.channels.split(',')

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
        make_daily_miniseed_files(data_dir, arc_dir, subfolders=subfolders, channels=channels,
                                  correct_meta=args.correct_metadata, metadata_args=meta_args)

        g_log.info("Processing complete!")
        logger.close_logs()
    except Exception as e:
        print(traceback.print_exc())
        parser.print_help()
        exit(1)
