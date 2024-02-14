"""
Take in arbitrary length miniSEED files for a given channel, and output data as day-long miniSEED files.

Author: K. Bosman
June 6, 2023
"""
import argparse
from datetime import datetime, timedelta
from glob import glob
import obspy
import os
import traceback

import nfsi_obs as nf
from utilities import logger

# TODO: Make script callable with arguments (easier to reuse)
# TODO: Include metadata correction (trace IDs)

channels_of_interest = ['SeisE', 'SeisN', 'SeisZ', 'SeisX']
base_dir = 'L:/Data/Ischia test deployment'
#subfolders = ['AQU-4261', 'AQU-8263', 'AQU-B063', 'AQU-8063-fixed']
subfolders = ['AQU-8063-fixed']

#outdir = os.path.join(base_dir, 'Day-long mseed')
# Default SDS archive folder. TODO: Make CL option.
arcdir = 'L:/Data/SDS'
if not os.path.exists(arcdir):
    os.makedirs(arcdir)

for sf in subfolders:
    print('Checking folder {}...'.format(sf))
    for ci in channels_of_interest:
        print('Looking for channel {}...'.format(ci))
        data_files = glob(os.path.join(base_dir, sf, '**', '*'+ci+'*.mseed'), recursive=True)

        full_data = obspy.Stream()
        for df in data_files:
            print('Reading {}...'.format(df))
            try:
                temp = obspy.read(df, header_byteorder='>')
                for tr in temp:
                    full_data.append(tr)
            except Exception as e:
                #print(traceback.print_exc())
                print(str(e))
                continue

        full_data.merge()
        print(full_data)
        full_data.print_gaps()

        for tr in full_data:
            start = tr.stats.starttime.datetime
            end = tr.stats.endtime.datetime + timedelta(days=1)
            startday = start.date()
            endday = end.date()

            cut = obspy.UTCDateTime(startday)
            while cut < endday:
                temp = tr.slice(cut, cut + 24 * 60 * 60, nearest_sample=False)
                stt = temp.split()  # deal with traces with gaps
                print(stt)

                output_dir = os.path.join(arcdir, str(cut.year), tr.stats.network, tr.stats.station, tr.stats.channel)
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
                        help="Comma-separated list of subfolders to be processed (optional).")
    parser.add_argument('--channels', dest="channels",
                        help="Comma-separated list of channel names to process (optional).")
    parser.add_argument('--correct_metadata', dest="correct_metadata", action='store_true',
                        help="Flag to correct channel ID(s) in output data.")
    parser.add_argument('--network', dest="network_id", default='XX',
                        help="Network identifier assigned by FDSN for this project. Default 'XX' for test data.")
    parser.add_argument('--channelmap', dest="channel_map",
                        help="File mapping as-recorded channel codes to their correct values.")
    parser.add_argument('--metadata', dest="metadata_file", help="Path to metadata file (dataless SEED or StationXML)")

    try:
        args = parser.parse_args()
        start_time = datetime.now()
        obs_id = args.obs_id

        g_log = logger.get_general_logger(start_time, obs_id)
        g_log.info("\n\n=====================================================================")
        g_log.info("Starting job: {0}".format(str(args)))

        if args.data_dir:
            data_dir = os.path.abspath(os.path.expanduser(os.path.expandvars(args.data_dir)))
        else:
            raise SyntaxError('No input data directory specified!')

        output_dir = None
        if args.outdir is not None:
            output_dir = os.path.abspath(os.path.expanduser(os.path.expandvars(args.outdir)))

        if args.correct_metadata:
            channel_map = None
            if args.channel_map:
                channel_map = nf.io.read_channel_map(args.channel_map)

            metadata_file = None
            if args.metadata_file:
                metadata_file = os.path.abspath(os.path.expanduser(os.path.expandvars(args.metadata_file)))

            # Process data files to apply clock drift correction and update metadata
            fix_metadata(data_dir, row, args.network_id, output_dir, channel_map, metadata_file)

        # Split data into day-long miniSEED files saved in archive_dir (SDS folder structure)

        g_log.info("Processing complete!")
        logger.close_logs()
    except Exception as e:
        print(traceback.print_exc())
        parser.print_help()
        exit(1)
