"""
Split an Aquarius data package into two separate folders, based on a specific time period to be cut out of the main data
package. The primary use case for this script is when a portion of the data needs to be reviewed for classified signals.

The StationXML corresponding to the useful data channels (Sensor1.xml) is copied to both locations.

Author: K. Bosman
July 20, 2026
"""
import argparse
from datetime import datetime, timedelta
from glob import glob
import obspy
import os
from shutil import copy2
import traceback
import warnings

from obspy.io.mseed.util import get_start_and_end_time

from utilities import logger

DEFAULT_CHANNELS = ['S1SeisEFR', 'S1SeisNFR', 'S1SeisZFR', 'S1SeisXFR']

def split_data_package(raw_dir, rem_path, split_path, channels, start, end):
    """
    Split Aquarius data package into 2 sub-folders. Save all data from `channels` during time period specified by
    `start` and `end` into `split_path`. All remaining data is saved into `rem_path`.

    Data files must be in miniSEED format (extension .mseed)
    """
    for root, dirs, files in os.walk(raw_dir):
        for f in files:
            g_log.info('Checking file: {}'.format(f))
            dest_dir = None
            source_path = os.path.join(root, f)
            relative_path = os.path.relpath(root, raw_dir)
            # Check file extension
            if f.endswith('.mseed'):
                ch_name = f.split('_')[1]
                if ch_name in channels:
                    # Check time span
                    times = get_start_and_end_time(source_path)
                    if times[0] < start and times[1] < start:
                        # Entirely before time span of interest
                        g_log.debug('File of interest, entirely before scrub period => OK')
                        dest_dir = os.path.join(rem_path, relative_path)
                    elif times[0] > end and times[1] > end:
                        # Entirely after time span of interest
                        g_log.debug('File of interest, entirely after scrub period => OK')
                        dest_dir = os.path.join(rem_path, relative_path)
                    elif times[0] > start and times[1] < end:
                        # Entirely within time span of interest
                        g_log.debug('File of interest, entirely within scrub period => Scrub')
                        dest_dir = os.path.join(split_path, relative_path)
                    else:
                        # Some overlap, need to split file
                        g_log.debug('File of interest, crosses boundary of scrub period. Splitting...')
                        # Ensure directories exist
                        os.makedirs(os.path.join(rem_path, relative_path), exist_ok=True)
                        os.makedirs(os.path.join(split_path, relative_path), exist_ok=True)
                        # Read data file
                        data = obspy.read(source_path)
                        before = data.slice(endtime=start)
                        during = data.slice(starttime=start, endtime=end)
                        after = data.slice(starttime=end)
                        # Check each to see if it has data in it
                        if len(before) > 0 or len(after) > 0:
                            outer = obspy.Stream()
                            for b in before:
                                outer.append(b)
                            for a in after:
                                outer.append(a)
                            outer = outer.merge().split()
                            outer.write(os.path.join(rem_path, relative_path, f), format="MSEED")

                        if len(during) > 0:
                            during.write(os.path.join(split_path, relative_path, f), format="MSEED")
                else:
                    # Not a channel of interest
                    g_log.debug('Not a channel of interest')
                    dest_dir = os.path.join(rem_path, relative_path)
            else:
                # Non-data file
                g_log.debug('Non-data file')
                dest_dir = os.path.join(rem_path, relative_path)
                # TODO: Copy metadata file(s) to scrub directory also to have a functionally-complete data package
                if f == 'Sensor1.xml':
                    g_log.debug('StationXML file. Copying to split directory also.')
                    dest2 = os.path.join(split_path, relative_path)
                    os.makedirs(dest2, exist_ok=True)
                    copy2(source_path, dest2)

            if dest_dir is not None:
                os.makedirs(dest_dir, exist_ok=True)
                copy2(source_path, dest_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pre-process OBS data and save in SDS archive format')
    parser.add_argument('--data_dir', dest="data_dir", help="Directory where raw miniSEED data is stored.")
    parser.add_argument('--scrub_dir', dest="scrub_dir",
                        help="Directory where data to be cut out of package will be stored.")
    parser.add_argument('--save_dir', dest="save_dir",
                        help="Directory where remainder of data package will be stored.")
    parser.add_argument('--channels', dest="channels",
                        help="Comma-separated list of channel names to process (optional).")
    parser.add_argument('--start', dest="start", default=None,
                        help="Start date/time for period to cut out, as YYYYMMDD[hh[mm[ss]]].")
    parser.add_argument('--end', dest="end", default=None,
                        help="End date/time for period to cut out (inclusive), as YYYYMMDD[hh[mm[ss]]].")
    parser.add_argument('--log_dir', dest="log_dir", help="Directory to save log files.")
    parser.add_argument('--debug', dest='debug', action='store_true',
                        help="Activate debug mode (more verbose logging).")

    try:
        args = parser.parse_args()
        run_start = datetime.now()

        if args.log_dir:
            logs_dir = os.path.abspath(os.path.expanduser(os.path.expandvars(args.log_dir)))
            g_log = logger.get_general_logger(run_start, 'Split', debug=args.debug, logs_dir=logs_dir)
        else:
            g_log = logger.get_general_logger(run_start, 'Split', debug=args.debug)

        g_log.info("\n\n=====================================================================")
        g_log.info("Starting job: {0}".format(str(args)))

        # Base directories
        if args.data_dir:
            data_dir = os.path.abspath(os.path.expanduser(os.path.expandvars(args.data_dir)))
        else:
            raise SyntaxError('No input data directory specified!')

        if args.save_dir:
            save_dir = os.path.abspath(os.path.expanduser(os.path.expandvars(args.save_dir)))
        else:
            warnings.warn('No output data directory specified! Using default path "_split" instead.')
            save_dir = os.path.join(os.path.split(data_dir)[0], os.path.split(data_dir)[1] + '_split')

        if args.scrub_dir:
            scrub_dir = os.path.abspath(os.path.expanduser(os.path.expandvars(args.scrub_dir)))
        else:
            warnings.warn('No output data directory specified! Using default path "_scrub" instead.')
            scrub_dir = os.path.join(os.path.split(data_dir)[0], os.path.split(data_dir)[1] + '_scrub')


        # Filtering by subfolder/channel
        channels = DEFAULT_CHANNELS
        if args.channels:
            if args.channels == 'all':
                channels = 'all'
            else:
                channels = args.channels.split(',')
        else:
            g_log.info('Splitting default channels, seismometer and hydrophone ONLY.')

        # Start and end dates
        startdate, enddate = None, None
        if args.start is not None:
            try:
                if len(args.start) == 8:
                    startdate = obspy.UTCDateTime(datetime.strptime(args.start, '%Y%m%d'))
                elif len(args.start) == 10:
                    startdate = obspy.UTCDateTime(datetime.strptime(args.start, '%Y%m%d%H'))
                elif len(args.start) == 12:
                    startdate = obspy.UTCDateTime(datetime.strptime(args.start, '%Y%m%d%H%M'))
                elif len(args.start) == 14:
                    startdate = obspy.UTCDateTime(datetime.strptime(args.start, '%Y%m%d%H%M%S'))
                else:
                    raise TypeError('Unknown timestamp format')
            except Exception as e:
                startdate = None
                g_log.warning('Invalid start date specified: {}'.format(args.start))
                pass
        if args.end is not None:
            try:
                if len(args.end) == 8:
                    enddate = obspy.UTCDateTime(datetime.strptime(args.end, '%Y%m%d') + timedelta(days=1))
                elif len(args.end) == 10:
                    enddate = obspy.UTCDateTime(datetime.strptime(args.end, '%Y%m%d%H') + timedelta(hours=1))
                elif len(args.end) == 12:
                    enddate = obspy.UTCDateTime(datetime.strptime(args.end, '%Y%m%d%H%M') + timedelta(minutes=1))
                elif len(args.end) == 14:
                    enddate = obspy.UTCDateTime(datetime.strptime(args.end, '%Y%m%d%H%M%S') + timedelta(seconds=1))
                else:
                    raise TypeError('Unknown timestamp format')
            except Exception as e:
                enddate = None
                g_log.warning('Invalid end date specified: {}'.format(args.end))
                pass

        timespan_str = ''
        if startdate is not None:
            timespan_str += 'Start: {}'.format(startdate.strftime('%Y-%m-%d %H:%M:%S'))
        else:
            timespan_str += 'No start date specified'
        if enddate is not None:
            timespan_str += ', End: {}'.format(enddate.strftime('%Y-%m-%d %H:%M:%S'))
        else:
            timespan_str += ', No end date specified'
        g_log.info(timespan_str)

        # Split data into two folders (nominally unclassified and to-scrub)
        split_data_package(data_dir, save_dir, scrub_dir, channels=channels, start=startdate, end=enddate)

        g_log.info("Data package split complete!")
        run_end = datetime.now()
        running = run_end - run_start
        g_log.info('Total runtime: {} seconds'.format(running.total_seconds()))
        logger.close_logs()
    except Exception as e:
        print(traceback.print_exc())
        parser.print_help()
        exit(1)
