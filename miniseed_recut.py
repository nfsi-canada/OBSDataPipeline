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

# Default SDS archive folder.
DEFAULT_ARCHIVE = 'L:/Data/SDS'
DEFAULT_CHANNELS = ['S1SeisEFR', 'S1SeisNFR', 'S1SeisZFR', 'S1SeisXFR']
BUFFER_MAX = 10


def read_and_recut(file_list, archive_dir=DEFAULT_ARCHIVE, start=None, end=None, correct_meta=False, net_id='XX',
                   station_info=None, ch_map=None, proj_meta=None, clock_drift=None):
    full_data = obspy.Stream()
    for df in file_list:
        g_log.info('Reading {}...'.format(df))
        try:
            temp = obspy.read(df, header_byteorder='>')
            temp.trim(start, end)
            for tr in temp:
                full_data.append(tr)
        except Exception as e:
            # print(traceback.print_exc())
            print(str(e))
            continue

    full_data.merge()
    g_log.info(full_data)
    full_data.print_gaps()

    # Correct metadata (if applicable)
    if correct_meta:
        g_log.info('Updating metadata...')
        full_data = nf.metadata.update_metadata(full_data, net_id, g_log, station_info, ch_map, proj_meta)

    # Cut and save day-long miniSEED files in SDS archive structure
    for tr in full_data:
        start_time = tr.stats.starttime.datetime
        end_time = tr.stats.endtime.datetime + timedelta(days=1)
        start_day = start_time.date()
        end_day = end_time.date()

        total_clock_drift, recording_start, recording_end = 0, obspy.UTCDateTime(start_time), obspy.UTCDateTime(end_time)
        if clock_drift is not None:
            try:
                clock_correction = clock_drift.loc[tr.stats.station]
                # TODO: Handle multiple entries with same station ID in a single deployment summary (should only occur in land test data)
                recording_start = obspy.UTCDateTime(pd.to_datetime(clock_correction['Recording Start Date/Time (UTC)']))
                recording_end = obspy.UTCDateTime(pd.to_datetime(clock_correction['Date/Time Recording Stopped (UTC)']))
                total_clock_drift = clock_correction['Clock Offset on Deck (ms)']
            except Exception as e:
                print(e)
                g_log.error('Unable to read clock drift information from deployment summary. No clock correction will '
                            'be applied.')
                total_clock_drift, recording_start, recording_end = 0, obspy.UTCDateTime(start_time), obspy.UTCDateTime(end_time)

        cut = obspy.UTCDateTime(start_day)
        while cut < end_day:
            # Timestamp which would become start of day after clock drift correction (negative = OBS clock behind GPS)
            shift = total_clock_drift * (cut - recording_start) / (recording_end - recording_start)
            cut_shifted = cut + (shift / 1000)
            g_log.debug('Clock shift: {0} milliseconds | Day "start": {1}'.format(
                shift, cut_shifted.strftime('%Y-%m-%d %H:%M:%S.%f')))

            temp = tr.slice(cut_shifted, cut_shifted + 24 * 60 * 60, nearest_sample=False)
            stt = temp.split()  # deal with traces with gaps
            g_log.info(stt)

            if len(stt) > 0:
                output_dir = os.path.join(archive_dir, str(cut.year), tr.stats.network, tr.stats.station,
                                          tr.stats.channel)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                # Correct time stamps for clock drift, if necessary (negative drift == OBS clock behind GPS)
                if abs(total_clock_drift) > 0:
                    for st in stt:
                        time_shift = total_clock_drift * (st.stats.starttime - recording_start) / (recording_end - recording_start)
                        st.stats.starttime -= (time_shift / 1000)
                    g_log.info('Time series shifted for clock drift correction.')
                    g_log.info(stt)

                outfile = os.path.join(output_dir, '{}.{}.{}.mseed'.format(tr.id, cut.year, cut.julday))
                if os.path.isfile(outfile):
                    g_log.info('Found existing SDS format data for date {0:04d}/{1:02d}/{2:02d}. Combining...'.format(
                        cut.year, cut.month, cut.day))
                    existing = obspy.read(outfile)
                    for et in existing:
                        stt.append(et)
                    stt.merge()
                    stt = stt.split()
                    g_log.info(stt)

                stt.write(outfile, format="MSEED")

            cut += 24 * 60 * 60


def make_daily_miniseed_files(data_dir, archive_dir, subfolders=None, channels=None, start=None, end=None,
                              correct_meta=False, metadata_args=None):
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

    g_log.info('Found {} miniSEED data files to process.'.format(len(data_files)))

    # Label data files by channel name
    labels = []
    for df in data_files:
        file_name = os.path.basename(df)
        ch_name = file_name.split('_')[1]
        labels.append({'channel': ch_name, 'path': df})
    labeled_files = pd.DataFrame(labels)
    g_log.info("Files contain data for {0} unique set(s) of channels".format(len(np.unique(labeled_files['channel'].values))))

    # Read metadata files (if necessary)
    station_info, ch_map, proj_meta, clock_info = None, None, None, None
    net_id = 'XX'
    if correct_meta:
        if not isinstance(metadata_args, dict):
            raise TypeError('Unrecognized type for metadata arguments (should be dict): {}'.format(type(metadata_args)))

        if 'metadata_file' in metadata_args:
            g_log.info("Reading metadata from file {0}".format(metadata_args['metadata_file']))
            filetype = os.path.splitext(metadata_args['metadata_file'])[-1]
            if filetype in ['.dataless', '.metadata']:
                # read as dataless SEED format
                station_info = nf.metadata.read_dataless(metadata_args['metadata_file'])
            elif filetype == '.xml':
                # read as StationXML format
                station_info = obspy.read_inventory(metadata_args['metadata_file'])
            else:
                raise TypeError("Unrecognized file type. Unable to read metadata.")

        if 'channel_map' in metadata_args:
            g_log.info("Reading channel ID mapping from file {0}".format(metadata_args['channel_map']))
            ch_map = nf.io.read_channel_map(metadata_args['channel_map'])

        if 'extra_meta' in metadata_args:
            if os.path.isfile(metadata_args['extra_meta']):
                g_log.info("Reading project metadata from {0}...".format(os.path.normpath(metadata_args['extra_meta'])))
                pj = open(metadata_args['extra_meta'])
                proj_meta = json.load(pj)

        if 'network' in metadata_args:
            net_id = metadata_args['network']

        if 'obs_log' in metadata_args:
            clock_info = metadata_args['obs_log'][['Recording Start Date/Time (UTC)', 'Date/Time Recording Stopped (UTC)', 'Clock Offset on Deck (ms)']]

    # Process data files by channel
    for label, files in labeled_files.groupby('channel'):
        g_log.info('Processing channel {}...'.format(label))

        all_files = sorted(files['path'].values)
        if len(all_files) > BUFFER_MAX:
            done_read = False
            idf = 0
            while not done_read:
                if idf+BUFFER_MAX > len(all_files):
                    buffer_files = all_files[idf:]
                    idf = len(all_files) + 1
                else:
                    buffer_files = all_files[idf:idf+BUFFER_MAX]
                    idf += BUFFER_MAX

                read_and_recut(buffer_files, archive_dir, start, end, correct_meta, net_id, station_info, ch_map, proj_meta, clock_info)

                if idf > len(all_files):
                    done_read = True
        else:
            read_and_recut(all_files, archive_dir, start, end, correct_meta, net_id, station_info, ch_map, proj_meta, clock_info)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pre-process OBS data and save in SDS archive format')
    parser.add_argument('--data_dir', dest="data_dir", help="Directory where raw miniSEED data is stored.")
    parser.add_argument('--archive_dir', dest="arc_dir",
                        help="Directory where corrected day-long miniSEED data files are to be stored.")
    parser.add_argument('--subfolders', dest="subfolders",
                        help="Comma-separated list of subfolders to be processed (optional). These must be immediate "
                             "children of data_dir.")
    parser.add_argument('--channels', dest="channels",
                        help="Comma-separated list of channel names to process (optional).")
    parser.add_argument('--start', dest="start", default=None,
                        help="Start date/time for output data, as YYYYMMDD[hh[mm[ss]]].")
    parser.add_argument('--end', dest="end", default=None,
                        help="End date/time for output data (inclusive), as YYYYMMDD[hh[mm[ss]]].")
    parser.add_argument('--correct_metadata', dest="correct_metadata", action='store_true',
                        help="Flag to correct channel ID(s) in output data.")
    parser.add_argument('--network', dest="network_id", default='XX',
                        help="Network identifier assigned by FDSN for this project. Default 'XX' for test data.")
    parser.add_argument('--relative_paths', dest="relative_paths", action="store_true",
                        help="Specify all file paths relative to data_dir (excluding archive_dir and log_dir).")
    parser.add_argument('--metadata', dest="metadata_file",
                        help="Path to metadata file (dataless SEED or StationXML). Channel IDs should match the raw "
                             "data (not corrected by channel_map).")
    parser.add_argument('--channelmap', dest="channel_map",
                        help="File mapping as-recorded channel codes to their correct values.")
    parser.add_argument('--extra_meta', dest="extra_meta",
                        help="Optional JSON file with extra description and QC information. Station/channel codes "
                             "should match the corrected trace IDs in channel_map, if applicable.")
    parser.add_argument('--log_dir', dest="log_dir", help="Directory to save log files.")
    parser.add_argument('--datalog', dest="datalog",
                        help="Log file from deployment/recovery. Must include station identifiers and clock drift "
                             "measurements. If not specified, assumed to be a file called 'log.xlsx' in the data "
                             "directory. Preferred format is XLSX (or similar spreadsheet) following NFSI template.")
    parser.add_argument('--logdelimiter', dest="log_delim",
                        help="If the OBS log file is delimited text (other than comma-delimited), use this to specify "
                             "the column delimiter.")
    parser.add_argument('--legacylogcols', dest="obslog_column_names_legacy", action="store_true",
                        help="Use legacy column names for OBS deployment log file.")
    parser.add_argument('--debug', dest='debug', action='store_true',
                        help="Activate debug mode (more verbose logging).")

    try:
        args = parser.parse_args()
        run_start = datetime.now()

        if args.log_dir:
            logs_dir = os.path.abspath(os.path.expanduser(os.path.expandvars(args.log_dir)))
            g_log = logger.get_general_logger(run_start, 'SDS', debug=args.debug, logs_dir=logs_dir)
        else:
            g_log = logger.get_general_logger(run_start, 'SDS', debug=args.debug)

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
            if args.channels == 'all':
                channels = 'all'
            else:
                channels = args.channels.split(',')

        # Start and end dates
        startdate, enddate = None, None
        if args.start is not None:
            try:
                if len(args.start) == 6:
                    startdate = obspy.UTCDateTime(datetime.strptime(args.start, '%Y%m%d'))
                elif len(args.start) == 8:
                    startdate = obspy.UTCDateTime(datetime.strptime(args.start, '%Y%m%d%H'))
                elif len(args.start) == 10:
                    startdate = obspy.UTCDateTime(datetime.strptime(args.start, '%Y%m%d%H%M'))
                elif len(args.start) == 12:
                    startdate = obspy.UTCDateTime(datetime.strptime(args.start, '%Y%m%d%H%M%S'))
                else:
                    raise TypeError('Unknown timestamp format')
            except Exception as e:
                startdate = None
                g_log.warning('Invalid start date specified: {}'.format(args.start))
                pass
        if args.end is not None:
            try:
                if len(args.end) == 6:
                    enddate = obspy.UTCDateTime(datetime.strptime(args.start, '%Y%m%d') + timedelta(days=1))
                elif len(args.end) == 8:
                    enddate = obspy.UTCDateTime(datetime.strptime(args.start, '%Y%m%d%H') + timedelta(hours=1))
                elif len(args.end) == 10:
                    enddate = obspy.UTCDateTime(datetime.strptime(args.start, '%Y%m%d%H%M') + timedelta(minutes=1))
                elif len(args.end) == 12:
                    enddate = obspy.UTCDateTime(datetime.strptime(args.start, '%Y%m%d%H%M%S') + timedelta(seconds=1))
                else:
                    raise TypeError('Unknown timestamp format')
            except Exception as e:
                enddate = None
                g_log.warning('Invalid end date specified: {}'.format(args.end))
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

            obs_log_data = None
            if args.datalog:
                if args.relative_paths:
                    data_log_file = os.path.normpath(os.path.join(data_dir, args.datalog))
                else:
                    data_log_file = os.path.normpath(
                        os.path.abspath(os.path.expanduser(os.path.expandvars(args.datalog))))
                g_log.info('Reading project metadata from {0}...'.format(data_log_file))
                obs_log_info = nf.io.parse_obs_log(data_log_file, names_in_file=not args.obslog_column_names_legacy)
                obs_log_data = obs_log_info['basic']

            meta_args = {
                'channel_map': channel_map,
                'metadata_file': metadata_file,
                'extra_meta': project_meta,
                'network': args.network_id,
                'obs_log': obs_log_data
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
