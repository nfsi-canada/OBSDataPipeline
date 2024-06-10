import argparse
from datetime import datetime, timedelta
from glob import glob
import os
import pandas as pd
from pynmeagps import NMEAReader, NMEAParseError
import re
import timeit
import warnings

# TODO: Clean up and/or combine with digest_sonardyne_logs.py. Determine actual usefulness.


class SonardyneParseError(Exception):
    """
    Sonardyne parsing error.
    """


def parse_scl(message):
    parsed = {}
    idx = 1
    if message[0] == '<':
        parsed['message_source'] = 'command'
    elif message[0] == '>':
        parsed['message_source'] = 'response'
    else:
        if re.match(r'\[.*\]', message[idx:]):
            parsed['message_source'] = 'data'
            idx += 1
            message = message[:-1]
        else:
            parsed['message_source'] = 'manual command'
            idx = 0

    if message[idx] == 'U':
        parsed['transceiver_UID'] = message[idx:idx+7]
        idx += 8    # Uhhhhhh,

    comm = re.match(r'([A-Za-z]+):(\d{4})?[,;]?', message[idx:])
    if comm is not None:
        parsed['command'] = comm.groups()[0].upper()
        parsed['address'] = comm.groups()[1]
        idx += len(comm[0])
    else:
        raise SonardyneParseError('Invalid message, skipping: {}'.format(message))

    parsed['message'] = message[idx:]

    # TODO: Interpret commands other than just MR
    if parsed['command'] == 'MR':
        # parse range information and stats
        if parsed['message_source'] == 'response':
            try:
                stat_split = re.match(r'(.*)\[([A-Za-z0-9.,;\-]+)\]', message[idx:])
                ac_stats = stat_split.groups()[1].split(',')
                for ac in ac_stats:
                    name = re.match(r'[A-Z]+', ac)
                    val = ac[len(name[0]):]
                    if len(val.split(';')) > 1:
                        # USBL response, multiple elements
                        try:
                            parsed[name[0]] = [float(x) for x in val.split(';')]
                        except ValueError:
                            parsed[name[0]] = [x for x in val.split(';')]
                    else:
                        parsed[name[0]] = float(val)
                rem = stat_split.groups()[0].split(';')
                for r in rem:
                    info = re.match(r'([A-Z]+)([0-9e\-.]+)', r)
                    parsed[info.groups()[0]] = float(info.groups()[1])
            except (TypeError, AttributeError):
                warnings.warn('Invalid range measurement: {}'.format(message))

    return parsed


def parse_usbl_log(comms_log_filepath):
    """
    Parse communications log file from Ranger 2 USBL system

    :param comms_log_filepath: Full path to communications log file

    :return pandas.DataFrame of all data read from log file
    """
    invalid_lines = 0

    filename = str(os.path.basename(comms_log_filepath))
    info = filename.split('.')[0].split('_')
    starttime = datetime.strptime(info[1], '%H%M%S')
    startdate = datetime.strptime(info[0], '%Y%m%d')
    sensorcat = info[2]
    sensornumber = info[3]
    sensortype = info[4]
    if sensorcat not in ['Transceiver', 'TOD', 'GNSS']:
        raise NotImplementedError('Unrecognized sensor category: {}'.format(sensorcat))

    # Read file contents
    f = open(comms_log_filepath, 'rb')
    prev_time = starttime - timedelta(seconds=1)    # Sometimes first point is in previous second... don't know why
    days = 0
    log_data = []
    lines = f.read().split(b'\r\n')
    for i in range(len(lines)):
        try:
            line = lines[i].decode('latin-1')
            bits = line.split()
            if len(bits) < 4:
                if len(bits) < 1:
                    # Filter out blank lines from file (whitespace only), except final line (ignore)
                    if i == (len(lines) - 1):
                        continue
                    warnings.warn('No information in line {}: {}'.format(i+1, line))
                else:
                    warnings.warn('Incomplete data line {}: {}'.format(i+1, line))
                invalid_lines += 1
                continue

            clocktime = datetime.strptime(bits[1], '%H:%M:%S.%f')
            if clocktime < prev_time - timedelta(seconds=10):
                # Occasionally commands and responses can get a little out of order if a lot is happening at the same time
                days += 1
                print('Rollover to next day: {}-{}-{}, {}'.format(startdate.year, startdate.month, startdate.day + days, clocktime.strftime('%H:%M:%S.%f')))
            # TODO: Error checking for end of month
            timestamp = datetime(startdate.year, startdate.month, startdate.day + days, clocktime.hour, clocktime.minute,
                                 clocktime.second, clocktime.microsecond)

            params = {
                'sensor_category': sensorcat,
                'sensor_number': sensornumber,
                'sensor_type': sensortype,
                'clock_time': timestamp,
            }
            msg = bits[3]
            msgtype = ''
            if sensorcat == 'Transceiver':
                try:
                    info = parse_scl(msg)
                    params.update(info)
                    msgtype = 'Sonardyne'
                except SonardyneParseError:
                    invalid_lines += 1
                    pass
            elif sensorcat in ['TOD', 'GNSS']:
                info = NMEAReader.parse(msg)
                if info is None:
                    warnings.warn('Invalid NMEA message (line {}): {}'.format(i+1, msg))
                    invalid_lines += 1
                    continue
                msgtype = info.talker + info.msgID
                for key in info.__dict__:
                    if key[0] != '_':
                        params.update({key: info.__dict__[key]})
            else:
                raise Exception('Unrecognized sensor category (uninterpretable): {}'.format(sensorcat))

            params.update({'message_type': msgtype})
            log_data.append(params)
            prev_time = clocktime
        except UnicodeDecodeError:
            warnings.warn('Unable to decode line {}, skipping'.format(i+1))
            invalid_lines += 1

    if invalid_lines > 0:
        print('Skipped {} invalid lines in log file'.format(invalid_lines))

    df = pd.DataFrame(log_data)
    return df


def parse_gnss(logfile):
    """
    Parse log file of GNSS data (raw messages only)
    """
    if re.search(r'20[0-9]{2}-[0-9]{2}-[0-9]{2}', os.path.basename(logfile)):
        dt_str = re.match(r'.*(20[0-9]{2})-([0-9]{2})-([0-9]{2}).*', os.path.basename(logfile)).groups()
        date_vals = [int(x) for x in dt_str]

    log_data = []
    f = open(logfile, 'rb')
    lines = f.read().split(b'\n')
    invalid_lines = 0
    for i in range(len(lines)):
        line = lines[i].decode('latin-1')
        try:
            info = NMEAReader.parse(line)
        except NMEAParseError as e:
            warnings.warn('Invalid NMEA message: {}'.format(e))
            continue

        if info is None:
            #warnings.warn('Invalid NMEA message (line {}): {}'.format(i + 1, line))
            invalid_lines += 1
            continue
        params = {}
        msgtype = info.talker + info.msgID
        for key in info.__dict__:
            if key[0] != '_':
                params.update({key: info.__dict__[key]})
        params.update({'message_type': msgtype})
        if 'time' in params:
            tm_vals = [params['time'].hour, params['time'].minute, params['time'].second]
            dttm = datetime(*date_vals, *tm_vals)
            params.update({'datetime': dttm})
        log_data.append(params)

    if invalid_lines > 0:
        print('Skipped {} invalid or unrecognized lines in log file'.format(invalid_lines))

    df = pd.DataFrame(log_data)
    return df


def parse_boat_tracker(logfile):
    """ Parse log file from Discovery boat tracker """
    f = open(logfile)
    logdata = []
    while True:
        temp = f.readline()
        if not temp:    # empty line for end of file
            break
        if temp[0] == '#':  # comment line
            continue
        if not temp.strip():    # blank line
            continue

        parts = temp.split()
        if len(parts) < 3:
            warnings.warn('Incomplete data line: {}'.format(temp))
            continue

        dttm = datetime.strptime(temp[:19], '%d-%m-%Y %H:%M:%S')
        coords = temp[20:].strip().split(',')

        params = {
            'datetime': dttm,
            'latitude': float(coords[0]),
            'longitude': float(coords[1])
        }
        logdata.append(params)

    df = pd.DataFrame(logdata)
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Digest log files from Ranger 2 USBL system.")
    parser.add_argument('folder', type=str, help="Path to folder containing log files.")

    full_start = timeit.default_timer()
    args = parser.parse_args()

    # Get file list in folder
    directory = os.path.abspath(os.path.expanduser(os.path.expandvars(args.folder)))
    if not os.path.isdir(directory):
        raise IOError('Input folder does not exist: {}'.format(os.path.normpath(directory)))
    all_files = glob(os.path.join(directory, '*.*'))

    time_info = []

    for lf in all_files:
        istart = timeit.default_timer()

        filename = os.path.basename(lf)
        if os.path.splitext(filename)[1] not in ['.txt', '.log', '.btr']:
            warnings.warn('Unrecognized log file extension: {}'.format(os.path.splitext(filename)[1]))
            continue

        print(filename)
        if re.match(r'[0-9]{8}_[0-9]{6}_[A-Za-z]+_[0-9]+_[A-Za-z0-9]+_Log#[0-9]+[-\w]*\.txt', filename):
            # valid USBL log file name
            log_info = parse_usbl_log(lf)
        elif re.search(r'GNSS', filename):
            log_info = parse_gnss(lf)
        elif re.match(r'[0-9]{2}-[0-9]{2}-[0-9]{4}.btr', filename):
            log_info = parse_boat_tracker(lf)
        else:
            warnings.warn('Invalid log file: {}'.format(filename))
            continue

        log_info.to_csv(os.path.join(directory, 'parsed_'+os.path.splitext(filename)[0]+'.csv'))

        iend = timeit.default_timer()
        time_info.append([filename, iend - istart])

    full_end = timeit.default_timer()

    print('Runtime by file:')
    for ti in time_info:
        print('{0}: {1:.3f} seconds ({2:.3f} minutes)'.format(ti[0], ti[1], ti[1] / 60))

    print('Total runtime: {0:.3f} seconds ({1:.3f} minutes)'.format(full_end - full_start, (full_end - full_start) / 60))
