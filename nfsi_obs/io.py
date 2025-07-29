from datetime import datetime, timedelta
import obspy
from obspy.core.inventory import Network, Station, Operator, Person
import os
import pandas as pd
from pynmeagps import NMEAReader
import warnings

from .sonardyne import SonardyneReader


def parse_obs_log(log_file, delimiter=',', network='XX', names_in_file=False):
    """

    :param log_file: spreadsheet-like file with information logged during OBS deployment/recovery
    :type log_file: str
    :param delimiter: separator string for delimited text files
    :type delimiter: str

    :return: dictionary with relevant information from the log file
    :rtype: dict
    """
    # TODO: Hoping to replace this entirely with Sensor Tracker integration
    # If there are changes to the template, these column names will need to be updated. Consider putting in config.ini?
    location_cols = ['Station', 'OBS Name', 'OBS ID', 'Launch Date/Time (UTC)', 'Date/Time on Seafloor (UTC)',
                     'Survey Start Date/Time (UTC)', 'Date/Time Released (UTC)', 'Surfaced Date/Time (UTC)',
                     'Recovery Date/Time (UTC)', 'Planned Latitude', 'Planned Longitude', 'Planned Depth (m)',
                     'Launch Latitude', 'Launch Longitude', 'Water Depth at Launch (m)',
                     'Distance Launch from Planned (km)', 'Surveyed Latitude', 'Surveyed Longitude',
                     'Surveyed Depth (m)', 'Survey Depth Error (m)', 'Survey East Error (m)', 'Survey North Error (m)',
                     'Horizontal Drift during Fall (m)', 'Bearing Surveyed from Launch', 'Surfacing Latitude',
                     'Surfacing Longitude', 'Horizontal Drift during Rise (m)', 'Bearing Surfacing from Surveyed',
                     'Recovery Latitude', 'Recovery Longitude', 'Drift on Surface (km)']
    deploy_cols = ['Station', 'Planned Latitude', 'Planned Longitude', 'Planned Depth (m)', 'Launch Latitude',
                   'Launch Longitude', 'Water Depth at Launch (m)', 'Distance Launch from Planned (km)',
                   'Launch Date/Time (UTC)', 'Date/Time on Seafloor (UTC)', 'Survey Start Date/Time (UTC)',
                   'Survey End Date/Time (UTC)', 'Programmed Release Date/Time (UTC)', 'OBS Name', 'OBS ID',
                   'Minimus Firmware', 'Femtomus Firmware', 'Acoustic Modem Address', 'Acoustic Modem UID',
                   'Apollo S/N', 'XMB S/N', 'Radio beacon frequency (MHz)', 'Battery SOC at Deployment (%)',
                   'Burn-wire Batch', 'Burn-wire Widget Test Voltage', 'Burn-wire Dunker Test Voltage', 'Deployment Comments']
    recover_cols = ['Station', 'Deployed Latitude', 'Deployed Longitude', 'Water Depth (m)', 'OBS Name', 'OBS ID',
                    'Acoustic Modem Address', 'Acoustic Modem UID', 'Date/Time Acoustic Contact Established (UTC)',
                    'Date/Time Released from Anchor (UTC)', 'Surfacing Date/Time (UTC)', 'On-Deck Date/Time (UTC)',
                    'Date/Time Recording Stopped (UTC)', 'Surfacing Latitude', 'Surfacing Longitude',
                    'Horizontal Drift during Rise (km)', 'Recovery Latitude', 'Recovery Longitude',
                    'Drift on Surface (km)', 'Clock Offset at Seabed (ms)', 'Clock Offset on Deck (ms)',
                    'Battery SOC at Recovery (%)', 'Backup hard drive IDs', 'Recovery Comments']

    log_info = {}
    dm_info = None

    filetype = os.path.splitext(log_file)[-1][1:]   # remove '.' from beginning of file extension string
    if filetype in ['xls', 'xlsx', 'xlsm', 'xlsb', 'odf', 'ods', 'odt']:
        # If file is an Excel/ODS format (standard template used)
        # index is station name (must be unique within each project)
        if names_in_file:
            # TODO: General way to specify parse_dates? Column numbers may change over time.
            # TODO: Set date/time values prior to 2021 to NaT
            locations = pd.read_excel(log_file, sheet_name='Locations', header=0, skiprows=2,
                                      parse_dates=[3, 4, 5, 6, 7, 8])
            deployment = pd.read_excel(log_file, sheet_name='Deployment Log', header=0, skiprows=2,
                                       parse_dates=[8, 9, 10, 11, 12, 13])
            recovery = pd.read_excel(log_file, sheet_name='Recovery Log', header=0, skiprows=2,
                                     parse_dates=[8, 9, 10, 11])
        else:
            locations = pd.read_excel(log_file, sheet_name='Locations', header=None, names=location_cols, skiprows=2,
                                      parse_dates=[3, 4, 5, 6, 7, 8])
            deployment = pd.read_excel(log_file, sheet_name='Deployment Log', header=None, names=deploy_cols, skiprows=2,
                                       parse_dates=[8, 9, 10, 11, 12, 13])
            recovery = pd.read_excel(log_file, sheet_name='Recovery Log', header=None, names=recover_cols, skiprows=2,
                                     parse_dates=[8, 9, 10, 11])
        for df in [locations, deployment, recovery]:
            df.dropna(subset=['Station', 'OBS Name'], inplace=True)     # Remove blank lines and stations not launched if present
            df.set_index('Station', drop=False, inplace=True)

        # TODO: Return subset of columns that actually exist
        if 'Survey Calculation Method' in locations.columns:
            loc_info = locations[['Station', 'OBS Name', 'OBS ID', 'Launch Date/Time (UTC)',
                                  'Date/Time on Seafloor (UTC)', 'Date/Time Released (UTC)', 'Recovery Date/Time (UTC)',
                                  'Survey Calculation Method']].copy()
        else:
            loc_info = locations[['Station', 'OBS Name', 'OBS ID', 'Launch Date/Time (UTC)',
                                  'Date/Time on Seafloor (UTC)', 'Date/Time Released (UTC)',
                                  'Recovery Date/Time (UTC)']].copy()
        dep_info = deployment[['Recording Start Date/Time (UTC)', 'Battery SOC at Deployment (%)',
                               'Deployment Comments']].copy()
        rec_info = recovery[['Deployed Latitude', 'Deployed Longitude', 'Water Depth (m)',
                             'Date/Time Recording Stopped (UTC)', 'Clock Offset on Deck (ms)',
                             'Battery SOC at Recovery (%)', 'Recovery Comments']].copy()
        dm_info = pd.concat([loc_info, dep_info, rec_info], axis=1)

        log_info['locations'] = locations
        log_info['deployment'] = deployment
        log_info['recovery'] = recovery
    elif filetype in ['csv', 'txt']:
        # If file is a delimited text file
        # TODO: Require a specific structure here? How to guarantee column names are appropriate?
        dm_info = pd.read_csv(log_file, sep=delimiter, parse_dates=[4, 5], skipinitialspace=True)

    log_info['basic'] = dm_info

    """
    # Create obspy.Inventory object and add to return dictionary -> needs debugging!
    nfsi = Operator(
        'National Facility for Seismological Investigations',
        contacts=[
            Person(['NFSI'], ['National Facility for Seismological Investigations'], ['nfsi@nfsi.ca']),
            Person(['Mladen Nedimovic'], ['National Facility for Seismological Investigations'], ['mladen@nfsi.ca']),
        ],
        website='https://www.nfsi.ca'
    )
    stations = []
    for i in dm_info.index:
        if not pd.isnull(dm_info.loc[i, 'Date/Time on Seafloor (UTC)']):
            data_start = obspy.UTCDateTime(dm_info.loc[i, 'Date/Time on Seafloor (UTC)'])
        elif not pd.isnull(dm_info.loc[i, 'Launch Date/Time (UTC)']):
            data_start = obspy.UTCDateTime(dm_info.loc[i, 'Launch Date/Time (UTC)'])
        else:
            data_start = obspy.UTCDateTime(1970, 1, 1)
        if not pd.isnull(dm_info.loc[i, 'Date/Time Released (UTC)']):
            data_end = obspy.UTCDateTime(dm_info.loc[i, 'Date/Time Released (UTC)'])
        elif not pd.isnull(dm_info.loc[i, 'Recovery Date/Time (UTC)']):
            data_end = obspy.UTCDateTime(dm_info.loc[i, 'Recovery Date/Time (UTC)'])
        else:
            data_end = obspy.UTCDateTime(2599, 12, 31)
        stations.append(Station(
            dm_info.loc[i, 'Station'],
            dm_info.loc[i, 'Deployed Latitude'],
            dm_info.loc[i, 'Deployed Longitude'],
            -dm_info.loc[i, 'Water Depth (m)'],
            start_date=data_start,
            end_date=data_end,
            alternate_code=dm_info.loc[i, 'OBS ID'],
            water_level=0,
            operators=[nfsi]
        ))
    log_info['inventory'] = obspy.Inventory(
        networks=[Network(
            network,
            stations=stations,
            start_date=obspy.UTCDateTime(min(dm_info['Launch Date/Time (UTC)'])),
            end_date=obspy.UTCDateTime(max(dm_info['Recovery Date/Time (UTC)'])),
            operators=[nfsi]
        )],
        source='National Facility for Seismological Investigations',
        sender='National Facility for Seismological Investigations'
    )
    """

    return log_info


def read_channel_map(ch_map_file, delimiter=','):
    """Read a spreadsheet or delimited text file mapping recorded channels to corrected channel codes"""
    ch_map_path = os.path.abspath(os.path.expanduser(os.path.expandvars(ch_map_file)))
    filetype = os.path.splitext(ch_map_path)[-1][1:]   # remove '.' from beginning of file extension string
    ch_map = None
    if filetype in ['xls', 'xlsx', 'xlsm', 'xlsb', 'odf', 'ods', 'odt']:
        # If file is an Excel/ODS format (standard template used)
        ch_map = pd.read_excel(ch_map_path)
    elif filetype in ['csv', 'txt']:
        # If file is a delimited text file
        ch_map = pd.read_csv(ch_map_path, sep=delimiter)

    if ch_map is not None:
        ch_map.set_index('Recorded channel ID', drop=False, inplace=True)
    return ch_map


def parse_usbl_log(comms_log_filepath):
    """
    Parse communications log file from Ranger 2 USBL system

    :param comms_log_filepath: Full path to communications log file

    :return pandas.DataFrame of all data read from log file
    """
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
    f = open(comms_log_filepath)
    prev_time = starttime
    days = 0
    log_data = []
    while True:
        temp = f.readline()
        if not temp:
            # Empty line returned for end of file
            break

        bits = temp.split()
        if len(bits) < 1:
            # Filter out blank lines from file (whitespace only)
            warnings.warn('No information in line: {}'.format(temp))

        clocktime = datetime.strptime(bits[1], '%H:%M:%S.%f')
        if clocktime < prev_time:
            days += 1
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
            info = SonardyneReader.parse(msg)
            params.update(info)
        elif sensorcat in ['TOD', 'GNSS']:
            info = NMEAReader.parse(msg)
            msgtype = info.talker + info.msgID
            for key in info.__dict__:
                if key[0] != '_':
                    params.update({key: info.__dict__[key]})
        else:
            raise Exception('Unrecognized sensor category (uninterpretable): {}'.format(sensorcat))

        params.update({'message_type': msgtype})
        log_data.append(params)
        prev_time = clocktime

    df = pd.DataFrame(log_data)
    return df


def parse_btr(btr_file_path, out=False):
    parsed = []
    data = open(btr_file_path)
    for line in data.readlines():
        if line[0] == '#':
            continue    # comment line
        elif not line.strip():
            continue    # blank line
        elif line[0] == 'D':
            continue    # header line
        else:
            # data line
            parts = line.split()
            dttm = datetime.strptime(' '.join(parts[:2]), '%d-%m-%Y %H:%M:%S')
            coords = parts[2].split(',')
            lat = float(coords[0])
            lon = float(coords[1])
            parsed.append({
                'datetime': dttm,
                'latitude': lat,
                'longitude': lon
            })

    track = pd.DataFrame(parsed)
    track.set_index('datetime', drop=False, inplace=True)
    track.sort_index(inplace=True)

    if out:
        out_most, extension = os.path.splitext(btr_file_path)
        outfile = out_most + '_reparse' + extension

        track.to_csv(outfile, index=False)

    return track
