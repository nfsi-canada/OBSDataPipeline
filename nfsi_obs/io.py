import obspy
from obspy.core.inventory import Network, Station, Operator, Person
import os
import pandas as pd


def parse_obs_log(log_file, delimiter=',', network='XX'):
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
                   'Apollo S/N', 'XMB S/N', 'Radio beacon frequency (MHz)', 'Battery SOC at Deployment',
                   'Burn-wire Batch', 'Burn-wire Widget Test Voltage', 'Burn-wire Dunker Test Voltage', 'Comments']
    recover_cols = ['Station', 'Deployed Latitude', 'Deployed Longitude', 'Water Depth (m)', 'OBS Name', 'OBS ID',
                    'Acoustic Modem Address', 'Acoustic Modem UID', 'Date/Time Acoustic Contact Established (UTC)',
                    'Date/Time Released from Anchor (UTC)', 'Surfacing Date/Time (UTC)', 'On-Deck Date/Time (UTC)',
                    'Date/Time Recording Stopped (UTC)', 'Surfacing Latitude', 'Surfacing Longitude',
                    'Horizontal Drift during Rise (km)', 'Recovery Latitude', 'Recovery Longitude',
                    'Drift on Surface (km)', 'Clock Offset at Seabed (ms)', 'Clock Offset on Deck (ms)',
                    'Battery SOC (%)', 'Backup hard drive IDs', 'Comments']

    log_info = {}
    dm_info = None

    filetype = os.path.splitext(log_file)[-1][1:]   # remove '.' from beginning of file extension string
    if filetype in ['xls', 'xlsx', 'xlsm', 'xlsb', 'odf', 'ods', 'odt']:
        # If file is an Excel/ODS format (standard template used)
        # index is station name (must be unique within each project)
        locations = pd.read_excel(log_file, sheet_name='Locations', header=None, names=location_cols, skiprows=2,
                                  parse_dates=[3, 4, 5, 6])
        deployment = pd.read_excel(log_file, sheet_name='Deployment Log', header=None, names=deploy_cols, skiprows=2,
                                   parse_dates=[8, 9, 10, 11, 12])
        recovery = pd.read_excel(log_file, sheet_name='Recovery Log', header=None, names=recover_cols, skiprows=2,
                                 parse_dates=[8, 9, 10, 11])
        for df in [locations, deployment, recovery]:
            df.set_index('Station', drop=False, inplace=True)

        loc_info = locations[['Station', 'OBS Name', 'OBS ID', 'Launch Date/Time (UTC)', 'Date/Time on Seafloor (UTC)', 'Date/Time Released (UTC)', 'Recovery Date/Time (UTC)']].copy()
        rec_info = recovery[['Deployed Latitude', 'Deployed Longitude', 'Water Depth (m)', 'Clock Offset on Deck (ms)']].copy()
        dm_info = pd.concat([loc_info, rec_info], axis=1)

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
