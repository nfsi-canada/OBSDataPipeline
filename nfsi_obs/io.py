import os
import pandas as pd


def parse_obs_log(log_file, delimiter=','):
    """

    :param log_file: spreadsheet-like file with information logged during OBS deployment/recovery
    :type log_file: str
    :param delimiter: separator string for delimited text files
    :type delimiter: str

    :return: dictionary with relevant information from the log file
    :rtype: dict
    """
    # If there are changes to the template, these column names will need to be updated. Consider putting in config.ini?
    location_cols = ['Station', 'OBS Name', 'OBS ID', 'Launch Date/Time (UTC)', 'Date/Time on Seafloor (UTC)',
                     'Survey Start Date/Time (UTC)', 'Date/Time Released (UTC)', 'Surfaced Date/Time (UTC)',
                     'Recovery Date/Time (UTC)', 'Planned Latitude', 'Planned Longitude', 'Planned Depth (m)',
                     'Launch Latitude', 'Launch Longitude', 'Water Depth at Launch (m)',
                     'Distance Launch from Planned (km)', 'Surveyed Latitude', 'Surveyed Longitude',
                     'Survey Horizontal Error (m)', 'Horizontal Drift during Fall (m)', 'Bearing Surveyed from Launch',
                     'Surfacing Latitude', 'Surfacing Longitude', 'Horizontal Drift during Rise (m)',
                     'Bearing Surfacing from Surveyed', 'Recovery Latitude', 'Recovery Longitude', 'Drift on Surface (m)']
    deploy_cols = ['Station', 'Planned Latitude', 'Planned Longitude', 'Planned Depth (m)', 'Launch Latitude',
                   'Launch Longitude', 'Water Depth at Launch (m)', 'Distance Launch from Planned (km)',
                   'Launch Date/Time (UTC)', 'Date/Time on Seafloor (UTC)', 'Survey Start Date/Time (UTC)',
                   'Survey End Date/Time (UTC)', 'Programmed Release Date/Time (UTC)', 'OBS Name', 'OBS ID',
                   'Minimus Firmware', 'Femtomus Firmware', 'Acoustic Modem Address', 'Acoustic Modem UID',
                   'Apollo S/N', 'XMB S/N', 'Radio beacon frequency (MHz)', 'Comments']
    recover_cols = ['Station', 'Surveyed Latitude', 'Surveyed Longitude', 'Water Depth (m)', 'OBS Name', 'OBS ID',
                    'Acoustic Modem Address', 'Acoustic Modem UID', 'Date/Time Acoustic Contact Established (UTC)',
                    'Date/Time Released from Anchor (UTC)', 'Surfacing Date/Time (UTC)', 'On Deck Date/Time (UTC)',
                    'Date/Time Recording Stopped (UTC)', 'Surfacing Latitude', 'Surfacing Longitude',
                    'Horizontal Drift during Rise (m)', 'Recovery Latitude', 'Recovery Longitude',
                    'Drift on Surface (m)', 'Clock Offset at Seabed (ms)', 'Clock Offset on Deck (ms)',
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

        dm_info = locations[['Station', 'OBS Name', 'OBS ID', 'Launch Date/Time (UTC)', 'Date/Time on Seafloor (UTC)', 'Date/Time Released (UTC)', 'Recovery Date/Time (UTC)']].copy()
        dm_info.join(recovery[['Surveyed Latitude', 'Surveyed Longitude', 'Water Depth (m)', 'Clock Offset on Deck (ms)']].copy())

        log_info['locations'] = locations
        log_info['deployment'] = deployment
        log_info['recovery'] = recovery
    elif filetype in ['csv', 'txt']:
        # If file is a delimited text file
        # TODO: Require a specific structure here? How to guarantee column names are appropriate?
        dm_info = pd.read_csv(log_file, sep=delimiter, parse_dates=[4, 5], skipinitialspace=True)

    log_info['basic'] = dm_info
    return log_info


def read_channel_map(ch_map_file, delimiter=','):
    """Read a spreadsheet or delimted text file mapping recorded channels to corrected channel codes"""
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
