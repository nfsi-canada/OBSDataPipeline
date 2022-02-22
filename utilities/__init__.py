"""
Miscellaneous helper functions
"""
import os
import pandas as pd

from . import logger, config_handler


def parse_obs_log(log_file, delimiter=','):
    """

    :param log_file: spreadsheet-like file with information logged during OBS deployment/recovery
    :type log_file: str

    :return: dictionary with relevant information from the log file
    :rtype: dict
    """
    log_info = {}

    filetype = os.path.splitext(log_file)[-1][1:]   # remove '.' from beginning of file extension string
    if filetype in ['xls', 'xlsx', 'xlsm', 'xlsb', 'odf', 'ods', 'odt']:
        # If file is an Excel/ODS format
        dm_info = pd.read_excel(log_file, sheet_name='Data Management', index_col=None, parse_dates=[4, 5])
    elif filetype in ['csv', 'txt']:
        # If file is a delimited text file
        dm_info = pd.read_csv(log_file, sep=delimiter, parse_dates=[4, 5], skipinitialspace=True)

    log_info['metadata']['basic'] = dm_info
    return log_info
