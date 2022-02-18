import logging
import os
from datetime import datetime

from . import config_handler

config = config_handler.get_config()
logs_dir = config.get('common', 'log_dir')

format_str = '%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s'
formatter = logging.Formatter(format_str)

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)

general_log_handle = None
general_log = None
verbose_setting = None


def get_general_logger(start=datetime.now(), obs_id="AQU-0000"):
    global general_log_handle
    global general_log

    if general_log:
        return general_log

    log = logging.getLogger('general_log')
    log.setLevel(logging.INFO)
    general_log_name = start.strftime('%Y-%m-%d_%H-%M-%S') + '_' + obs_id + '.log'
    general_log_handle = logging.FileHandler(os.path.join(logs_dir, general_log_name), mode='a')
    general_log_handle.setLevel(logging.INFO)
    general_log_handle.setFormatter(formatter)
    log.addHandler(general_log_handle)
    log.addHandler(ch)
    general_log = log
    return log


def get_stream_logger():
    log = logging.getLogger('terminal')
    log.setLevel(logging.DEBUG)
    log.addHandler(ch)
    return log


def close_logs():
    global general_log_handle

    if general_log_handle:
        general_log_handle.close()
        general_log_handle = None

