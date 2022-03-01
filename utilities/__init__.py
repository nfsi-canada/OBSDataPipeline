"""
Miscellaneous helper functions
"""
from . import logger, config_handler


def check_nan(value):
    """ NaN checker that works for strings """
    return value != value
