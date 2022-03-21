"""
Miscellaneous helper functions
"""
from . import logger, config_handler
from .report_generator import ReportGenerator


def check_nan(value):
    """ NaN checker that works for strings """
    return value != value
