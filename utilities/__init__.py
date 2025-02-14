"""
Miscellaneous helper functions
"""
from . import logger, config_handler
from .report_generator import ReportGenerator


def check_nan(value):
    """ NaN checker that works for strings """
    return value != value


def time_period_string(deltaT, factor=1):
    """
    Generate string representation of the length of a time period (given in seconds). The breakpoints between units of
    time (days, hours, minutes, seconds) are controlled by the constant `factor`. For example, a factor of 2 will cause
    periods of less than 2 minutes to be represented as a number of seconds, and periods between 2 minutes and 2 hours
    to be represented as a number of minutes.
    """
    if deltaT > factor * 24 * 60 * 60:
        return '{:.1f}-day'.format(deltaT / 60 / 60 / 24)
    elif deltaT > factor * 60 * 60:
        return '{:.1f}-hour'.format(deltaT / 60 / 60)
    elif deltaT > factor * 60:
        return '{:.1f}-minute'.format(deltaT / 60)
    else:
        return '{:.1f}-second'.format(deltaT)
