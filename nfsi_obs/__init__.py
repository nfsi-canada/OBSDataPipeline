from ioos_qc import utils as iq_utils
import numpy as np
import pandas as pd

from . import io, metadata, mseed, plotting, waveform


def get_true_periods(data, times=None):
    """
    Compile a list of "times" when `data` holds True values. Each item includes start and end time/index, and number of
    consecutive True samples.

    If `data` is a pandas.Series object, the index must be monotonically increasing.

    :param data: 1D array-like or pandas.Series, values must be interpretable as booleans
    :param times: 1D array-like of timestamps, same length as `data`. Ignored if `data` is a pandas.Series object
    :return: list of periods with True values
    """
    periods = []

    if isinstance(data, pd.Series):
        # use index as "time"
        values = data.values
        times = data.index.values
    else:
        values = np.squeeze(np.array(data))

    if times is not None:
        # Check that "times" are monotonically increasing
        if not iq_utils.check_timestamps(times):
            raise AssertionError('Time/index values are not correctly sorted. Must be monotonically increasing.')
        # Check that arrays are the same length
        if len(values) != len(times):
            raise AssertionError('Data and index are different lengths: {0} vs {1}'.format(len(values), len(times)))
    else:
        times = range(len(values))

    # Check that data values are 1D
    if values.size > len(values):
        raise AssertionError('Data values are not 1-dimensional')
    if times.size > len(times):
        raise AssertionError('Provided timestamps are not 1-dimensional')

    # Find sections of data array with True values
    prev = values[0]
    start = times[0]
    npt = 1
    for i in range(1, len(values)):
        if values[i] == prev:
            npt += 1
        else:
            if prev:
                periods.append([start, times[i-1], npt])
            prev = values[i]
            start = times[i]
            npt = 1

    return np.array(periods)
