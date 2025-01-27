from ioos_qc import utils as iq_utils
import numpy as np
import obspy
import pandas as pd

from . import io, metadata, mseed, plotting, sonardyne, waveform


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


def rolling_window_stats(trace, window_length=3*24*60*60, window_offset=24*60*60, full=False):
    """

    :param trace: obspy.core.trace.Trace object
    :param window_length: length of window for averaging in seconds
    :param window_offset: offset between adjacent windows in seconds
    :param full: if True, calculate all possible statistics
    :return:
    """
    from sklearn.linear_model import LinearRegression

    trace_start = trace.meta.starttime
    trace_end = trace.meta.endtime
    first_window = obspy.UTCDateTime(trace_start.year, trace_start.month, trace_start.day)
    last_window = obspy.UTCDateTime(trace_end.year, trace_end.month, trace_end.day + 1) - window_offset

    window_stats = []
    window_start = first_window
    while window_start < last_window:
        center = window_start + window_length / 2
        end = window_start + window_length
        window = trace.slice(window_start, end)

        # Check for empty trace
        sections = window.split()
        sections.merge()
        if len(sections) < 1:
            # No valid data in window
            window_start += window_offset
            continue

        if sections[0].stats.npts > 0:
            stats = [window_start.datetime, end.datetime, center.datetime, window.data.min(), window.max(), window.data.mean()]
            if full:
                days = (center - trace_start) / 60 / 60 / 24
                secs = np.array(window.times(type='relative'))
                if isinstance(window.data, np.ma.MaskedArray):
                    mask = np.ma.getmaskarray(window.data)
                    secs_valid = secs[mask == False].reshape(-1, 1)
                    valid_data = window.data[mask == False]
                    reg = LinearRegression().fit(secs_valid, valid_data)
                    r2 = reg.score(secs_valid, valid_data)  # R^2 coefficient of linear fit (should be very close to 1)
                else:
                    reg = LinearRegression().fit(secs.reshape(-1, 1), window.data)
                    r2 = reg.score(secs.reshape(-1, 1), window.data)
                gradient = reg.coef_[0] * 1000 * 60 * 60 * 24  # convert V/s to mV/day for voltage gradient
                stats.extend([gradient, r2, days])
            window_stats.append(stats)
        window_start += window_offset

    return window_stats


def clip_data(unclipped, high_clip, low_clip):
    """
    Remove values from input data that are above `high_clip` and below `low_clip`. Returned series (np.array) has np.nan
    in place of clipped values.

    :param unclipped: input data, array-like
    :param high_clip: upper clip threshold, np.float
    :param low_clip: lower clip threshold, np.float
    :return: np.array
    """
    np_unclipped = np.array(unclipped)
    cond_clip = (np_unclipped > high_clip) | (np_unclipped < low_clip)
    np_clipped = np.where(cond_clip, np.nan, np_unclipped)
    return np_clipped


def ewma_fb(column, span):
    """
    Apply forwards, backwards exponential weighted moving average (EWMA) to data column.

    :param column: pandas.Series
    :param span: int
    :return:
    """
    # Forwards EWMA
    fwd = pd.Series.ewm(column, span=span).mean()
    # Backwards EWMA
    bwd = pd.Series.ewm(column[::-1], span=span).mean()
    # Mean of forwards and backwards EWMA
    stacked_ewma = np.vstack((fwd, bwd[::-1]))
    fb_ewma = np.mean(stacked_ewma, axis=0)
    return fb_ewma


def remove_outliers(input, fbewma, delta):
    """
    Remove data points from `input` that differ from `fbewma` by greater than +/- `delta`

    :param input: array-like
    :param fbewma: array-like
    :param delta: np.float
    :return:
    """

def remove_write_spikes(trace, range_clips=None, delta=None, span=None):
    """
    Remove spikes due to Aquarius data writes (normally every 45 minutes while deployed). This function is intended for
    use only with the external pressure and temperature data. The signals observed on other channels for the data writes
    have a slightly different character and have not been tested with this function.

    Modified from SO example (stackoverflow.com/questions/37556487/remove-spikes-from-signal-in-python)

    :param trace: obspy.core.trace.Trace object
    :return:
    """
    trace_df = pd.DataFrame(index=pd.to_datetime(trace.times('timestamp')))
    trace_df['datetime'] = pd.to_datetime(trace.times('timestamp'))
    trace_df['as_recorded'] = trace.data

    # Clip data, if desired
    if range_clips is not None:
        trace_df['clipped'] = clip_data(trace.data, *range_clips)
    else:
        trace_df['clipped'] = trace.data

    # Calculate FBEWMA

    # Remove outliers

    # Interpolate

    # Construct output obspy.Trace and return
