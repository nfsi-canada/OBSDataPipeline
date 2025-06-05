import warnings

from datetime import datetime
from ioos_qc import utils as iq_utils
import matplotlib.pyplot as plt
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
    end_day = trace_end + 1 * 24 * 60 * 60
    first_window = obspy.UTCDateTime(trace_start.year, trace_start.month, trace_start.day)
    last_window = obspy.UTCDateTime(end_day.year, end_day.month, end_day.day) - window_offset

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


def calc_tilt_from_mems(meta_df):
    """
    Calculate OBS tilt from vertical using 3-component MEMS accelerometer/tiltmeter reading.

    :param meta_df: DataFrame including columns 'AccZ', 'AccN' and 'AccE' for accelerometer reading
    :return: tilt angle from vertical and bearing (clockwise from OBS North)
    """
    try:
        if {'AccZ', 'AccN', 'AccE'}.issubset(meta_df.columns):
            mems_acc = [meta_df[c].values[0] for c in ['AccZ', 'AccN', 'AccE']]
            if abs(mems_acc[0]) > 0:
                tilt_deg = np.degrees(np.arctan(np.sqrt(mems_acc[1] ** 2 + mems_acc[2] ** 2) / mems_acc[0]))
                tilt_az = np.degrees(np.arctan2(mems_acc[2], mems_acc[1]))
                if tilt_az < 0:
                    tilt_az += 360
                return tilt_deg, tilt_az
        else:
            warnings.warn("MEMS reading not provided for tilt calculation. Expected columns 'AccZ', 'AccN' and 'AccE'.")
            return None
    except TypeError as e:
        # TypeError if AccZ is None (from abs(None))
        warnings.warn('Invalid MEMS reading provided: {}'.format(mems_acc))
        return None


def calc_tilt_rotation(dep_meta, rec_meta):
    """
    Calculate apparent rotation of tilt axis between 2 MEMS readings. Input DataFrames must include columns 'AccZ', 'AccN' and 'AccE'.

    :param dep_meta: initial tilt measurement, pandas.DataFrame
    :param rec_meta: final tilt measurement, pandas.DataFrame
    :return: 3-D rotation angle between tilt axis vectors in degrees
    """
    mems_1 = [dep_meta[c].values[0] for c in ['AccZ', 'AccN', 'AccE']]
    mems_2 = [rec_meta[c].values[0] for c in ['AccZ', 'AccN', 'AccE']]

    # Inverse cosine of normalized dot product
    rotation = np.degrees(np.arccos(
        (mems_1[0]*mems_2[0] + mems_1[1]*mems_2[1] + mems_1[2]*mems_2[2]) /
        (np.sqrt(mems_1[0]**2 + mems_1[1]**2 + mems_1[2]**2) * np.sqrt(mems_2[0]**2 + mems_2[1]**2 + mems_2[2]**2))
    ))
    return rotation


def clip_data(unclipped, low_clip, high_clip):
    """
    Remove values from input data that are above `high_clip` and below `low_clip`. Returned series (np.array) has np.nan
    in place of clipped values.

    :param unclipped: input data, array-like
    :param low_clip: lower clip threshold, np.float
    :param high_clip: upper clip threshold, np.float
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
    :return: np.array
    """
    # Forwards EWMA
    fwd = pd.Series.ewm(column, span=span).mean()
    # Backwards EWMA
    bwd = pd.Series.ewm(column[::-1], span=span).mean()
    # Mean of forwards and backwards EWMA
    stacked_ewma = np.vstack((fwd, bwd[::-1]))
    fb_ewma = np.mean(stacked_ewma, axis=0)
    return fb_ewma


def remove_outliers(raw_data, average, delta):
    """
    Remove data points from `raw_data` that differ from `average` by greater than +/- `delta`

    :param raw_data: array-like
    :param average: array-like
    :param delta: np.float
    :return: np.array
    """
    np_raw = np.array(raw_data)
    np_average = np.array(average)
    cond_delta = (np.abs(np_raw - np_average) > delta)
    no_outliers = np.where(cond_delta, np.nan, np_raw)
    return no_outliers

def remove_write_spikes(trace, range_clips=None, delta=1, span=None, qcplot=False, savedf=False, dfpath=None):
    """
    Remove spikes due to Aquarius data writes (normally every 45 minutes while deployed). This function is intended for
    use only with the external pressure and temperature data. The signals observed on other channels for the data writes
    have a slightly different character and have not been tested with this function.

    Modified from SO example (stackoverflow.com/questions/37556487/remove-spikes-from-signal-in-python)

    :param trace: obspy.core.trace.Trace object
    :param range_clips: 2-element tuple or list,
    :param delta: np.float, outlier threshold for difference between input data and FBEWMA
    :param span: int or list-like of ints, number of sample points to use for FBEWMA calculation
    :param qcplot: bool, if True, plot data series for inspection (pauses execution)
    :param savedf: bool, if True, save DataFrame of intermediate steps
    :param dfpath: str, path to save DataFrame (default in current directory with name {trace_id}_spike_removal_{datetime.now}.csv)
    :return: obspy.core.trace.Trace object
    """
    mode = 'single'
    if span is None:
        warnings.warn('FBEWMA window length not specified. Using default value of 10 sample points.')
        span = 10
    else:
        try:
            span = int(span)
        except TypeError:
            try:
                if len(span) > 1:
                    print('Multiple averaging windows given ({0}), will use minimum result for outlier removal.'.format(span))
                    mode = 'multi'
            except Exception:
                warnings.warn('Unable to perform spike removal! Unrecognized input provided for FBEWMA window length: {0}'.format(span))
                return trace

    trace_df = pd.DataFrame(index=pd.to_datetime(trace.times('timestamp')*1e9))
    trace_df['as_recorded'] = trace.data

    # Clip data, if desired
    if range_clips is not None:
        trace_df['clipped'] = clip_data(trace.data, *range_clips)
    else:
        trace_df['clipped'] = trace.data

    # Calculate forwards-backwards exponential weighted moving average
    if mode == 'single':
        trace_df['fbewma'] = ewma_fb(trace_df['clipped'], span)
    elif mode == 'multi':
        all_keys = []
        for s in span:
            key = 'fbewma_{}'.format(s)
            trace_df[key] = ewma_fb(trace_df['clipped'], s)
            all_keys.append(key)
        trace_df['fbewma'] = trace_df[all_keys].min(axis=1)

    # Remove outliers
    trace_df['remove_outliers'] = remove_outliers(trace_df['clipped'].tolist(), trace_df['fbewma'].tolist(), delta)

    # Interpolate
    trace_df['interpolated'] = trace_df['remove_outliers'].interpolate()
    # Cut remaining NaNs from beginning and end of interpolated data
    interpolated = trace_df['interpolated'].dropna()
    # Force to integer type if input data is integer
    if trace.data.dtype == int:
        interpolated.round()
        interpolated = interpolated.astype(int)

    # QC plot (optional)
    if qcplot:
        trace_df.plot()
        plt.show()

    if savedf:
        if dfpath is None:
            dfpath = '{}_spike_removal_{}.csv'.format(trace.id, datetime.now())
        trace_df.to_csv(dfpath)

    # Construct output obspy.Trace and return
    interp_stats = trace.stats.copy()
    if interp_stats.location != '9X':
        interp_stats.location = '9X'
    else:
        interp_stats.location = 'TF'
    interp_stats.mseed.dataquality = 'Q'
    interp_trace = obspy.Trace(interpolated.to_numpy(), interp_stats)
    return interp_trace
