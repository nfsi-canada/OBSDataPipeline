from matplotlib import mlab
import multiprocessing
import numpy as np
import obspy
import psutil
from scipy import signal
import warnings

from concurrent.futures import ThreadPoolExecutor, as_completed

from .helpers import psd_period_binning_single, setup_freq_bins


class PSDProcess(multiprocessing.Process):
    def __init__(self, in_queue, result_queue, calc_acc=False, psd_kwargs=None):
        super().__init__()
        self.queue = in_queue
        self.result_out = result_queue
        self.calc_acc = calc_acc
        if psd_kwargs is not None:
            self.psd_kwargs = psd_kwargs
        else:
            self.psd_kwargs = {}

    def run(self):
        while not self.queue.empty():
            trace_window = self.queue.get()
            psd, frq = mlab.psd(trace_window.data, **self.psd_kwargs)
            midpoint = trace_window.stats.starttime + (trace_window.stats.endtime - trace_window.stats.starttime) / 2
            print(midpoint)

            result = {
                'freq': frq,
                'psd_asis': psd,
                'timestamp': midpoint.timestamp
            }

            if self.calc_acc:
                acc_psd = psd * (2 * np.pi * frq) * (2 * np.pi * frq)
                result.update({'acc_psd': acc_psd})

            # Send results out to calling Process
            self.result_out.put(result)


def calc_psds_from_queue(in_queue, result_queue, calc_acc=False, num=0, **psd_kwargs):
    while not in_queue.empty():
        trace_window = in_queue.get()
        psd, frq = mlab.psd(trace_window.data, **psd_kwargs)
        midpoint = trace_window.stats.starttime + (trace_window.stats.endtime - trace_window.stats.starttime) / 2
        #print(midpoint)

        result = {
            'freq': frq,
            'psd_asis': psd,
            'timestamp': midpoint.timestamp
        }

        if calc_acc:
            acc_psd = psd * (2 * np.pi * frq) * (2 * np.pi * frq)
            result.update({'acc_psd': acc_psd})

        # Send results out to calling Process
        result_queue.put(result)

    #print('Process {} finished'.format(num))


def calc_psds_multiproc(trace, win_len, overlap, sub_overlap, endtime=None, buffered=False, calc_acc=False, seg_len=pow(2, 17), max_processes=None):
    """
    Calculate PSDs of seismic data (as obspy.core.trace.Trace object)

    :param trace: input seismic data, measured as ground velocity
    :type trace: obspy.core.trace.Trace
    :param int win_len: window length for each PSD curve in seconds
    :param float overlap: fractional window overlap (0-1)
    :param float sub_overlap: fractional overlap for sub-windows used in PSD calculation (Welch's average periodogram method)
    :param endtime: end time for calculation window (will analyze windows which include `endtime`), obspy.UTCDateTime
    :param bool buffered: whether the input data is being processed as part of a buffer or not
    :param calc_acc: if True, assume input data is velocity (seismometer) and convert to acceleration
    :param seg_len: length of PSD segment for average periodogram method (see matplotlib.mlab.psd) in data points
    :param max_processes: maximum number of parallel processes to use for PSD calculation

    :returns: Calculated PSD curves in acceleration (if seismometer) and data units, corresponding frequencies, start of next window (if buffered is True)
    """
    # Calculate PSDs in velocity
    psd_kwargs = {
        'NFFT': seg_len,
        'Fs': trace.meta.sampling_rate,
        'noverlap': int(sub_overlap*seg_len),
        'window': signal.get_window('hann', seg_len, False),
        'detrend': 'linear'
    }

    # Add all data windows to processing Queue
    print('Building data queue...')
    trace_queue = multiprocessing.Queue()
    results = multiprocessing.Queue()
    num_windows = 0
    for sect in trace.slide(win_len, win_len * (1 - overlap), nearest_sample=False):
        if endtime is not None:
            if sect.stats.starttime > endtime:
                continue    # skip windows which start after `endtime` and reset next start to include last window in next section of buffer

        trace_queue.put(sect)
        num_windows += 1

    # Create processes
    num_cores = psutil.cpu_count(logical=False)
    if max_processes is None:
        max_processes = int(max([1, num_cores - 1]))
    if max_processes > num_cores:
        warnings.warn('Specified number of processes ({0}) is greater than number of cores available ({1})'.format(max_processes, num_cores))
        max_processes = int(max([1, num_cores - 1]))
    print('Using {0} processes to calculate PSD curves for {1} windows'.format(max_processes, num_windows))
    #processes = [PSDProcess(trace_queue, results, calc_acc, psd_kwargs) for i in range(max_processes)]
    processes = [multiprocessing.Process(target=calc_psds_from_queue, args=(trace_queue, results, calc_acc, i), kwargs=psd_kwargs) for i in range(max_processes)]

    # Start processes
    print('Starting processing...')
    for process in processes:
        process.start()

    # Collect PSD results as they are produced:
    print('Collecting PSD results...', flush=True)
    all_psds = []
    for i in range(num_windows):
        psd_result = results.get()
        all_psds.append(psd_result)
        #print(psd_result['timestamp'])

    # Wait for processes to finish (join)
    for process in processes:
        process.join()
    print('Done processing. Organizing results...')

    # Organize results array
    sorted_psds = sorted(all_psds, key=lambda p: p['timestamp'])
    freqs = [p['freq'] for p in sorted_psds]
    vel_psds = [p['psd_asis'] for p in sorted_psds]
    times = [p['timestamp'] for p in sorted_psds]
    acc_psds = []
    if calc_acc:
        acc_psds = [p['acc_psd'] for p in sorted_psds]

    # Cleanup and return
    if buffered:
        next_win_start = obspy.UTCDateTime(max(times)) - win_len * 0.5 + win_len * (1 - overlap)
        return acc_psds, vel_psds, freqs, times, next_win_start
    else:
        return acc_psds, vel_psds, freqs, times


def process_pool_init(psd_kwargs, calc_acc=False):
    """
    Initializer for Process Pool used in calc_psds_proc_pool
    :return:
    """
    global calc_acc_psds
    global psd_calc_kwargs
    calc_acc_psds = calc_acc
    psd_calc_kwargs = psd_kwargs

def calc_psds_with_pool(trace_window):
    global calc_acc_psds
    global psd_calc_kwargs

    psd, frq = mlab.psd(trace_window.data, **psd_calc_kwargs)
    midpoint = trace_window.stats.starttime + (trace_window.stats.endtime - trace_window.stats.starttime) / 2

    result = {
        'freq': frq,
        'psd_asis': psd,
        'timestamp': midpoint.timestamp
    }

    if calc_acc_psds:
        acc_psd = psd * (2 * np.pi * frq) * (2 * np.pi * frq)
        result.update({'acc_psd': acc_psd})

    return result


def calc_psds_proc_pool(trace, win_len, overlap, sub_overlap, endtime=None, buffered=False, calc_acc=False, seg_len=pow(2, 17), max_processes=None):
    """
    Calculate PSDs of seismic data (as obspy.core.trace.Trace object)

    :param trace: input seismic data, measured as ground velocity
    :type trace: obspy.core.trace.Trace
    :param int win_len: window length for each PSD curve in seconds
    :param float overlap: fractional window overlap (0-1)
    :param float sub_overlap: fractional overlap for sub-windows used in PSD calculation (Welch's average periodogram method)
    :param endtime: end time for calculation window (will analyze windows which include `endtime`), obspy.UTCDateTime
    :param bool buffered: whether the input data is being processed as part of a buffer or not
    :param calc_acc: if True, assume input data is velocity (seismometer) and convert to acceleration
    :param seg_len: length of PSD segment for average periodogram method (see matplotlib.mlab.psd) in data points
    :param max_processes: maximum number of parallel processes to use for PSD calculation

    :returns: Calculated PSD curves in acceleration (if seismometer) and data units, corresponding frequencies, start of next window (if buffered is True)
    """
    # Calculate PSDs in velocity
    psd_kwargs = {
        'NFFT': seg_len,
        'Fs': trace.meta.sampling_rate,
        'noverlap': int(sub_overlap*seg_len),
        'window': signal.get_window('hann', seg_len, False),
        'detrend': 'linear'
    }

    if endtime is not None:
        trace.trim(endtime=endtime+win_len+1)

    """
    # Add all data windows to processing Queue
    print('Building data queue...')
    trace_windows = []
    num_windows = 0
    for sect in trace.slide(win_len, win_len * (1 - overlap), nearest_sample=False):
        if endtime is not None:
            if sect.stats.starttime > endtime:
                continue    # skip windows which start after `endtime` and reset next start to include last window in next section of buffer

        trace_windows.append(sect)
        num_windows += 1
    """
    num_windows = int(((len(trace) / trace.meta.sampling_rate) - win_len) / (win_len * (1 - overlap)))

    # Create processes
    num_cores = psutil.cpu_count(logical=False)
    if max_processes is None:
        max_processes = int(max([1, num_cores * 0.75 - 1]))
    if max_processes > num_cores:
        warnings.warn('Specified number of processes ({0}) is greater than number of cores available ({1})'.format(max_processes, num_cores))
        max_processes = int(max([1, num_cores - 1]))
    print('Using {0} processes to calculate PSD curves for {1} windows'.format(max_processes, num_windows))
    pool = multiprocessing.pool.Pool(max_processes, initializer=process_pool_init, initargs=[psd_kwargs, calc_acc])

    # Process data
    print('Starting processing...')
    results = pool.map(calc_psds_with_pool, trace.slide(win_len, win_len * (1 - overlap), nearest_sample=False))

    # Organize results array
    print('Done processing. Organizing results...')
    sorted_psds = sorted(results, key=lambda p: p['timestamp'])
    freqs = [p['freq'] for p in sorted_psds]
    vel_psds = [p['psd_asis'] for p in sorted_psds]
    times = [p['timestamp'] for p in sorted_psds]
    acc_psds = []
    if calc_acc:
        acc_psds = [p['acc_psd'] for p in sorted_psds]

    # Cleanup and return
    if buffered:
        next_win_start = obspy.UTCDateTime(max(times)) - win_len * 0.5 + win_len * (1 - overlap)
        return acc_psds, vel_psds, freqs, times, next_win_start
    else:
        return acc_psds, vel_psds, freqs, times


def calc_psds_plain(trace_window, calc_acc=False, binned=False, f_bins=None, **psd_kwargs):
    """
    Calculate PSD for data in trace_window.

    :param trace_window: obspy.Trace object containing data to analyze
    :param calc_acc: if True, assume input data is velocity (seismometer) and calculate PSD both as-is and in acceleration units
    :param binned: if True, also calculate frequency-binned/smoothed version of PSD
    :param f_bins: frequency bin information, as returned by .helpers.setup_freq_bins
    :param smoothing_width_octaves: passed to setup_freq_bins
    :param step_octaves: passed to setup_freq_bins
    :param psd_kwargs: all remaining keyword arguments are passed to mlab.psd

    :return: dictionary containing PSD data (arrays of frequency, amplitude, smoothed amplitude; timestamp at midpoint of window; frequency bin information)
    """
    # default values for frequency binning
    smoothing_width = psd_kwargs.pop('smoothing_width_octaves', 0.5)
    step_octaves = psd_kwargs.pop('step_octaves', 0.125)

    psd, frq = mlab.psd(trace_window.data, **psd_kwargs)
    midpoint = trace_window.stats.starttime + (trace_window.stats.endtime - trace_window.stats.starttime) / 2

    result = {
        'freq': frq,
        'psd_asis': psd,
        'timestamp': midpoint.timestamp
    }

    if binned:
        if f_bins is None:
            f_bins = setup_freq_bins(frequencies=frq, smoothing_width_octaves=smoothing_width, step_octaves=step_octaves)
        binned_asis = psd_period_binning_single(psd[1:], frq[1:], f_bins)
        result.update({'binned_asis': binned_asis})
        result.update({'frequency_bins': f_bins})

    if calc_acc:
        acc_psd = psd * (2 * np.pi * frq) * (2 * np.pi * frq)
        result.update({'acc_psd': acc_psd})
        if binned:
            binned_acc = psd_period_binning_single(acc_psd[1:], frq[1:], f_bins)
            result.update({'binned_acc': binned_acc})

    # Send results out to calling Process
    return result


def calc_psds_thread_pool(trace, win_len, overlap, sub_overlap, endtime=None, buffered=False, calc_acc=False, binned=False, seg_len=pow(2, 17), max_processes=None, **kwargs):
    """
    Calculate PSDs of seismic data (as obspy.core.trace.Trace object). Remaining keyword arguments are passed to calc_psds_plain.

    :param trace: input seismic data, measured as ground velocity
    :type trace: obspy.core.trace.Trace
    :param int win_len: window length for each PSD curve in seconds
    :param float overlap: fractional window overlap (0-1)
    :param float sub_overlap: fractional overlap for sub-windows used in PSD calculation (Welch's average periodogram method)
    :param endtime: end time for calculation window (will analyze windows which include `endtime`), obspy.UTCDateTime
    :param bool buffered: whether the input data is being processed as part of a buffer or not
    :param calc_acc: if True, assume input data is velocity (seismometer) and convert to acceleration
    :param binned: if True, also calculate frequency-binned/smoothed version of PSDs
    :param seg_len: length of PSD segment for average periodogram method (see matplotlib.mlab.psd) in data points
    :param max_processes: maximum number of parallel processes to use for PSD calculation

    :returns: Calculated PSD curves in acceleration (if seismometer) and data units, corresponding frequencies and timestamps, frequency-binned/smoothed versions of PSDs, start of next window (if buffered is True)
    """
    # Calculate PSDs in velocity
    psd_kwargs = kwargs.copy()
    psd_kwargs.update({
        'NFFT': seg_len,
        'Fs': trace.meta.sampling_rate,
        'noverlap': int(sub_overlap*seg_len),
        'window': signal.get_window('hann', seg_len, False),
        'detrend': 'linear'
    })

    if endtime is not None:
        trace.trim(endtime=endtime+win_len+1)
    num_windows = int(((len(trace) / trace.meta.sampling_rate) - win_len) / (win_len * (1 - overlap)))

    # Create processes
    num_cores = psutil.cpu_count(logical=False)
    if max_processes is None:
        max_processes = int(max([1, num_cores * 0.75 - 1]))
    if max_processes > num_cores:
        warnings.warn('Specified number of threads ({0}) is greater than number of cores available ({1})'.format(max_processes, num_cores))
        max_processes = int(max([1, num_cores - 1]))
    print('Using {0} threads to calculate PSD curves for {1} windows'.format(max_processes, num_windows))

    all_psds = []
    with ThreadPoolExecutor(max_workers=max_processes) as executor:
        futures = [executor.submit(calc_psds_plain, trace, calc_acc, binned, **psd_kwargs) for trace in trace.slide(win_len, win_len * (1 - overlap), nearest_sample=False)]

        # Add results to running list
        for future in as_completed(futures):
            result = future.result()
            all_psds.append(result)

    print('Done processing. Organizing results...')

    # Organize results array
    sorted_psds = sorted(all_psds, key=lambda p: p['timestamp'])
    freqs = [p['freq'] for p in sorted_psds]
    vel_psds = [p['psd_asis'] for p in sorted_psds]
    times = [p['timestamp'] for p in sorted_psds]
    acc_psds, binned_asis, binned_acc = [], [], []
    if calc_acc:
        acc_psds = [p['acc_psd'] for p in sorted_psds]
    if binned:
        binned_asis = [p['binned_asis'] for p in sorted_psds]
        if calc_acc:
            binned_acc = [p['binned_acc'] for p in sorted_psds]

    # Cleanup and return
    if buffered:
        next_win_start = obspy.UTCDateTime(max(times)) - win_len * 0.5 + win_len * (1 - overlap)
        return acc_psds, vel_psds, freqs, times, binned_asis, binned_acc, next_win_start
    else:
        return acc_psds, vel_psds, freqs, times, binned_asis, binned_acc
