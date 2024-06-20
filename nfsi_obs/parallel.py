from matplotlib import mlab
import multiprocessing
import numpy as np
import obspy
from scipy import signal
import sys
import warnings


class PSDProcess(multiprocessing.Process):
    def __init__(self, in_queue, result_queue, done_event, calc_acc=False, psd_kwargs=None):
        super().__init__()
        self.queue = in_queue
        self.result_out = result_queue
        self.done_event = done_event
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

        self.done_event.set()
        sys.exit(0)


def calc_psds_from_queue(in_queue, result_queue, done_event, calc_acc=False, num=0, **psd_kwargs):
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

    print('Process {} finished'.format(num))
    #done_event.set()


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
    num_cores = multiprocessing.cpu_count()
    if max_processes is None:
        max_processes = int(max([1, num_cores * 0.75 - 1]))
    if max_processes > num_cores:
        warnings.warn('Specified number of processes ({0}) is greater than number of cores available ({1})'.format(max_processes, num_cores))
        max_processes = num_cores - 1
    print('Using {0} processes to calculate PSD curves for {1} windows'.format(max_processes, num_windows))
    finished_processing = multiprocessing.Event()
    #processes = [PSDProcess(trace_queue, results, finished_processing, calc_acc, psd_kwargs) for i in range(max_processes)]
    processes = [multiprocessing.Process(target=calc_psds_from_queue, args=(trace_queue, results, finished_processing, calc_acc, i), kwargs=psd_kwargs) for i in range(max_processes)]

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
