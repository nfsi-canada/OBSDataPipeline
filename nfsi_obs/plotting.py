import gc
import matplotlib.pyplot as plt
from matplotlib import mlab
import numpy as np
import obspy
import os
import re
from scipy import signal
import timeit

from .waveform import WaveformPlotting
from .metadata import get_channel_type, update_metadata
from .extenders import cut_trace
from .parallel import calc_psds_thread_pool


QARTOD_COLOURS = {
    1: 'g',
    2: 'b',
    3: 'y',
    4: 'r',
    9: '0.5'
}

# Global seismic noise models from Peterson (1993), for plotting with acceleration PSDs
NLNM_A = [-162.36, -166.7, -170., -166.4, -168.6, -159.98, -141.1, -71.36, -97.26, -132.18, -205.27, -37.65, -114.37, -160.58, -187.5, -216.47, -185., -168.34, -217.43, -258.28, -346.88]
NLNM_B = [5.64, 0., -8.3, 28.9, 52.48, 29.81, 0., -99.77, -66.49, -31.57, 36.16, -104.33, -47.1, -16.28, 0., 15.7, 0., -7.61, 11.9, 26.6, 48.75]
NLNM_P = [0.1, 0.17, 0.4, 0.8, 1.24, 2.4, 4.3, 5., 6., 10., 12., 15.6, 21.9, 31.6, 45., 70., 101., 154., 328., 600., 10000.]
NLNM = [[1./p for p in NLNM_P], [a + b * np.log10(p) for a, b, p in zip(NLNM_A, NLNM_B, NLNM_P)]]

NHNM_A = [-108.73, -150.34, -122.31, -116.85, -108.48, -74.66, 0.66, -93.37, 73.54, -151.52, -206.66]
NHNM_B = [-17.23, -80.5, -23.87, 32.51, 18.08, -32.95, -127.18, -22.42, -163.98, 10.01, 31.63]
NHNM_P = [0.1, 0.22, 0.32, 0.8, 3.8, 4.6, 6.3, 7.9, 15.4, 20., 354.8]
NHNM = [[1./p for p in NHNM_P], [a + b * np.log10(p) for a, b, p in zip(NHNM_A, NHNM_B, NHNM_P)]]


def month_start_end(dttm):
    """
    Return start and end of current month for input date/time.

    :param dttm: obspy.UTCDateTime object
    :return: start and end of current month, obspy.UTCDateTime
    """
    yr = dttm.year
    mn = dttm.month + 1
    start = obspy.UTCDateTime(yr, mn - 1, 1)
    if mn > 12:
        yr += 1
        mn -= 12
    end = obspy.UTCDateTime(yr, mn, 1)

    return start, end


def round_obspy_date(dttm):
    """ Get date of obspy.UTCDateTime object """
    return obspy.UTCDateTime(dttm.year, dttm.month, dttm.day)


def date_ticks(start, end, max_ticks=10):
    """
    Return list of tick locations and labels at appropriate spacing (integer days) between start and end times

    :param start: start date/time, obspy.UTCDateTime
    :param end: end date/time, obspy.UTCDateTime
    :param max_ticks: maximum number of ticks, default 10
    :return:
    """
    round_start = round_obspy_date(start)
    round_end = round_obspy_date(end + 24 * 60 * 60 - 1)

    span = round_end - round_start
    max_interval = span / (max_ticks - 1)   # time between ticks in seconds
    day_interval = np.ceil(max_interval / 60 / 60 / 24)    # time between ticks in days

    tm_ticks, tm_ticklabels = [round_start.timestamp], [round_start.strftime('%Y-%m-%d')]
    dt = round_start + day_interval * 24 * 60 * 60
    while dt <= round_end:
        tm_ticks.append(dt.timestamp)
        tm_ticklabels.append(dt.strftime('%Y-%m-%d'))
        dt += day_interval * 24 * 60 * 60

    return tm_ticks, tm_ticklabels


def trace_plot(trace, outdir, dmin=None, dmax=None, qc_config=None, use_existing_plots=False, strict_lims=False):
    """
    Make time series plot(s) of an obspy.core.trace.Trace object, raw and corrected (if response information included).

    :param trace: obspy.core.trace.Trace object
    :param outdir: path to output directory
    :param dmin: minimum data value for plot y-axis
    :param dmax: maximum data value for plot y-axis
    :param qc_config: dictionary of QC configuration in ioos_qc compatible format, optional
    :param use_existing_plots: check if plots exist and do not re-create if present, False by default
    :param strict_lims: if True, y-axis plot limits `dmin` and `dmax` are used regardless of data range; False by
    default, which will use automatic plot limits if smaller than range (dmin, dmax)

    :return: path to plot PNG file
    """
    # Add horizontal bars for QARTOD fail and suspect thresholds
    qc_bars = False
    low_fail, high_fail, low_sus, high_sus = None, None, None, None
    if qc_config is not None:
        if 'qartod' in qc_config:
            if 'gross_range_test' in qc_config['qartod']:
                ranges = qc_config['qartod']['gross_range_test']
                if 'fail_span' in ranges:
                    low_fail = [-1e32, float(ranges['fail_span'][0])]
                    high_fail = [float(ranges['fail_span'][1]), 1e32]
                    if dmin is not None:
                        if dmin < float(ranges['fail_span'][0]):
                            low_fail[0] = dmin
                        else:
                            low_fail = None     # low fail threshold outside plot limits
                    if dmax is not None:
                        if dmax > float(ranges['fail_span'][1]):
                            high_fail[1] = dmax
                        else:
                            high_fail = None    # high fail threshold outside plot limits
                if 'suspect_span' in ranges:
                    low_sus = [-1e32, float(ranges['suspect_span'][0])]
                    high_sus = [float(ranges['suspect_span'][1]), 1e32]
                    if low_fail is not None:
                        if low_fail[1] < float(ranges['suspect_span'][0]):
                            low_sus[0] = low_fail[1]
                        else:
                            low_sus = None  # Low fail threshold is equal or greater than low suspect threshold
                    elif dmin is not None:
                        if dmin < float(ranges['suspect_span'][0]):
                            low_sus[0] = dmin
                        else:
                            low_sus = None  # Low suspect threshold outside plot limits

                    if high_fail is not None:
                        if high_fail[0] > float(ranges['suspect_span'][1]):
                            high_sus[1] = high_fail[0]
                        else:
                            high_sus = None     # High fail threshold is equal or less than high suspect threshold
                    elif dmax is not None:
                        if dmax > float(ranges['suspect_span'][1]):
                            high_sus[1] = dmax
                        else:
                            high_sus = None     # High suspect threshold outside plot limits
    if any([low_sus is not None, low_fail is not None, high_sus is not None, high_fail is not None]):
        qc_bars = True

    # Apply instrument sensitivity if provided
    if hasattr(trace.meta, 'response'):
        trace.remove_sensitivity()
        # Plot data in real units
        full_data_plot = os.path.join(outdir, 'full_{0}.png'.format(trace.id))
        if not (use_existing_plots and os.path.isfile(full_data_plot)):
            waveform = WaveformPlotting(stream=trace, handle=True)
            fig = waveform.plot_waveform(label_traces=False)
            ax = plt.gca()
            if qc_bars:
                for fail in [low_fail, high_fail]:
                    # TODO: Low_fail bar gives a diagonal line (not horizontal) for external temperature, fine for other traces...
                    if fail is not None:
                        ax.axhspan(fail[0], fail[1], alpha=0.1, color='r')
                for sus in [low_sus, high_sus]:
                    if sus is not None:
                        ax.axhspan(sus[0], sus[1], alpha=0.1, color='y')
            if hasattr(trace.meta, 'description'):
                ax.set_ylabel("{0} ({1})".format(trace.meta.description, trace.meta.response.instrument_sensitivity.input_units))
            # Set axis y-limits (if necessary)
            if not strict_lims:
                auto_y = ax.get_ylim()
                if (dmin is None) or (dmin < auto_y[0]):
                    dmin = auto_y[0]
                if (dmax is None) or (dmax > auto_y[1]):
                    dmax = auto_y[1]
            ax.set_ylim(dmin, dmax)
            ax.grid(True, ls=':')
            fig.savefig(full_data_plot)
            plt.close(fig)
        return full_data_plot
    else:
        # Plot raw data (counts as recorded)
        raw_data_plot = os.path.join(outdir, 'raw_{0}.png'.format(trace.id))
        if not (use_existing_plots and os.path.isfile(raw_data_plot)):
            if qc_bars and not hasattr(trace.meta, 'response'):
                # Only plot QC ranges here if no response info included (otherwise values will be meaningless here)
                waveform = WaveformPlotting(stream=trace, handle=True)
                fig = waveform.plot_waveform(label_traces=False)
                ax = plt.gca()
                for fail in [low_fail, high_fail]:
                    if fail is not None:
                        ax.axhspan(fail[0], fail[1], alpha=0.1, color='r')
                for sus in [low_sus, high_sus]:
                    if sus is not None:
                        ax.axhspan(sus[0], sus[1], alpha=0.1, color='y')
                # Set axis y-limits (if necessary)
                if not strict_lims:
                    auto_y = ax.get_ylim()
                    if (dmin is None) or (dmin < auto_y[0]):
                        dmin = auto_y[0]
                    if (dmax is None) or (dmax > auto_y[1]):
                        dmax = auto_y[1]
                ax.set_ylim(dmin, dmax)
                ax.grid(True, ls=':')   # TODO: Grid not plotting...
                # Save figure
                fig.savefig(raw_data_plot)
                plt.close(fig)
            else:
                waveform = WaveformPlotting(stream=trace, outfile=raw_data_plot)
                waveform.plot_waveform(label_traces=False)
        return raw_data_plot


def qartod_plot(trace, outdir, check='gross_range_check', use_existing_plots=False):
    """
    Make time series plot of an obspy.core.trace.Trace object containing QC results from a QARTOD test (ioos_qc format).

    :param trace: obspy.core.trace.Trace object
    :param outdir: path to output directory
    :param check: string representing the type of QC check performed, used in output plot file name
    :param use_existing_plots: check if plots exist and do not re-create if present, False by default

    :return: path to plot PNG file
    """
    # Plot test results
    qc_plot = os.path.join(outdir, 'QC_{0}_{1}.png'.format(check, trace.id))
    if not (use_existing_plots and os.path.isfile(qc_plot)):
        waveform = WaveformPlotting(stream=trace, handle=True, linestyle=None, color=QARTOD_COLOURS, marker='.', qartod=True)
        fig = waveform.plot_waveform(label_traces=False)
        ax = plt.gca()
        ax.set_yticks([1, 2, 3, 4])
        ax.set_yticklabels(['pass', 'undetermined', 'suspect', 'fail'])
        plt.grid(True, ls=':')
        fig.savefig(qc_plot)
        plt.close(fig)
    return qc_plot


def spectrogram(trace, outdir, spec_win, overlap, sub_overlap=0.75, cmap=None, slim=[None, None], use_existing_plots=False):
    """
    Plot spectrogram of seismic data (as obspy.core.trace.Trace object)

    :param trace: obspy.core.trace.Trace object
    :param outdir: path to output directory
    :param spec_win: spectrogram window in seconds
    :param overlap: window overlap (0-1)
    :param use_existing_plots: check if plots exist and do not re-create if present, False by default

    :return: path to plot PNG file
    """
    seismometer = False
    if re.match(r'[A-Z]H[1-3ABCENRTUVWZ]', trace.stats.channel):
        seismometer = True

    # Calculate all PSDs
    apsds, vpsds, freqs, times = calc_psds(trace, spec_win, overlap, sub_overlap, calc_acc=seismometer)

    # Spectrogram plot from PSDs
    spectrogram_plot = os.path.join(outdir, 'spec_{0}.png'.format(trace.id))
    # TODO: Set appropriate x-ticks for time span
    if not (use_existing_plots and os.path.isfile(spectrogram_plot)):
        if cmap is None:
            cmap = 'viridis'
        npts = int(spec_win * trace.stats.sampling_rate)
        nover = int(overlap * npts)

        spec_fig, sax = plt.subplots(1, 1, num=1, clear=True, figsize=(8, 4.8))
        if seismometer:
            spec_psds = 10. * np.log10(np.transpose(apsds))
        else:
            spec_psds = 10. * np.log10(np.transpose(vpsds))
        spec_psds = np.flipud(spec_psds)

        pad_xextent = (npts - nover) / trace.stats.sampling_rate / 2
        xextent = np.min(times) - pad_xextent, np.max(times) + pad_xextent
        xmin, xmax = xextent
        extent = xmin, xmax, freqs[0][0], freqs[0][-1]

        im = sax.imshow(spec_psds, cmap=cmap, extent=extent, vmin=slim[0], vmax=slim[1], origin='upper')
        sax.axis('auto')
        sax._sci(im)
        sax.set_yscale('log')
        sax.set_ylim(ymin=8e-3, ymax=trace.stats.sampling_rate / 2)
        sax.set_ylabel('Frequency (Hz)')
        plt.tight_layout()
        spec_fig.savefig(spectrogram_plot)

    return spectrogram_plot


def calc_psds(trace, win_len, overlap, sub_overlap, endtime=None, buffered=False, calc_acc=False, seg_len=pow(2, 17)):
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
    #seg_len = pow(2, 17)
    freqs, vel_psds, times = [], [], []
    next_win_start = None
    if buffered:
        next_win_start = trace.stats.starttime

    # Calculate PSDs in velocity
    hit_end = False
    for sect in trace.slide(win_len, win_len * (1 - overlap), nearest_sample=False):
        if endtime is not None:
            if sect.stats.starttime > endtime:
                hit_end = True
                continue    # skip windows which start after `endtime` and reset next start to include last window in next section of buffer
        psd, frq = mlab.psd(sect.data, NFFT=seg_len, Fs=trace.meta.sampling_rate,
                            noverlap=int(sub_overlap*seg_len),
                            window=signal.get_window('hann', seg_len, False), detrend='linear')
        freqs.append(frq)
        vel_psds.append(psd)
        midpoint = sect.stats.starttime + (sect.stats.endtime - sect.stats.starttime) / 2
        times.append(midpoint.timestamp)
        if buffered:
            next_win_start = next_win_start + win_len * (1 - overlap)

    if hit_end:
        next_win_start = next_win_start - win_len * (1 - overlap)

    # Convert PSDs to acceleration (if necessary)
    acc_psds = []
    if calc_acc:
        for f, p in zip(freqs, vel_psds):
            apsd = p * (2 * np.pi * f) * (2 * np.pi * f)
            acc_psds.append(apsd)

    if buffered:
        return acc_psds, vel_psds, freqs, times, next_win_start
    else:
        return acc_psds, vel_psds, freqs, times


def psd_plot(trace, outdir, win_len, overlap, sub_overlap=0.75, density=False, use_existing_plots=False):
    """
    Plot PSDs of seismic data (as obspy.core.trace.Trace object). If the input trace is from a seismometer (channel code
    "H"), the returned plot will be in acceleration. Otherwise, the plot will be in sensor units (e.g. pressure).

    :param trace: obspy.core.trace.Trace object
    :param outdir: path to output directory
    :param win_len: window length for each PSD curve
    :param overlap: fractional window overlap (0-1)
    :param sub_overlap: fractional overlap for sub-windows used in PSD calculation (Welch's average periodogram method)
    :param density: if True, plot as probabilistic PSD (heatmap density of curves); False by default
    :param use_existing_plots: check if plots exist and do not re-create if present; False by default

    :return: path to plot PNG file
    """
    seismometer = False
    if re.match(r'[A-Z]H[1-3ABCENRTUVWZ]', trace.stats.channel):
        seismometer = True
        psd_asis = os.path.join(outdir, 'psd_seismic_vel_{0}.png'.format(trace.id))
        psd_a_plot = os.path.join(outdir, 'psd_seismic_acc_{0}.png'.format(trace.id))
        if use_existing_plots and os.path.isfile(psd_asis) and os.path.isfile(psd_a_plot):
            return psd_a_plot
    else:
        psd_asis = os.path.join(outdir, 'psd_{0}.png'.format(trace.id))
        if use_existing_plots and os.path.isfile(psd_asis):
            return psd_asis

    # Calculate all PSDs
    apsds, vpsds, freqs, times = calc_psds(trace, win_len, overlap, sub_overlap, calc_acc=seismometer)

    # TODO: Allow plot as probabilistic PSD (density of curves)

    # Plot PSDs in sensor units (velocity or pressure)
    if not (use_existing_plots and os.path.isfile(psd_asis)):
        psd_v_fig, vax = plt.subplots(1, 1, num=1, clear=True)
        for f, v in zip(freqs, vpsds):
            vax.plot(f, 10 * np.log10(v), c='0.7', lw=0.5, marker=None)
        vax.set_xscale('log')
        vax.set_xlabel('Frequency (Hz)')
        vax.set_ylabel('Power Spectral Density (dB)')
        psd_v_fig.savefig(psd_asis)

    # Plot acceleration PSDs (if channel is a seismometer)
    if seismometer:
        if not (use_existing_plots and os.path.isfile(psd_a_plot)):
            psd_a_fig, aax = plt.subplots(1, 1, num=1, clear=True)
            aax.plot(NLNM[0], NLNM[1], c='k', lw=0.5, marker=None)
            aax.plot(NHNM[0], NHNM[1], c='k', lw=0.5, marker=None)
            for f, a in zip(freqs, apsds):
                aax.plot(f, 10 * np.log10(a), c='0.8', lw=0.5, marker=None)
            aax.set_xscale('log')
            plt.grid(True, ls=':')
            aax.set_xlabel('Frequency (Hz)')
            aax.set_ylabel('Power Spectral Density (dB)')
            aax.set_xlim(xmin=1e-3)
            psd_a_fig.savefig(psd_a_plot)

        return psd_a_plot

    return psd_asis


def next_plot_window(prev_end, plot_length=None):
    """
    Update plot_end for next time window. Used with buffer_seismic_data(). Default behaviour is 1 calendar month per plot.
    """
    plot_start = prev_end
    if plot_length is None:
        next_mn = plot_start.month + 1
        if next_mn > 12:
            plot_end = obspy.UTCDateTime(plot_start.year + 1, next_mn - 12, 1)
        else:
            plot_end = obspy.UTCDateTime(plot_start.year, next_mn, 1)
    else:
        plot_end = plot_start + (plot_length * 24 * 60 * 60)

    return plot_start, plot_end


def plot_filenames(outdir, ch_id, plot_start, plot_end, asis=False):
    spec_psd_plot = os.path.join(outdir, 'spec_psd_{0}_{1}_to_{2}.png'.format(ch_id, plot_start.strftime('%Y-%m-%d'),
                                                                              (plot_end - 1).strftime('%Y-%m-%d')))
    spectrogram_plot = os.path.join(outdir, 'spec_{0}_{1}_to_{2}.png'.format(ch_id, plot_start.strftime('%Y-%m-%d'),
                                                                             (plot_end - 1).strftime('%Y-%m-%d')))
    if asis:
        psd_asis = os.path.join(outdir, 'psd_{0}_{1}_to_{2}.png'.format(ch_id, plot_start.strftime('%Y-%m-%d'),
                                                                        (plot_end - 1).strftime('%Y-%m-%d')))
        return psd_asis, spec_psd_plot, spectrogram_plot
    else:
        psd_v_plot = os.path.join(outdir, 'psd_vel_{0}_{1}_to_{2}.png'.format(ch_id, plot_start.strftime('%Y-%m-%d'),
                                                                              (plot_end - 1).strftime('%Y-%m-%d')))
        psd_a_plot = os.path.join(outdir, 'psd_acc_{0}_{1}_to_{2}.png'.format(ch_id, plot_start.strftime('%Y-%m-%d'),
                                                                              (plot_end - 1).strftime('%Y-%m-%d')))
        return psd_a_plot, spec_psd_plot, spectrogram_plot, psd_v_plot


def buffer_seismic_data(files, outdir, g_log, net_id='XX', station_info=None, channel_map=None, project_meta=None,
                        psd_win=3600, spec_win=3600, overlap=0.5, psd_over=0.75, plot_length=None, ch_id=None,
                        start=None, end=None, detrend=False, spec_cmap=None, use_existing_plots=False, parallel=False,
                        max_processes=None):
    """
    Analyze seismic data stored in raw data files and create PSD and spectrogram plots. File paths in *files* should be
    listed in chronological order. Files must be readable by obspy.read()

    :param files: list of paths for raw data files
    :param outdir: path to output directory
    :param g_log: Logging object, specifying general log used by the calling script
    :param net_id: 2-character FDSN network code, default 'XX' for test data
    :param station_info
    :param channel_map
    :param project_meta
    :param psd_win: window length for each PSD curve in seconds, default 3600 (1 hour)
    :param spec_win: window length for spectrogram in seconds, default 3600 (1 hour)
    :param overlap: fractional window overlap (0-1), used for both PSD and spectrogram, default 50%
    :param psd_over: fractional overlap for sub-windows used in PSD calculation (Welch's average periodogram method), default 75%
    :param plot_length: optional length of time period to plot in each output PNG in days, otherwise defaults to
    plotting by calendar month
    :param ch_id: optional channel identifier to specify which channel to plot in multi-channel data files
    :param start: start time for data to be analyzed (e.g. when OBS reaches seafloor)
    :param end: end time for data to be analyzed (e.g. when OBS releases from anchor)
    :param detrend: if True, remove trend from trace data (RMS linear fit)
    :param spec_cmap: colormap to use for spectrogram plot
    :param use_existing_plots: check if plots exist and do not re-create if present, False by default
    :param parallel: run PSD calculation with multiprocessing parallelization

    :return: dictionary of channel information for auto-report generation
    """
    # TODO: Error on last file if after end date/time (see AQU-C760 drifter (2S.L129) data)

    func_start = timeit.default_timer()
    timing = {
        'setup': 0.,
        'loop_setup': 0.,
        'file_read': 0.,
        'time_cut': 0.,
        'plot_admin': 0.,
        'buffer_build': 0.,
        'apply_meta': 0.,
        'this_channel': 0.,
        'meta_admin': 0.,
        'gap_test': 0.,
        'data_clean': 0.,
        'psd_calc': 0.,
        'psd_buffer': 0.,
        'spec_calc': 0.,
        'trace_plot': 0.,
        'psd_plot': 0.,
        'spec_plot': 0.,
        'report_info': 0.,
        'array_reset': 0.,
    }
    report_info = {'order': 100}
    channel_info = None
    hydrophone = False
    if ch_id is not None:
        input_type = get_channel_type(ch_id.split('.')[-1])
        report_info['channelType'] = input_type
        if input_type != 'seismic':
            g_log.warn('Data buffering not yet implemented for non-seismic channel {0} of type {1}'.format(ch_id, input_type))
            return report_info

        if re.match(r'[A-Z]D[HF]', ch_id.split('.')[-1]):
            hydrophone = True

    # If no start/end information given, fallback to start/end dates from project metadata (JSON or [future] ST integration)
    if start is None and 'start_date' in project_meta:
        start = obspy.UTCDateTime(project_meta['start_date'])
    if end is None and 'end_date' in project_meta:
        end = obspy.UTCDateTime(project_meta['end_date']) + 24 * 60 * 60

    # Default colormap for spectrogram if none specified
    if spec_cmap is None:
        spec_cmap = 'viridis'

    buffer_length = 2   # number of files to keep in memory at a given time; testing shows using more files per buffer loop does not improve performance
    psd_a_plots, psd_v_plots, spec_plots = [], [], []
    all_gaps = []

    i = 0
    latest_data = None
    last_start = None
    first_psd_start = None
    next_psd_start = None
    first_spec_start = None     # technically allow PSD and spectrogram to have different window lengths at the moment..
    next_spec_start = None
    plot_end = None
    plot_start = obspy.UTCDateTime(1970, 1, 1)
    make_plot = False   # only create a plot when necessary
    psd_temp_results = {
        'psd_array': None,
        'vpsd_array': None,
        'psd_freqs': None,
        'psd_times': None
    }
    timing['setup'] += timeit.default_timer() - func_start

    proc_complete, all_data_read = False, False
    while not proc_complete:
        try:
            loop_start = timeit.default_timer()
            g_log.info('Starting buffer loop...')
            # Read data into buffer, keep copy of last file read
            buffer = obspy.Stream()
            files_in_buffer = 0
            setup_time = timeit.default_timer()
            timing['loop_setup'] += setup_time - loop_start

            # Read next data file if nothing saved from previous loop iteration
            if latest_data is None:
                try:
                    latest_data = obspy.read(files[i])
                    g_log.info('Read file {0}'.format(files[i]))
                    print(latest_data)
                except Exception as e:
                    g_log.error(str(e))
                    g_log.error('Error reading file {0}'.format(files[i]))
                    latest_data = obspy.Stream()
                i += 1
                if i >= len(files):
                    all_data_read = True
            done_read = timeit.default_timer()
            timing['file_read'] += done_read - setup_time

            # Trim data to window of interest
            latest_data.trim(start, end, nearest_sample=False)
            # Check for empty stream (no data in file, or no data within window of interest)
            if len(latest_data) < 1:
                latest_data = None
                continue
            trim_time = timeit.default_timer()
            timing['time_cut'] += trim_time - done_read

            # Set start and end of current plot time window if not already set (should only need for first loop iteration)
            if plot_end is None:
                if plot_length is None:
                    # default behaviour plots a single calendar month in each image
                    plot_start, plot_end = month_start_end(latest_data[0].stats.starttime)
                else:
                    # time window in days specified by plot_length
                    plot_start = round_obspy_date(latest_data[0].stats.starttime)
                    plot_end = plot_start + (plot_length * 24 * 60 * 60)

            # Check for existing plots with current plot start/end, skip if use_existing_plots == True
            if ch_id is not None:
                plot_files = plot_filenames(outdir, ch_id, plot_start, plot_end, asis=hydrophone)
                files_exist = [os.path.isfile(fn) for fn in plot_files]
                while use_existing_plots and all(files_exist):
                    if len(plot_files) > 3:
                        psd_v_plots.append(plot_files[3])
                    psd_a_plots.append({
                        'image': plot_files[0],
                        'start': plot_start.strftime('%Y-%m-%d'),
                        'end': (plot_end - 1).strftime('%Y-%m-%d')
                    })
                    spec_plots.append({
                        'image': plot_files[1],
                        'start': plot_start.strftime('%Y-%m-%d'),
                        'end': (plot_end - 1).strftime('%Y-%m-%d')
                    })

                    plot_start, plot_end = next_plot_window(plot_end, plot_length=plot_length)
                    plot_files = plot_filenames(outdir, ch_id, plot_start, plot_end, asis=hydrophone)
                    files_exist = [os.path.isfile(fn) for fn in plot_files]
                    # TODO: Skip unnecessary PSD calculations if using existing plots
            plot_setup = timeit.default_timer()
            timing['plot_admin'] += plot_setup - trim_time

            # Add trace data from first data file to buffer
            for tr in latest_data:
                if ch_id is not None:
                    if tr.id != ch_id:
                        continue  # ignore all other channels if *ch_id* is specified
                buffer.append(tr.copy())  # have to add one trace at a time to existing Stream object
                if (tr.stats.starttime > last_start) or (last_start is None):
                    last_start = tr.stats.starttime
            files_in_buffer += 1
            buffer.merge()

            mid_plot = True
            if buffer.count() > 0:
                mid_plot = (buffer[0].stats.endtime < plot_end)
            buffer_init = timeit.default_timer()
            timing['buffer_build'] += buffer_init - plot_setup

            # Fill remaining space in buffer with new files, keeping a copy of the last one read as "latest_data"
            while (files_in_buffer < buffer_length) and mid_plot and (i < len(files)) and (not all_data_read):
                read_start = timeit.default_timer()
                try:
                    latest_data = obspy.read(files[i])
                    g_log.info('Read file {0}'.format(files[i]))
                    print(latest_data)
                    i += 1
                    done_read = timeit.default_timer()
                    timing['file_read'] += done_read - read_start
                except Exception as e:
                    msg = str(e)
                    g_log.error(msg)
                    g_log.error('Error reading file {0}'.format(files[i]))
                    i += 1
                    done_read = timeit.default_timer()
                    timing['file_read'] += done_read - read_start
                    continue
                latest_data.trim(start, end, nearest_sample=False)  # trim to time window of interest
                trim_time = timeit.default_timer()
                timing['time_cut'] += trim_time - done_read
                if len(latest_data) > 0:
                    for tr in latest_data:
                        if ch_id is not None:
                            if tr.id != ch_id:
                                continue    # ignore all other channels if *ch_id* is specified
                        buffer.append(tr.copy())   # have to add one trace at a time to existing Stream object
                        if (tr.stats.starttime > last_start) or (last_start is None):
                            last_start = tr.stats.starttime
                    files_in_buffer += 1
                    buffer.merge()
                if buffer.count() > 0:
                    mid_plot = (buffer[0].stats.endtime < plot_end)     # Complains if there are no traces in the buffer (e.g. no matching channel IDs from latest data)
                buff_add = timeit.default_timer()
                timing['buffer_build'] += buff_add - trim_time

            if i >= len(files):
                all_data_read = True
                make_plot = True

            if ch_id is None:
                ch_id = buffer[0].id    # channel ID before correction (use to ensure same channel analyzed throughout)
                if re.match(r'[A-Z]D[HF]', ch_id.split('.')[-1]):
                    hydrophone = True

            buffer_full = timeit.default_timer()
            # Update metadata from other sources
            buffer = update_metadata(buffer, net_id, g_log, station_info, channel_map, project_meta)
            meta_time = timeit.default_timer()
            timing['apply_meta'] += meta_time - buffer_full

            if len(buffer) > 1:
                g_log.warn('Multiple channels present in data files, analyzing first one only: {0}'.format(buffer[0].id))
            this_channel = buffer[0]    # Only look at first channel in files
            pull_time = timeit.default_timer()
            timing['this_channel'] += pull_time - meta_time

            report_info['seedID'] = this_channel.id
            report_info['channelName'] = this_channel.id
            for metaKey, reportKey in zip(['description', 'azimuth', 'dip'], ['channelName', 'azimuth', 'dip']):
                if hasattr(this_channel.meta, metaKey):
                    report_info[reportKey] = this_channel.meta[metaKey]

            # Check that this is a seismic channel
            channel_type = get_channel_type(this_channel.stats.channel)
            report_info['channelType'] = channel_type
            if channel_type != 'seismic':
                g_log.warn('Data buffering only implemented for seismic channels. Channel {0} is type {1}.'.format(this_channel.id, channel_type))
                timing['meta_admin'] += timeit.default_timer() - pull_time
                return report_info, all_gaps, timing

            if channel_info is None:
                if project_meta is not None:
                    try:
                        channel_info = list(filter(lambda ch: ch['channel_id'] == this_channel.id.split('.')[-1], project_meta['channels']))[0]
                    except (KeyError, IndexError):
                        pass

            spec_lim = [None, None]
            if channel_info is not None:
                if 'hide' in channel_info:
                    if channel_info['hide']:
                        g_log.info('Channel {0} hidden from report. Skipping analysis.'.format(this_channel.id))
                        timing['meta_admin'] += timeit.default_timer() - pull_time
                        return report_info, all_gaps, timing
                if 'spec_min' in channel_info:
                    spec_lim[0] = float(channel_info['spec_min'])
                if 'spec_max' in channel_info:
                    spec_lim[1] = float(channel_info['spec_max'])
                if 'order' in channel_info:
                    report_info['order'] = int(channel_info['order'])
                if 'qc_config' in channel_info:
                    g_log.warn('QARTOD QC checks not yet implemented for buffered data, config ignored')
                if 'shift' in channel_info:
                    spec_shift = float(channel_info['shift'][this_channel.stats.station])
                    spec_lim = [(x + spec_shift) for x in spec_lim]

            more_meta = timeit.default_timer()
            timing['meta_admin'] += more_meta - pull_time

            # Gap test
            # TODO: Would be nice if this could account for overlap between consecutive buffer sections to not duplicate gap info...
            gaps = this_channel.split().get_gaps()
            if len(gaps) > 0:
                all_gaps.extend(gaps)
                g_log.info('Found {0} gap(s) or overlap(s) in recorded data'.format(len(gaps)))
                this_channel.split().print_gaps()
            gap_time = timeit.default_timer()
            timing['gap_test'] += gap_time - more_meta

            # Apply channel sensitivity and remove linear trend, if applicable
            if hasattr(this_channel.meta, 'response'):
                this_channel.remove_sensitivity()
            if detrend:
                this_channel.detrend('linear')
            data_clean = timeit.default_timer()
            timing['data_clean'] += data_clean - gap_time

            if last_start is not None:
                # Windows start from midnight UTC on the first day of data collection
                start_of_day = round_obspy_date(this_channel.stats.starttime)
                if first_psd_start is None:
                    pre_windows = np.floor((this_channel.stats.starttime - start_of_day) / (psd_win * (1 - overlap)))
                    first_psd_start = start_of_day + pre_windows * psd_win * (1 - overlap)
                if first_spec_start is None:
                    pre_windows = np.floor((this_channel.stats.starttime - start_of_day) / (spec_win * (1 - overlap)))
                    first_spec_start = start_of_day + pre_windows * spec_win * (1 - overlap)

                # If "last_end" timestamps are set from previous loop iteration, use those as "first_start" timestamps
                if next_psd_start is not None:
                    first_psd_start = next_psd_start

                if next_spec_start is not None:
                    first_spec_start = next_spec_start

            if this_channel.stats.starttime > plot_end:
                # Update plot time window if all data is out of range
                if plot_length is None:
                    # default behaviour plots a single calendar month in each image
                    plot_start, plot_end = month_start_end(this_channel.stats.starttime)
                else:
                    # time window in days specified by plot_length
                    plot_start = round_obspy_date(this_channel.stats.starttime)
                    plot_end = plot_start + (plot_length * 24 * 60 * 60)

            if (this_channel.stats.endtime > plot_end) or (i >= len(files)):
                # data in buffer spans a plot breakpoint, or last file read => make plots this pass
                make_plot = True

            plot_admin = timeit.default_timer()
            timing['plot_admin'] += plot_admin - data_clean

            # Calculate PSDs and save to running lists
            psd_start = first_psd_start
            new_data = cut_trace(this_channel, psd_start, None, nearest_sample=True, pad=True)
            if parallel:
                apsds, vpsds, freqs, times, next_psd_start = calc_psds_thread_pool(new_data, psd_win, overlap, psd_over, endtime=plot_end, buffered=True, calc_acc=(not hydrophone), max_processes=max_processes)
            else:
                apsds, vpsds, freqs, times, next_psd_start = calc_psds(new_data, psd_win, overlap, psd_over, endtime=plot_end, buffered=True, calc_acc=(not hydrophone))
            psd_calc_time = timeit.default_timer()
            timing['psd_calc'] += psd_calc_time - plot_admin

            for running, current in zip(['psd_array', 'vpsd_array', 'psd_freqs'], [apsds, vpsds, freqs]):
                if psd_temp_results[running] is None:
                    psd_temp_results[running] = current
                else:
                    psd_temp_results[running] = np.concatenate((psd_temp_results[running], current), axis=0)
            if psd_temp_results['psd_times'] is None:
                psd_temp_results['psd_times'] = times
            else:
                psd_temp_results['psd_times'] = np.concatenate((psd_temp_results['psd_times'], times), axis=None)
            psd_arr_build = timeit.default_timer()
            timing['psd_buffer'] += psd_arr_build - psd_calc_time

            spec_calc_time = timeit.default_timer()
            timing['spec_calc'] += spec_calc_time - psd_arr_build
        except Exception as e:
            # TODO: Separate error handling for data read and data analysis
            msg = 'nfsi_obs.plotting.buffer_seismic_data: Error processing raw data files, latest file: {0}'.format(files[i-1])
            g_log.error(str(e))
            g_log.error(msg)

        if make_plot:
            try:
                start_plotting = timeit.default_timer()
                # TODO: Decide about trace plot, maybe downsample to 5Hz before plotting?
                trace_time = timeit.default_timer()
                timing['trace_plot'] += trace_time - start_plotting

                plot_files = plot_filenames(outdir, this_channel.id, plot_start, plot_end, asis=hydrophone)
                get_filenames = timeit.default_timer()
                timing['plot_admin'] += get_filenames - trace_time

                # PSD plots
                if hydrophone:
                    psd_asis = plot_files[0]
                else:
                    psd_asis = plot_files[3]
                if not (use_existing_plots and os.path.isfile(psd_asis)):
                    psd_v_fig, vax = plt.subplots(1, 1, num=1, clear=True, figsize=(8, 4.8))
                    for f, v in zip(psd_temp_results['psd_freqs'], psd_temp_results['vpsd_array']):
                        vax.plot(f, 10. * np.log10(v), c='0.8', lw=0.5, marker=None)
                    vax.set_xscale('log')
                    plt.grid(True, ls=':')
                    vax.set_xlabel('Frequency (Hz)')
                    vax.set_ylabel('Power Spectral Density (dB)')
                    vax.set_xlim(xmin=1e-3)
                    plt.tight_layout()
                    psd_v_fig.savefig(psd_asis)

                if not hydrophone:
                    psd_v_plots.append(psd_asis)
                    if not (use_existing_plots and os.path.isfile(plot_files[0])):
                        psd_a_fig, aax = plt.subplots(1, 1, num=1, clear=True, figsize=(8, 4.8))
                        aax.plot(NLNM[0], NLNM[1], c='k', lw=0.5, marker=None)
                        aax.plot(NHNM[0], NHNM[1], c='k', lw=0.5, marker=None)
                        for f, a in zip(psd_temp_results['psd_freqs'], psd_temp_results['psd_array']):
                            aax.plot(f, 10. * np.log10(a), c='0.8', lw=0.5, marker=None)
                        aax.set_xscale('log')
                        plt.grid(True, ls=':')
                        aax.set_xlabel('Frequency (Hz)')
                        aax.set_ylabel('Power Spectral Density (dB)')
                        aax.set_xlim(xmin=1e-3)
                        plt.tight_layout()
                        psd_a_fig.savefig(plot_files[0])

                done_psds = timeit.default_timer()
                timing['psd_plot'] += done_psds - get_filenames
                psd_a_plots.append({
                    'image': plot_files[0],
                    'start': plot_start.strftime('%Y-%m-%d'),
                    'end': (plot_end - 1).strftime('%Y-%m-%d')
                })
                report_add = timeit.default_timer()
                timing['report_info'] += report_add - done_psds

                # X-axis ticks for spectrogram plots (actual date strings rather than timestamps)
                # TODO: Add minor ticks every day?
                tm_x_ticks, tm_x_ticklabels = date_ticks(plot_start, plot_end)

                # Spectrogram plot from PSDs
                if not (use_existing_plots and os.path.isfile(plot_files[1])):
                    sfrq = psd_temp_results['psd_freqs'][0]
                    npts = int(spec_win * this_channel.stats.sampling_rate)
                    nover = int(overlap * npts)

                    spec_fig, sax = plt.subplots(1, 1, num=1, clear=True, figsize=(8, 4.8))
                    if hydrophone:
                        spec_psds = 10. * np.log10(np.transpose(psd_temp_results['vpsd_array']))
                    else:
                        spec_psds = 10. * np.log10(np.transpose(psd_temp_results['psd_array']))
                    spec_psds = np.flipud(spec_psds)

                    pad_xextent = (npts - nover) / this_channel.stats.sampling_rate / 2
                    xextent = np.min(psd_temp_results['psd_times']) - pad_xextent, np.max(psd_temp_results['psd_times']) + pad_xextent
                    xmin, xmax = xextent
                    extent = xmin, xmax, sfrq[0], sfrq[-1]

                    im = sax.imshow(spec_psds, cmap=spec_cmap, extent=extent, vmin=spec_lim[0], vmax=spec_lim[1], origin='upper')
                    sax.axis('auto')
                    sax._sci(im)
                    sax.set_yscale('log')
                    sax.set_ylim(ymin=8e-3, ymax=this_channel.stats.sampling_rate/2)
                    sax.set_ylabel('Frequency (Hz)')
                    # Set appropriate x-ticks for time span (also changes x-lim)
                    sax.set_xticks(tm_x_ticks, tm_x_ticklabels, horizontalalignment='right')
                    sax.tick_params(axis='x', rotation=40)
                    plt.tight_layout()
                    spec_fig.savefig(plot_files[1])

                done_spec_psd = timeit.default_timer()
                timing['spec_plot'] += done_spec_psd - report_add

                spec_plots.append({
                    'image': plot_files[1],
                    'start': plot_start.strftime('%Y-%m-%d'),
                    'end': (plot_end - 1).strftime('%Y-%m-%d')
                })
                report_add = timeit.default_timer()
                timing['report_info'] += report_add - done_spec_psd

            except Exception as e:
                msg = 'nfsi_obs.plotting.buffer_seismic_data: Error creating plots, latest file: {0}'.format(files[i - 1])
                g_log.error(str(e))
                g_log.error(msg)
            finally:
                final_start = timeit.default_timer()
                # Reset temp arrays for PSDs
                psd_temp_results = {
                    'psd_array': None,
                    'vpsd_array': None,
                    'psd_freqs': None,
                    'psd_times': None
                }

                # Reset temp arrays for spectrogram
                reset_arr = timeit.default_timer()
                timing['array_reset'] += reset_arr - final_start

                make_plot = False
                # Update plot_end for next time window
                plot_start, plot_end = next_plot_window(plot_end, plot_length=plot_length)
                timing['plot_admin'] += timeit.default_timer() - reset_arr

                if all_data_read and (plot_start > this_channel.stats.endtime):
                    proc_complete = True

                gc.collect()

    add_to_report = timeit.default_timer()
    report_info['psdLoc'] = psd_a_plots
    report_info['specLoc'] = spec_plots
    report_info['hydrophone'] = hydrophone
    timing['report_info'] += timeit.default_timer() - add_to_report
    timing['trace_analysis'] = timeit.default_timer() - func_start
    return report_info, all_gaps, timing
