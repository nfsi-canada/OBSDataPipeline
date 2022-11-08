import matplotlib.pyplot as plt
from matplotlib import mlab
import numpy as np
import obspy
import os
from scipy import signal

from .waveform import WaveformPlotting


QARTOD_COLOURS = {
    1: 'g',
    2: 'b',
    3: 'y',
    4: 'r',
    9: '0.5'
}


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


def trace_plot(trace, outdir, dmin=None, dmax=None, qc_config=None):
    """
    Make time series plot(s) of an obspy.core.trace.Trace object, raw and corrected (if response information included).

    :param trace: obspy.core.trace.Trace object
    :param outdir: path to output directory
    :param dmin: minimum data value for plot y-axis
    :param dmax: maximum data value for plot y-axis
    :param qc_config: dictionary of QC configuration in ioos_qc compatible format, optional

    :return: path to plot PNG file
    """
    # Add horizontal bars for QARTOD fail and suspect limits
    qc_bars = False
    low_fail, high_fail, low_sus, high_sus = None, None, None, None
    if qc_config is not None:
        if 'qartod' in qc_config:
            if 'gross_range_test' in qc_config['qartod']:
                ranges = qc_config['qartod']['gross_range_test']
                if 'fail_span' in ranges:
                    low_fail = [-1e32, float(ranges['fail_span'][0])]
                    high_fail = [float(ranges['fail_span'][1]), 1e32]
                    if dmin is not None and dmin < float(ranges['fail_span'][0]):
                        low_fail[0] = dmin
                    if dmax is not None and dmax > float(ranges['fail_span'][1]):
                        high_fail[1] = dmax
                if 'suspect_span' in ranges:
                    low_sus = [-1e32, float(ranges['suspect_span'][0])]
                    high_sus = [float(ranges['suspect_span'][1]), 1e32]
                    if low_fail is not None:
                        if low_fail[1] < float(ranges['suspect_span'][0]):
                            low_sus[0] = low_fail[1]
                        else:
                            low_sus = None  # Low end of fail range is equal or greater than low end of suspect range
                    elif dmin is not None and dmin < float(ranges['suspect_span'][0]):
                        low_sus[0] = dmin

                    if high_fail is not None:
                        if high_fail[0] > float(ranges['suspect_span'][1]):
                            high_sus[1] = high_fail[0]
                        else:
                            high_sus = None     # High end of fail range is equal or less than high end of suspect range
                    elif dmax is not None and dmax > float(ranges['suspect_span'][1]):
                        high_sus[1] = dmax
    if any([low_sus is not None, low_fail is not None, high_sus is not None, high_fail is not None]):
        qc_bars = True

    # TODO: Combine this with full data plot below, so "raw" plot only gets made if no instrument response present
    # Plot raw data (counts as recorded)
    raw_data_plot = os.path.join(outdir, 'raw_{0}.png'.format(trace.id))
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
        fig.savefig(raw_data_plot)
        plt.close(fig)
    else:
        waveform = WaveformPlotting(stream=trace, outfile=raw_data_plot)
        waveform.plot_waveform(label_traces=False)

    # Apply instrument sensitivity if provided
    if hasattr(trace.meta, 'response'):
        trace.remove_sensitivity()
        # Plot data in real units
        full_data_plot = os.path.join(outdir, 'full_{0}.png'.format(trace.id))
        waveform = WaveformPlotting(stream=trace, handle=True)
        fig = waveform.plot_waveform(label_traces=False)
        ax = plt.gca()
        if qc_bars:
            for fail in [low_fail, high_fail]:
                if fail is not None:
                    ax.axhspan(fail[0], fail[1], alpha=0.1, color='r')
            for sus in [low_sus, high_sus]:
                if sus is not None:
                    ax.axhspan(sus[0], sus[1], alpha=0.1, color='y')
        if hasattr(trace.meta, 'description'):
            ax.set_ylabel("{0} ({1})".format(trace.meta.description, trace.meta.response.instrument_sensitivity.input_units))
            ax.set_ylim(dmin, dmax)
        plt.grid(True, ls=':')
        fig.savefig(full_data_plot)
        plt.close(fig)
        return full_data_plot
    else:
        return raw_data_plot


def qartod_plot(trace, outdir, check='gross_range_check'):
    """
    Make time series plot of an obspy.core.trace.Trace object containing QC results from a QARTOD test (ioos_qc format).

    :param trace: obspy.core.trace.Trace object
    :param outdir: path to output directory
    :param check: string representing the type of QC check performed, used in output plot file name

    :return: path to plot PNG file
    """
    # Plot test results
    qc_plot = os.path.join(outdir, 'QC_{0}_{1}.png'.format(check, trace.id))
    waveform = WaveformPlotting(stream=trace, handle=True, linestyle=None, color=QARTOD_COLOURS, marker='.', qartod=True)
    fig = waveform.plot_waveform(label_traces=False)
    ax = plt.gca()
    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels(['pass', 'undetermined', 'suspect', 'fail'])
    plt.grid(True, ls=':')
    fig.savefig(qc_plot)
    plt.close(fig)
    return qc_plot


def spectrogram(trace, outdir, spec_win, overlap):
    """
    Plot spectrogram of seismic data (as obspy.core.trace.Trace object)

    :param trace: obspy.core.trace.Trace object
    :param outdir: path to output directory
    :param spec_win: spectrogram window in seconds
    :param overlap: window overlap (0-1)

    :return: path to plot PNG file
    """
    spectrogram_plot = os.path.join(outdir, 'spec_{0}.png'.format(trace.id))
    # Alternate spectrogram method (lower memory usage than through obspy)
    npts = int(spec_win * trace.meta.sampling_rate)
    nover = int(overlap * npts)
    sfig, sax = plt.subplots(1, 1)
    plt.specgram(trace.data, NFFT=npts, Fs=trace.meta.sampling_rate, window=signal.get_window('hann', npts, False),
                 noverlap=nover, detrend='linear', scale='dB')
    sax.set_yscale('log')
    sax.set_ylim(ymin=1e-3, ymax=trace.meta.sampling_rate / 2)
    sax.set_ylabel('Frequency (Hz)')
    sfig.savefig(spectrogram_plot)

    return spectrogram_plot


def psd_plot(trace, outdir, win_len, overlap, sub_overlap):
    """
    Plot PSDs of seismic data (as obspy.core.trace.Trace object)

    :param trace: obspy.core.trace.Trace object
    :param outdir: path to output directory
    :param win_len: window length for each PSD curve
    :param overlap: fractional window overlap (0-1)
    :param sub_overlap: fractional overlap for sub-windows used in PSD calculation (Welch's average periodogram method)

    :return: path to plot PNG file
    """
    psd_v_plot = os.path.join(outdir, 'psd_seismic_vel_{0}.png'.format(trace.id))
    psd_a_plot = os.path.join(outdir, 'psd_seismic_acc_{0}.png'.format(trace.id))
    freqs, psds = [], []
    psd_v_fig, vax = plt.subplots(1, 1)
    for sect in trace.slide(win_len, win_len * (1 - overlap)):
        seg_len = pow(2, 17)
        psd, frq = plt.psd(sect.data, NFFT=seg_len, Fs=trace.meta.sampling_rate,
                           noverlap=sub_overlap*trace.meta.sampling_rate,
                           window=signal.get_window('hann', seg_len, False), detrend='linear', color='0.7',
                           linewidth=0.5)
        freqs.append(frq)
        psds.append(psd)
    vax.set_xscale('log')
    vax.set_xlabel('Frequency (Hz)')
    vax.set_ylabel('Amplitude (dB)')
    psd_v_fig.savefig(psd_v_plot)

    # Convert PSDs to acceleration and plot
    psd_a_fig, aax = plt.subplots(1, 1)
    for f, p in zip(freqs, psds):
        apsd = p * (2 * np.pi * f) * (2 * np.pi * f)
        aax.plot(f, 10 * np.log10(apsd), c='0.8', lw=0.5, marker=None)
    aax.set_xscale('log')
    plt.grid(True, ls=':')
    aax.set_xlabel('Frequency (Hz)')
    aax.set_ylabel('Amplitude (dB)')
    psd_a_fig.savefig(psd_a_plot)

    return psd_a_plot


def buffer_seismic_data(files, outdir, g_log, psd_win, spec_win, overlap, psd_over, plot_length=None, ch_id=None):
    """
    Analyze seismic data stored in raw data files and create PSD and spectrogram plots. File paths in *files* should be
    listed in chronological order. Files must be readable by obspy.read()

    :param files: list of paths for raw data files
    :param outdir: path to output directory
    :param g_log: Logging object, specifying general log used by the calling script
    :param psd_win: window length for each PSD curve in seconds
    :param spec_win: window length for spectrogram in seconds
    :param overlap: fractional window overlap (0-1), used for both PSD and spectrogram
    :param psd_over: fractional overlap for sub-windows used in PSD calculation (Welch's average periodogram method)
    :param plot_length: optional length of time period to plot in each output PNG in days, otherwise defaults to
    plotting by calendar month
    :param ch_id: optional channel identifier to specify which channel to plot in multi-channel data files

    :return: path(s) to plot PNG file(s)
    """
    buffer_length = 2   # number of files to keep in memory at a given time, will optimize later
    seg_len = pow(2, 17)    # segment length used for PSD (Welch's average periodogram method in matplotlib.mlab.psd)
    psd_a_plots, psd_v_plots, spec_plots = [], [], []

    i = 0
    latest_data = None
    last_start = None
    first_psd_start = None
    last_psd_end = None
    first_spec_start = None     # technically allow PSD and spectrogram to have different window lengths at the moment..
    last_spec_end = None
    plot_end = None
    plot_start = obspy.UTCDateTime(1970, 1, 1)
    make_plot = False   # only create a plot when necessary
    spec_array = None
    spec_times = None
    psd_array, vpsd_array, psd_freqs = [], [], []
    while i < len(files):
        # Read data into buffer, keep copy of last file read
        buffer = obspy.Stream()
        files_in_buffer = 0

        # Read next data file if nothing saved from previous loop iteration, add first file to buffer
        if latest_data is None:
            latest_data = obspy.read(files[i])
        if plot_end is None:
            # Set start and end of current plot time window if not already set
            if plot_length is None:
                # default behaviour plots a single calendar month in each image
                plot_start, plot_end = month_start_end(latest_data[0].stats.starttime)
            else:
                # time window in days specified by plot_length
                plot_start = obspy.UTCDateTime(latest_data[0].stats.starttime.year, latest_data[0].stats.starttime.julday)
                plot_end = plot_start + (plot_length * 24 * 60 * 60)

        for tr in latest_data:
            if ch_id is not None:
                if tr.id != ch_id:
                    continue  # ignore all other channels if *ch_id* is specified
            buffer.append(tr)  # have to add one trace at a time to existing Stream object
            if tr.stats.starttime > last_start:
                last_start = tr.stats.starttime
        i += 1
        files_in_buffer += 1
        buffer.merge()

        # Fill remaining space in buffer with new files, keeping a copy of the last one read as "latest_data"
        while (files_in_buffer < buffer_length) and (buffer[0].endtime < plot_end):
            latest_data = obspy.read(files[i])
            for tr in latest_data:
                if ch_id is not None:
                    if tr.id != ch_id:
                        continue    # ignore all other channels if *ch_id* is specified
                buffer.append(tr)   # have to add one trace at a time to existing Stream object
                if tr.stats.starttime > last_start:
                    last_start = tr.stats.starttime
            i += 1
            files_in_buffer += 1
            buffer.merge()

        if len(buffer) > 1:
            g_log.warn('Multiple channels present in data files, analyzing first one only: {0}'.format(buffer[0].id))
        this_channel = buffer[0]    # Only look at first channel in files

        if last_start is not None:
            # Windows start from midnight UTC on the first day of data collection
            start_of_day = obspy.UTCDateTime(this_channel.stats.starttime.year, this_channel.stats.starttime.julday)
            if first_psd_start is None:
                pre_windows = np.floor((this_channel.stats.starttime - start_of_day) / psd_win)
                first_psd_start = start_of_day + psd_win * pre_windows
            if first_spec_start is None:
                pre_windows = np.floor((this_channel.stats.starttime - start_of_day) / spec_win)
                first_spec_start = start_of_day + spec_win * pre_windows

            # If "last_end" timestamps are set from previous loop iteration, use those as "first_start" timestamps
            # otherwise set initial "last_end" timestamps
            if last_psd_end is None:
                last_psd_end = first_psd_start
            else:
                first_psd_start = last_psd_end

            if last_spec_end is None:
                last_spec_end = first_spec_start
            else:
                first_spec_start = last_spec_end

        if this_channel.stats.starttime > plot_end:
            # Update plot time window if all data is out of range
            if plot_length is None:
                # default behaviour plots a single calendar month in each image
                plot_start, plot_end = month_start_end(this_channel.stats.starttime)
            else:
                # time window in days specified by plot_length
                plot_start = obspy.UTCDateTime(this_channel.stats.starttime.year, this_channel.stats.starttime.julday)
                plot_end = plot_start + (plot_length * 24 * 60 * 60)

        if this_channel.stats.endtime > plot_end:
            # data in buffer spans a plot breakpoint => make plots this pass
            make_plot = True

        # Calculate PSDs in velocity
        psd_start = first_psd_start
        vpsds, freqs = [], []
        while psd_start < min(this_channel.stats.endtime - psd_win, plot_end):
            data = this_channel.slice(psd_start, psd_start + psd_win)
            psd, frq = mlab.psd(data.data, NFFT=seg_len, Fs=this_channel.meta.sampling_rate,
                                noverlap=psd_over*this_channel.meta.sampling_rate,
                                window=signal.get_window('hann', seg_len, False), detrend='linear')
            freqs.append(frq)
            vpsds.append(psd)
            last_psd_end = psd_start + psd_win
            psd_start = psd_start + psd_win * (1 - overlap)

        # Convert PSDs to acceleration and save to running lists
        for f, p in zip(freqs, vpsds):
            apsd = p * (2 * np.pi * f) * (2 * np.pi * f)
            psd_array.append(apsd)
            vpsd_array.append(p)
            psd_freqs.append(f)

        # TODO: Calculate spectrogram

        if make_plot:
            # TODO: Make plots, then reset temp arrays of results
            # TODO: Decide about trace plot, maybe downsample to 5Hz before plotting?

            # PSD plots
            psd_v_plot = os.path.join(outdir,
                                      'psd_vel_{0}_{1}_to_{2}.png'.format(this_channel.id,
                                                                          plot_start.datetime.strftime('%Y-%m-%d'),
                                                                          (plot_end-1).datetime.strftime('%Y-%m-%d')))
            psd_v_fig, vax = plt.subplots(1, 1)
            for f, v in zip(psd_freqs, vpsd_array):
                vax.plot(f, 10. * np.log10(v), c='0.8', lw=0.5, marker=None)
            vax.set_xscale('log')
            vax.set_xlabel('Frequency (Hz)')
            vax.set_ylabel('Power Spectral Density (dB)')
            psd_v_fig.savefig(psd_v_plot)
            psd_v_plots.append(psd_v_plot)

            psd_a_plot = os.path.join(outdir,
                                      'psd_acc_{0}_{1}_to_{2}.png'.format(this_channel.id,
                                                                          plot_start.datetime.strftime('%Y-%m-%d'),
                                                                          (plot_end-1).datetime.strftime('%Y-%m-%d')))
            psd_a_fig, aax = plt.subplots(1, 1)
            for f, a in zip(psd_freqs, psd_array):
                aax.plot(f, 10. * np.log10(a), c='0.8', lw=0.5, marker=None)
            aax.set_xscale('log')
            plt.grid(True, ls=':')
            aax.set_xlabel('Frequency (Hz)')
            aax.set_ylabel('Power Spectral Density (dB)')
            psd_a_fig.savefig(psd_a_plot)
            psd_a_plots.append(psd_a_plot)

            # TODO: Spectrogram plot

            # TODO: Reset temp arrays
            psd_array, vpsd_array, psd_freqs = [], [], []

            # Update plot_end for next time window
            plot_start = plot_end
            if plot_length is None:
                next_mn = plot_start.month + 1
                if next_mn > 12:
                    plot_end = obspy.UTCDateTime(plot_start.year + 1, next_mn - 12, 1)
                else:
                    plot_end = obspy.UTCDateTime(plot_start.year, next_mn, 1)
            else:
                plot_end = plot_start + (plot_length * 24 * 60 * 60)

    return psd_a_plots, psd_v_plots, spec_plots


def buffered_spectrogram(files, outdir, spec_win, overlap, plot_length=30, ch_id=None):
    """
    Plot spectrogram(s) of seismic data stored in raw data files. File paths in *files* should be listed in
    chronological order. Files must be readable by obspy.read()

    :param files: list of paths for raw data files
    :param outdir: path to output directory
    :param spec_win: spectrogram window in seconds
    :param overlap: window overlap (0-1)
    :param plot_length: length of time period to plot in each output PNG in days, default 30
    :param ch_id: optional channel identifier to specify which channel to plot in multi-channel data files

    :return: path(s) to plot PNG file(s)
    """
    buffer_length = 2   # number of files to keep in memory at a given time, will optimize later
    plot_files = []

    return plot_files
