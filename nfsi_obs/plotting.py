import matplotlib.pyplot as plt
from matplotlib import mlab
import numpy as np
import obspy
import os
from scipy import signal

from .waveform import WaveformPlotting
from .metadata import get_channel_type, update_metadata
from .extenders import cut_trace


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
    sfig, sax = plt.subplots(1, 1, num=1, clear=True)
    plt.specgram(trace.data, NFFT=npts, Fs=trace.meta.sampling_rate, window=signal.get_window('hann', npts, False),
                 noverlap=nover, detrend='linear', scale='dB')
    sax.set_yscale('log')
    sax.set_ylim(ymin=1e-3, ymax=trace.meta.sampling_rate / 2)
    sax.set_ylabel('Frequency (Hz)')
    sfig.savefig(spectrogram_plot)

    return spectrogram_plot


def calc_psds(trace, win_len, overlap, sub_overlap, endtime=None, buffered=False):
    """
    Calculate PSDs of seismic data (as obspy.core.trace.Trace object)

    :param trace: input seismic data, measured as ground velocity
    :type trace: obspy.core.trace.Trace
    :param int win_len: window length for each PSD curve in seconds
    :param float overlap: fractional window overlap (0-1)
    :param float sub_overlap: fractional overlap for sub-windows used in PSD calculation (Welch's average periodogram method)
    :param endtime: end time for calculation window (will analyze windows which include `endtime`), obspy.UTCDateTime
    :param bool buffered: whether the input data is being processed as part of a buffer or not

    :returns: Calculated PSD curves in acceleration and velocity, corresponding frequencies, start of next window (if buffered is True)
    """
    seg_len = pow(2, 17)
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
                            noverlap=sub_overlap*trace.meta.sampling_rate,
                            window=signal.get_window('hann', seg_len, False), detrend='linear')
        freqs.append(frq)
        vel_psds.append(psd)
        midpoint = sect.stats.starttime + (sect.stats.endtime - sect.stats.starttime) / 2
        times.append(midpoint.timestamp)
        if buffered:
            next_win_start = next_win_start + win_len * (1 - overlap)

    if hit_end:
        next_win_start = next_win_start - win_len * (1 - overlap)

    # Convert PSDs to acceleration
    acc_psds = []
    for f, p in zip(freqs, vel_psds):
        apsd = p * (2 * np.pi * f) * (2 * np.pi * f)
        acc_psds.append(apsd)

    if buffered:
        return acc_psds, vel_psds, freqs, times, next_win_start
    else:
        return acc_psds, vel_psds, freqs, times


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

    # Calculate all PSDs
    apsds, vpsds, freqs, times = calc_psds(trace, win_len, overlap, sub_overlap)

    # Plot velocity PSDs
    psd_v_fig, vax = plt.subplots(1, 1, num=1, clear=True)
    for f, v in zip(freqs, vpsds):
        vax.plot(f, 10 * np.log10(v), c='0.7', lw=0.5, marker=None)
    vax.set_xscale('log')
    vax.set_xlabel('Frequency (Hz)')
    vax.set_ylabel('Power Spectral Density (dB)')
    psd_v_fig.savefig(psd_v_plot)

    # Plot acceleration PSDs
    psd_a_fig, aax = plt.subplots(1, 1, num=1, clear=True)
    for f, a in zip(freqs, apsds):
        aax.plot(f, 10 * np.log10(a), c='0.8', lw=0.5, marker=None)
    aax.set_xscale('log')
    plt.grid(True, ls=':')
    aax.set_xlabel('Frequency (Hz)')
    aax.set_ylabel('Power Spectral Density (dB)')
    psd_a_fig.savefig(psd_a_plot)

    return psd_a_plot


def buffer_seismic_data(files, outdir, g_log, net_id='XX', station_info=None, channel_map=None, project_meta=None,
                        psd_win=3600, spec_win=3600, overlap=0.5, psd_over=0.75, plot_length=None, ch_id=None, start=None, end=None, detrend=False):
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

    :return: path(s) to plot PNG file(s)
    """
    report_info = {'order': 100}
    channel_info = None
    if ch_id is not None:
        input_type = get_channel_type(ch_id.split('.')[-1])
        report_info['channelType'] = input_type
        if input_type != 'seismic':
            g_log.warn('Data buffering not yet implemented for non-seismic channel {0} of type {1}'.format(ch_id, input_type))
            return report_info

    buffer_length = 2   # number of files to keep in memory at a given time, will optimize later
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
    spec_array = None
    spec_times = None
    psd_temp_results = {
        'psd_array': None,
        'vpsd_array': None,
        'psd_freqs': None,
        'psd_times': None
    }
    #psd_array, vpsd_array, psd_freqs = [], [], []
    while i < len(files):
        # Read data into buffer, keep copy of last file read
        buffer = obspy.Stream()
        files_in_buffer = 0

        # Read next data file if nothing saved from previous loop iteration
        if latest_data is None:
            latest_data = obspy.read(files[i])
            i += 1
        # Trim data to window of interest
        latest_data.trim(start, end, nearest_sample=False)
        # Check for empty stream (no data in file, or no data within window of interest)
        if len(latest_data) < 1:
            latest_data = None
            continue

        # Set start and end of current plot time window if not already set (should only need for first loop iteration)
        if plot_end is None:
            if plot_length is None:
                # default behaviour plots a single calendar month in each image
                plot_start, plot_end = month_start_end(latest_data[0].stats.starttime)
            else:
                # time window in days specified by plot_length
                plot_start = obspy.UTCDateTime(latest_data[0].stats.starttime.year, latest_data[0].stats.starttime.month, latest_data[0].stats.starttime.day)
                plot_end = plot_start + (plot_length * 24 * 60 * 60)

        # Add trace data from first data file to buffer
        for tr in latest_data:
            if ch_id is not None:
                if tr.id != ch_id:
                    continue  # ignore all other channels if *ch_id* is specified
            buffer.append(tr)  # have to add one trace at a time to existing Stream object
            if (tr.stats.starttime > last_start) or (last_start is None):
                last_start = tr.stats.starttime
        files_in_buffer += 1
        buffer.merge()

        mid_plot = True
        if buffer.count() > 0:
            mid_plot = (buffer[0].stats.endtime < plot_end)

        # Fill remaining space in buffer with new files, keeping a copy of the last one read as "latest_data"
        while (files_in_buffer < buffer_length) and mid_plot and (i < len(files)):
            latest_data = obspy.read(files[i])
            i += 1
            latest_data.trim(start, end, nearest_sample=False)  # trim to time window of interest
            if len(latest_data) > 0:
                for tr in latest_data:
                    if ch_id is not None:
                        if tr.id != ch_id:
                            continue    # ignore all other channels if *ch_id* is specified
                    buffer.append(tr)   # have to add one trace at a time to existing Stream object
                    if (tr.stats.starttime > last_start) or (last_start is None):
                        last_start = tr.stats.starttime
                files_in_buffer += 1
                buffer.merge()
            if buffer.count() > 0:
                mid_plot = (buffer[0].stats.endtime < plot_end)     # Complains if there are no traces in the buffer (e.g. no matching channel IDs from latest data)

        if ch_id is None:
            ch_id = buffer[0].id    # channel ID before correction (use to ensure same channel analyzed throughout)

        # Update metadata from other sources
        buffer = update_metadata(buffer, net_id, g_log, station_info, channel_map, project_meta)

        if len(buffer) > 1:
            g_log.warn('Multiple channels present in data files, analyzing first one only: {0}'.format(buffer[0].id))
        this_channel = buffer[0]    # Only look at first channel in files

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
            return report_info

        if channel_info is None:
            if project_meta is not None:
                try:
                    channel_info = list(filter(lambda ch: ch['channel_id'] == this_channel.id.split('.')[-1], project_meta['channels']))[0]
                except (KeyError, IndexError):
                    pass

        if channel_info is not None:
            if 'hide' in channel_info:
                if channel_info['hide']:
                    g_log.info('Channel {0} hidden from report. Skipping analysis.'.format(this_channel.id))
                    return report_info
            if 'order' in channel_info:
                report_info['order'] = int(channel_info['order'])
            if 'qc_config' in channel_info:
                g_log.warn('QARTOD QC checks not yet implemented for buffered data, config ignored')

        # Gap test
        # TODO: Would be nice if this could account for overlap between consecutive buffer sections to not duplicate gap info...
        gaps = this_channel.split().get_gaps()
        if len(gaps) > 0:
            all_gaps.extend(gaps)
            g_log.info('Found {0} gap(s) or overlap(s) in recorded data'.format(len(gaps)))
            this_channel.split().print_gaps()

        # Apply channel sensitivity and remove linear trend, if applicable
        if hasattr(this_channel.meta, 'response'):
            this_channel.remove_sensitivity()
        if detrend:
            this_channel.detrend('linear')

        if last_start is not None:
            # Windows start from midnight UTC on the first day of data collection
            start_of_day = obspy.UTCDateTime(this_channel.stats.starttime.year, this_channel.stats.starttime.month, this_channel.stats.starttime.day)
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
                plot_start = obspy.UTCDateTime(this_channel.stats.starttime.year, this_channel.stats.starttime.month, this_channel.stats.starttime.day)
                plot_end = plot_start + (plot_length * 24 * 60 * 60)

        if (this_channel.stats.endtime > plot_end) or (i == len(files)):
            # data in buffer spans a plot breakpoint, or last file read => make plots this pass
            make_plot = True

        # Calculate PSDs and save to running lists
        psd_start = first_psd_start
        new_data = cut_trace(this_channel, psd_start, None, nearest_sample=True, pad=True)
        apsds, vpsds, freqs, times, next_psd_start = calc_psds(new_data, psd_win, overlap, psd_over, endtime=plot_end, buffered=True)

        for running, current in zip(['psd_array', 'vpsd_array', 'psd_freqs'], [apsds, vpsds, freqs]):
            if psd_temp_results[running] is None:
                psd_temp_results[running] = current
            else:
                psd_temp_results[running] = np.concatenate((psd_temp_results[running], current), axis=0)
        if psd_temp_results['psd_times'] is None:
            psd_temp_results['psd_times'] = times
        else:
            psd_temp_results['psd_times'] = np.concatenate((psd_temp_results['psd_times'], times), axis=None)
        #psd_array = np.concatenate((psd_array, apsds), axis=0)
        #vpsd_array = np.concatenate((vpsd_array, vpsds), axis=0)
        #psd_freqs = np.concatenate((psd_freqs, freqs), axis=0)

        # Calculate spectrogram
        npts = int(spec_win * this_channel.stats.sampling_rate)
        nover = int(overlap * npts)
        analysis_end = min(this_channel.stats.endtime, plot_end)
        num_win = np.floor((analysis_end - first_spec_start - spec_win) / (spec_win * (1 - overlap)))
        last_spec_start = first_spec_start + num_win * spec_win * (1 - overlap)
        spec_data = this_channel.slice(first_spec_start, last_spec_start + spec_win)
        spec, sfrq, t = mlab.specgram(spec_data.data, NFFT=npts, Fs=this_channel.stats.sampling_rate,
                                      window=signal.get_window('hann', npts, False), noverlap=nover, detrend='linear')
        st = np.array([(spec_data.stats.starttime + tm).timestamp for tm in t])
        if spec_array is None:
            spec_array = np.array(spec)
            spec_times = np.array(st)
        else:
            spec_array = np.concatenate((spec_array, spec), axis=1)
            spec_times = np.concatenate((spec_times, st), axis=None)
        next_spec_start = last_spec_start + spec_win * (1 - overlap)    # start time for next iteration of buffer loop

        if make_plot:
            # TODO: Decide about trace plot, maybe downsample to 5Hz before plotting?

            # PSD plots
            psd_v_plot = os.path.join(outdir,
                                      'psd_vel_{0}_{1}_to_{2}.png'.format(this_channel.id,
                                                                          plot_start.datetime.strftime('%Y-%m-%d'),
                                                                          (plot_end-1).datetime.strftime('%Y-%m-%d')))
            psd_v_fig, vax = plt.subplots(1, 1, num=1, clear=True, figsize=(8, 5))
            for f, v in zip(psd_temp_results['psd_freqs'], psd_temp_results['vpsd_array']):
                vax.plot(f, 10. * np.log10(v), c='0.8', lw=0.5, marker=None)
            vax.set_xscale('log')
            vax.set_xlabel('Frequency (Hz)')
            vax.set_ylabel('Power Spectral Density (dB)')
            plt.tight_layout()
            psd_v_fig.savefig(psd_v_plot)
            psd_v_plots.append(psd_v_plot)

            psd_a_plot = os.path.join(outdir,
                                      'psd_acc_{0}_{1}_to_{2}.png'.format(this_channel.id,
                                                                          plot_start.datetime.strftime('%Y-%m-%d'),
                                                                          (plot_end-1).datetime.strftime('%Y-%m-%d')))
            psd_a_fig, aax = plt.subplots(1, 1, num=1, clear=True, figsize=(8, 5))
            for f, a in zip(psd_temp_results['psd_freqs'], psd_temp_results['psd_array']):
                aax.plot(f, 10. * np.log10(a), c='0.8', lw=0.5, marker=None)
            aax.set_xscale('log')
            plt.grid(True, ls=':')
            aax.set_xlabel('Frequency (Hz)')
            aax.set_ylabel('Power Spectral Density (dB)')
            plt.tight_layout()
            psd_a_fig.savefig(psd_a_plot)
            psd_a_plots.append({
                'image': psd_a_plot,
                'start': plot_start.strftime('%Y-%m-%d'),
                'end': (plot_end - 1).strftime('%Y-%m-%d')
            })

            # Spectrogram plot from PSDs
            spec_psd_plot = os.path.join(outdir,
                                            'spec_psd_{0}_{1}_to_{2}.png'.format(this_channel.id,
                                                                             plot_start.datetime.strftime('%Y-%m-%d'),
                                                                             (plot_end-1).datetime.strftime('%Y-%m-%d')))
            spec_fig, sax = plt.subplots(1, 1, num=1, clear=True, figsize=(8, 5))
            spec_psds = 10. * np.log10(np.transpose(psd_temp_results['vpsd_array']))
            spec_psds = np.flipud(spec_psds)

            tm_x_ticks, tm_x_ticklabels = [plot_start.timestamp], [plot_start.strftime('%Y-%m-%d')]
            dt = plot_start + 24 * 60 * 60
            while dt < plot_end:
                tm_x_ticks.append(dt.timestamp)
                tm_x_ticklabels.append(dt.strftime('%Y-%m-%d'))
                dt += 24 * 60 * 60
            tm_x_ticks.append(plot_end.timestamp)
            tm_x_ticklabels.append(plot_end.strftime('%Y-%m-%d'))

            pad_xextent = (npts - nover) / this_channel.stats.sampling_rate / 2
            xextent = np.min(psd_temp_results['psd_times']) - pad_xextent, np.max(psd_temp_results['psd_times']) + pad_xextent
            xmin, xmax = xextent
            extent = xmin, xmax, sfrq[0], sfrq[-1]

            im = sax.imshow(spec_psds, cmap=None, extent=extent, vmin=None, vmax=None, origin='upper')
            sax.axis('auto')
            sax._sci(im)
            sax.set_yscale('log')
            sax.set_ylim(ymin=8e-3, ymax=this_channel.stats.sampling_rate/2)
            sax.set_ylabel('Frequency (Hz)')
            # Set appropriate x-ticks for time span (also changes x-lim)
            sax.set_xticks(tm_x_ticks, tm_x_ticklabels, horizontalalignment='right')
            sax.tick_params(axis='x', rotation=40)
            plt.tight_layout()
            spec_fig.savefig(spec_psd_plot)

            # Reset temp arrays for PSDs
            #psd_array, vpsd_array, psd_freqs = [], [], []
            psd_temp_results = {
                'psd_array': None,
                'vpsd_array': None,
                'psd_freqs': None,
                'psd_times': None
            }

            # Spectrogram plot
            spectrogram_plot = os.path.join(outdir,
                                            'spec_{0}_{1}_to_{2}.png'.format(this_channel.id,
                                                                             plot_start.datetime.strftime('%Y-%m-%d'),
                                                                             (plot_end-1).datetime.strftime('%Y-%m-%d')))
            spec_fig, sax = plt.subplots(1, 1, num=1, clear=True, figsize=(8, 5))
            spec_array = 10. * np.log10(spec_array)
            spec_array = np.flipud(spec_array)

            pad_xextent = (npts - nover) / this_channel.stats.sampling_rate / 2
            xextent = np.min(spec_times) - pad_xextent, np.max(spec_times) + pad_xextent
            xmin, xmax = xextent
            extent = xmin, xmax, sfrq[0], sfrq[-1]

            im = sax.imshow(spec_array, cmap=None, extent=extent, vmin=None, vmax=None, origin='upper')
            sax.axis('auto')
            sax._sci(im)
            sax.set_yscale('log')
            sax.set_ylim(ymin=8e-3, ymax=this_channel.stats.sampling_rate/2)
            sax.set_ylabel('Frequency (Hz)')
            # Set appropriate x-ticks for time span (also changes x-lim)
            sax.set_xticks(tm_x_ticks, tm_x_ticklabels, horizontalalignment='right')
            sax.tick_params(axis='x', rotation=40)
            plt.tight_layout()
            spec_fig.savefig(spectrogram_plot)
            spec_plots.append({
                'image': spectrogram_plot,
                'start': plot_start.strftime('%Y-%m-%d'),
                'end': (plot_end - 1).strftime('%Y-%m-%d')
            })

            # Reset temp arrays for spectrogram
            spec_array, spec_times = None, None

            make_plot = False
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

    report_info['psdLoc'] = psd_a_plots
    report_info['specLoc'] = spec_plots
    return report_info
