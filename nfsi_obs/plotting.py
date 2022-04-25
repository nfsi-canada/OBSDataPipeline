import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import signal

from .waveform import WaveformPlotting


def trace_plot(trace, outdir, dmin, dmax, qc_config=None):
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
                    if dmin < float(ranges['fail_span'][0]):
                        low_fail = [dmin, float(ranges['fail_span'][0])]
                    if dmax > float(ranges['fail_span'][1]):
                        high_fail = [float(ranges['fail_span'][1]), dmax]
                if 'suspect_span' in ranges:
                    if dmin < float(ranges['suspect_span'][0]):
                        if low_fail is not None:
                            if low_fail[1] < float(ranges['suspect_span'][0]):
                                low_sus = [low_fail[1], float(ranges['suspect_span'][0])]
                        else:
                            low_sus = [dmin, float(ranges['suspect_span'][0])]
                    if dmax > float(ranges['suspect_span'][1]):
                        if high_fail is not None:
                            if high_fail[0] > float(ranges['suspect_span'][1]):
                                high_sus = [float(ranges['suspect_span'][1]), high_fail[0]]
                        else:
                            high_sus = [float(ranges['suspect_span'][1]), dmax]
    if any([low_sus is not None, low_fail is not None, high_sus is not None, high_fail is not None]):
        qc_bars = True

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
    plt.specgram(trace.data, NFFT=npts, Fs=trace.meta.sampling_rate, window=signal.get_window('hamming', npts, False),
                 noverlap=nover, detrend='linear', scale='dB')
    sax.set_yscale('log')
    sax.set_ylim(ymin=1e-3, ymax=trace.meta.sampling_rate / 2)
    sax.set_ylabel('Frequency (Hz)')
    sfig.savefig(spectrogram_plot)

    return spectrogram_plot


def psd_plot(trace, outdir, win_len, overlap):
    """
    Plot PSDs of seismic data (as obspy.core.trace.Trace object)

    :param trace: obspy.core.trace.Trace object
    :param outdir: path to output directory
    :param win_len: window length for each PSD curve
    :param overlap: window overlap (0-1)

    :return: path to plot PNG file
    """
    psd_v_plot = os.path.join(outdir, 'psd_seismic_vel_{0}.png'.format(trace.id))
    psd_a_plot = os.path.join(outdir, 'psd_seismic_acc_{0}.png'.format(trace.id))
    freqs, psds = [], []
    psd_v_fig, vax = plt.subplots(1, 1)
    for sect in trace.slide(win_len, win_len * (1 - overlap)):
        seg_len = pow(2, 17)
        psd, frq = plt.psd(sect.data, NFFT=seg_len, Fs=trace.meta.sampling_rate,
                           window=signal.get_window('hamming', seg_len, False), detrend='linear', color='0.7',
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
