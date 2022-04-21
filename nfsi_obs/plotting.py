import matplotlib.pyplot as plt
import os

from .waveform import WaveformPlotting


def trace_plot(trace, outdir, dlims, qc_config=None):
    """
    Make time series plot(s) of an obspy.core.trace.Trace object, raw and corrected (if response information included).

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
                    if dlims[0] < float(ranges['fail_span'][0]):
                        low_fail = [dlims[0], float(ranges['fail_span'][0])]
                    if dlims[1] > float(ranges['fail_span'][1]):
                        high_fail = [float(ranges['fail_span'][1]), dlims[1]]
                if 'suspect_span' in ranges:
                    if dlims[0] < float(ranges['suspect_span'][0]):
                        if low_fail is not None:
                            if low_fail[1] < float(ranges['suspect_span'][0]):
                                low_sus = [low_fail[1], float(ranges['suspect_span'][0])]
                        else:
                            low_sus = [dlims[0], float(ranges['suspect_span'][0])]
                    if dlims[1] > float(ranges['suspect_span'][1]):
                        if high_fail is not None:
                            if high_fail[0] > float(ranges['suspect_span'][1]):
                                high_sus = [float(ranges['suspect_span'][1]), high_fail[0]]
                        else:
                            high_sus = [float(ranges['suspect_span'][1]), dlims[1]]
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
            ax.set_ylim(dlims[0], dlims[1])
        plt.grid(True, ls=':')
        fig.savefig(full_data_plot)
        plt.close(fig)
        return full_data_plot
    else:
        return raw_data_plot
