import numpy as np


def setup_freq_bins(f_limits=None, frequencies=None, smoothing_width_octaves=1, step_octaves=0.125):
    """
    Generate bin edges to be used in psd_period_binning. Identical process and output to
    obspy.signal.spectral_estimation.PPSD._setup_period_binning
    """
    if f_limits is None:
        if frequencies is not None:
            f_limits = [frequencies[1], frequencies[-1]]
        else:
            raise ValueError('Either frequency limits or list of frequencies must be provided.')

    step_factor = 2 ** step_octaves
    smoothing_width_factor = 2 ** smoothing_width_octaves
    # Calculate edges and center of first frequency bin, such that the center frequency is the lower limit specified in function input
    f_left = f_limits[0] / (smoothing_width_factor ** 0.5)
    f_right = f_left * smoothing_width_factor
    f_center = np.sqrt(f_left * f_right)
    # Make lists of bin edges for full frequency range
    f_octaves_left = [f_left]
    f_octaves_right = [f_right]
    f_octaves_center = [f_center]
    while f_center < f_limits[1]:
        f_left *= step_factor
        f_right = f_left * smoothing_width_factor
        f_center = np.sqrt(f_left * f_right)
        # Append to lists
        f_octaves_left.append(f_left)
        f_octaves_right.append(f_right)
        f_octaves_center.append(f_center)

    f_octaves_left = np.array(f_octaves_left)
    f_octaves_right = np.array(f_octaves_right)
    f_octaves_center = np.array(f_octaves_center)
    if frequencies is not None:
        valid = f_octaves_right > frequencies[0]
        valid &= f_octaves_left < frequencies[-1]
        f_octaves_left = f_octaves_left[valid]
        f_octaves_right = f_octaves_right[valid]
        f_octaves_center = f_octaves_center[valid]

    return np.vstack([f_octaves_left,
                         f_octaves_center / (step_factor ** 0.5),
                         f_octaves_center,
                         f_octaves_center * (step_factor ** 0.5),
                         f_octaves_right])


def psd_period_binning_single(psd, freqs, f_bins):
    """
    Calculate smoothed/binned PSD curve (for PPSD-style 2D histogram plots). Input PSD should be as output by
    calc_psds(), i.e. not yet converted to dB, ignoring first entry (f=0). Output PSD is given in dB. Method copied
    from obspy.signal.spectral_estimation.PPSD.__process

    :param psd: input PSD, excluding DC term
    :param freqs: frequency values corresponding to PSD
    :param f_bins: frequency bin information, as returned by setup_freq_bins

    :return: smoothed PSD
    """
    # Convert PSD to dB
    spec = 10 * np.log10(psd)

    # Smooth PSD according to bins setup by setup_freq_bins()
    smoothed_psd = []
    for f_left, f_right, f_center in zip(f_bins[0, :], f_bins[4, :], f_bins[2, :]):
        bits = spec[(f_left <= freqs) & (freqs <= f_right)]
        # If there are no data points within smoothing bin, choose value at closest frequency point
        if (len(bits) < 1) & (f_center > freqs[0]) & (f_center < freqs[-1]):
            val = spec[np.argmin(np.abs(freqs - f_center))]
            smoothed_psd.append(val)
        else:
            smoothed_psd.append(bits.mean())
    smoothed_psd = np.array(smoothed_psd, dtype=np.float32)

    return smoothed_psd


def psd_period_binning_multi(psds, freqs, f_bins):
    """
    Calculate smoothed/binned PSD curves (for PPSD-style 2D histogram plots). Input PSDs should be directly from
    calc_psds(), i.e. not yet converted to dB, ignoring first entry (f=0). All PSDs must share the same frequency axis.
    Output PSDs are given in dB. Calls psd_period_binning_single for each individual PSD.

    :param psds: input PSDs, excluding DC term; 2D array-like
    :param freqs: frequency values corresponding to PSD; 1D array-like
    :param f_bins: frequency bin information, as returned by setup_freq_bins

    :return: smoothed PSDs
    """
    # Smooth PSDs according to bins setup by setup_freq_bins()
    all_smooth = []
    for psd in psds:
        smoothed_psd = psd_period_binning_single(psd, freqs, f_bins)
        all_smooth.append(smoothed_psd)
    all_smooth = np.array(all_smooth, dtype=np.float32)

    return all_smooth


