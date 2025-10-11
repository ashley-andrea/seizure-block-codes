
# ====== NEED TO BE FIXED BECAUSE THIS WILL HAPPEN AFTER THE INTERPOLATION ======= #

import numpy as np
from scipy import signal
from scipy.interpolate import interp1d


def apply_highpass_filter(processed_rr, cutoff_freq=0.0033, filter_order=4):
    """
    Apply high-pass filter to RR interval signal to remove low-frequency components.
    
    Returns:
    --------
    dict : Dictionary containing filtered data
        - 'filtered_rr_intervals_ms': High-pass filtered RR intervals
        - 'rpeak_times_s': Corresponding R-peak times
        - 'original_rr_intervals_ms': Original RR intervals for comparison
        - 'filter_info': Information about the applied filter
    """

    rr_intervals_ms = processed_rr['dvc_rr_intervals_ms']
    rpeak_times_s = processed_rr['dvc_rpeak_times_s']
    fs = processed_rr['sampling_rate']

    # Convert inputs to numpy arrays
    rr_intervals = np.array(rr_intervals_ms)
    rpeak_times = np.array(rpeak_times_s)

    
    if len(rr_intervals) < 10:
        raise ValueError("Insufficient data points for filtering (minimum 10 required)")
    
    print(f"Sampling frequency: {fs:.4f} Hz")
    
    # Check if cutoff frequency is valid
    nyquist_freq = fs / 2.0
    if cutoff_freq >= nyquist_freq:
        raise ValueError(f"Cutoff frequency ({cutoff_freq} Hz) must be less than Nyquist frequency ({nyquist_freq:.4f} Hz)")
    
    # Design high-pass Butterworth filter
    normalized_cutoff = cutoff_freq / nyquist_freq
    b, a = signal.butter(filter_order, normalized_cutoff, btype='high', analog=False)
    
    # Apply zero-phase filtering to avoid phase distortion
    filtered_rr = signal.filtfilt(b, a, rr_intervals)
    
    # Prepare filter information
    filter_info = {
        'cutoff_frequency_hz': cutoff_freq,
        'filter_order': filter_order,
        'filter_type': 'Butterworth high-pass',
        'sampling_frequency_hz': fs,
        'nyquist_frequency_hz': nyquist_freq,
        'normalized_cutoff': normalized_cutoff
    }
    
    processed_rr['hp_filtered_rr_intervals_ms'] = filtered_rr
    processed_rr['hp_filter_info'] = filter_info

    return processed_rr