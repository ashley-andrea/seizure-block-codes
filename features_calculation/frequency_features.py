import pyhrv.frequency_domain as fd
import numpy as np


def compute_frequency_features(rpeaks, window_minutes=10, tolerance_ms=1300, welch_params=None):
    """
    Compute frequency features in non-overlapping windows of specified length,
    ensuring no RR interval is broken (±tolerance around boundaries).

    Parameters:
    -----------
    rpeaks : list or np.ndarray
        Array of R-peak timestamps in seconds.
    sampling_rate : int
        Sampling frequency of the ECG signal (Hz).
    window_minutes : float
        Desired window length in minutes.
    tolerance_ms : int
        Max allowed deviation (±) around window boundary in ms.
    welch_params : dict, optional
        Dictionary of parameters to pass to fd.welch_psd(). 
        Default parameters if not provided:
        {
            'fbands': None,
            'nfft': 2**12,
            'detrend': True,
            'window': 'hamming',
            'show': True,
            'show_param': True,
            'legend': True,
            'figsize': None,
            'mode': 'normal'
        }

    Returns:
    --------
    results_list : list of dict
        Each element contains:
        - 'start_time_s': window start time in seconds
        - 'end_time_s': window end time in seconds
        - 'rpeaks_window': array of R-peak timestamps (in seconds) within the window
        - 'features': HRV features for the window
    """
    # Default Welch PSD parameters
    default_welch_params = {
        'fbands': None,
        'nfft': 2**12,
        'detrend': True,
        'window': 'hamming',
        'show': True,
        'show_param': True,
        'legend': True,
        'figsize': None,
        'mode': 'normal'
    }
    
    # Update with user-provided parameters
    if welch_params is not None:
        default_welch_params.update(welch_params)
    
    rpeaks = np.array(rpeaks)
    timestamps_ms = rpeaks * 1000  # Convert to milliseconds for processing
    total_duration = timestamps_ms[-1]

    window_length_ms = window_minutes * 60 * 1000  # convert to ms
    start_time = 0
    results_list = []

    print(f"Processing {window_minutes}-minute windows with tolerance ±{tolerance_ms}ms")
    
    while True:
        end_target = start_time + window_length_ms

        # find R-peak timestamp closest to target within ± tolerance
        diff = np.abs(timestamps_ms - end_target)
        valid_indices = np.where(diff <= tolerance_ms)[0]

        if len(valid_indices) == 0:
            # if no valid cut found within tolerance, stop
            print(f"No valid cut found at {end_target/1000:.2f}s. Stopping.")
            break

        cut_idx = valid_indices[np.argmin(diff[valid_indices])]
        end_time = timestamps_ms[cut_idx]

        # Extract R-peaks that fit entirely within this window
        mask = (timestamps_ms >= start_time) & (timestamps_ms <= end_time)
        rpeaks_window = rpeaks[mask]

        if len(rpeaks_window) > 2:  # Need at least 3 peaks to compute 2 RR intervals
            print(f"Window: {start_time/1000:.2f}s - {end_time/1000:.2f}s ({len(rpeaks_window)} peaks)")
            
            features = fd.welch_psd(
                nni=None,
                rpeaks=rpeaks_window,
                **default_welch_params
            )
            
            # Store features and window information
            results_list.append({
                'start_time_s': start_time / 1000,
                'end_time_s': end_time / 1000,
                'rpeaks_window': rpeaks_window,
                'features': features
            })
        else:
            print(f"Skipping window {start_time/1000:.2f}s - {end_time/1000:.2f}s (only {len(rpeaks_window)} peaks)")

        # move to next window
        start_time = end_time

        if start_time + window_length_ms > total_duration:
            break

    print(f"Total windows processed: {len(results_list)}")
    return results_list


'''
## EXAMPLE USAGE:

# Default configuration
results = compute_frequency_features(
    rpeaks=new_rpeak_times_s
)

# Custom configuration
custom_welch = {
    'nfft': 2**14,
    'window': 'hann',
    'show': False,
    'detrend': False
}

results = compute_frequency_features(
    rpeaks=new_rpeak_times_s,
    window_minutes=5,
    welch_params=custom_welch
)
'''