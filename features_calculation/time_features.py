import numpy as np
import pyhrv.time_domain as td

def compute_hrv_features(rr_intervals, timestamps_rr, sampling_rate=512, window_minutes=5, tolerance_ms=1300):
    """
    Compute HRV features in non-overlapping windows of specified length,
    ensuring no RR interval is broken (±tolerance around boundaries).

    Parameters:
    -----------
    rr_intervals : list or np.ndarray
        Sequence of RR intervals in milliseconds.
    sampling_rate : int
        Sampling frequency of the ECG signal (Hz).
    window_minutes : float
        Desired window length in minutes.
    tolerance_ms : int
        Max allowed deviation (±) around window boundary in ms.

    Returns:
    --------
    results_list : list of dict
        Each element contains HRV features for one window.
    """
    rr_intervals = np.array(rr_intervals)
    timestamps = np.array(timestamps_rr[1:]) * 1000
    total_duration = timestamps[-1]

    window_length_ms = window_minutes * 60 * 1000  # convert to ms
    start_time = 0
    results_list = []

    print("Processing window")
    while True:
        end_target = start_time + window_length_ms

        # find RR timestamp closest to target within ± tolerance
        diff = np.abs(timestamps - end_target)
        valid_indices = np.where(diff <= tolerance_ms)[0]

        if len(valid_indices) == 0:
            # if no valid cut found within tolerance, stop
            break

        cut_idx = valid_indices[np.argmin(diff[valid_indices])]
        end_time = timestamps[cut_idx]

        # Extract RR intervals that fit entirely within this window
        mask = (timestamps > start_time) & (timestamps <= end_time)
        rr_window = rr_intervals[mask]

        if len(rr_window) > 1:
            features = td.time_domain(rr_window, sampling_rate=sampling_rate, plot=False)
            # Store both features and window times (in seconds)
            results_list.append({
                'start_time_s': start_time / 1000,
                'end_time_s': end_time / 1000,
                'features': features
            })

        # move to next window
        start_time = end_time

        if start_time + window_length_ms > total_duration:
            break

    return results_list


def compute_hrv_features_sliding(rr_intervals, timestamps, sampling_rate=512,
                                 window_minutes=5, step_minutes=1, tolerance_ms=1300):
    """
    Compute HRV features over overlapping 5-min windows (every 1 minute)
    with tolerance-based alignment to nearest RR timestamp.
    """
    rr_intervals = np.array(rr_intervals)
    timestamps = np.array(timestamps[1:]) * 1000  # convert to ms

    window_ms = window_minutes * 60 * 1000
    step_ms = step_minutes * 60 * 1000
    total_duration = timestamps[-1]

    results = []
    current_end_target = window_ms  # first end point (5 min)

    print("Starting sliding HRV computation...")

    while current_end_target <= total_duration:
        diff = np.abs(timestamps - current_end_target)
        valid_indices = np.where(diff <= tolerance_ms)[0]

        if len(valid_indices) == 0:
            print(f"No RR timestamp within tolerance for target end time {current_end_target} ms. Skipping this window.")
            current_end_target += step_ms
            continue

        cut_idx = valid_indices[np.argmin(diff[valid_indices])]
        end_time = timestamps[cut_idx]

        # ---- Align start_time based on the new end_time ----
        start_target = end_time - window_ms
        # Find RR timestamp close to start_target
        diff_start = np.abs(timestamps - start_target)
        valid_start_indices = np.where(diff_start <= tolerance_ms)[0]
        if len(valid_start_indices) > 0:
            start_idx = valid_start_indices[np.argmin(diff_start[valid_start_indices])]
            start_time = timestamps[start_idx]
        else:
            start_time = start_target  # fallback if no close RR timestamp

        # ---- Extract RR intervals inside this window ----
        mask = (timestamps > start_time) & (timestamps <= end_time)
        rr_window = rr_intervals[mask]

        if len(rr_window) > 1:
            features = td.time_domain(rr_window, sampling_rate=sampling_rate, plot=False)
            results.append({
                'minute': end_time / 60000,  # convert to minutes
                'win_start_time_s': start_time / 1000,
                'win_end_time_s': end_time / 1000,
                'features': features
            })

        # ---- Move to next step ----
        current_end_target += step_ms

    print("Finished processing.")
    return results






