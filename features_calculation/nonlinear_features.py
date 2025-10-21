import numpy as np
import pyhrv.nonlinear as nl


def compute_nonlinear_features_sliding(rpeaks, sampling_rate=256,
                                 window_minutes=5, step_minutes=1, show_plot=False, kwargs_poincare = {}, kwargs_sampen = {}, kwargs_dfa= {}, tolerance_ms=1300):
    """
    Compute HRV features over overlapping 5-min windows (every 1 minute)
    with tolerance-based alignment to nearest RR timestamp.
    """
    rpeaks = np.array(rpeaks)
    timestamps = rpeaks * 1000  # Convert to milliseconds for processing
    total_duration = timestamps[-1]

    window_ms = window_minutes * 60 * 1000
    step_ms = step_minutes * 60 * 1000

    results = []
    current_end_target = window_ms  # first end point (5 min)

    print("Starting sliding nonLinear features computation...")

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
        rr_window = rpeaks[mask]

        if len(rr_window) > 1:
            features = nl.nonlinear(rpeaks=rr_window, sampling_rate=sampling_rate, show=show_plot, kwargs_poincare=kwargs_poincare, kwargs_sampen=kwargs_sampen, kwargs_dfa=kwargs_dfa)
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