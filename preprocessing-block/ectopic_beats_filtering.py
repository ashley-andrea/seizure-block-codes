import numpy as np

'''
Distribution, Variability and Characteristics (DVC) Method for HRV Preprocessing
--------------------------------------------------------------------------------

We want to preserve the equality between the ordinate of each point
and the difference of its abscissa and the previous abscissa value (RR[i] = t[i] - t[i-1]), 
thus we need filtering that enforces time dependence, replacing interpolation.


- Why starting by deleting the >1.3 RR intervals? What does "deleting" entail in this case?

The DVC algorithm relies on the statistical properties of the recent, valid RR intervals to 
generate new data. If an abnormally long interval were included, it would skew the statistics 
and cause the algorithm to generate unrealistic new beats during the imputation phase. 
Removing these clear artifacts is a necessary cleaning step BEFORE any reconstruction can begin.

"Deleting" means removing both the RR interval value and its corresponding timestamp. 
This action creates a gap in the time series. The algorithm does not shift all subsequent 
timestamps to the left to close the gap. Doing so would destroy the time-dependent nature of the 
signal, which is precisely what the authors criticize in other methods.
The gap is handled by the Data Imputation process, designed to fill it in a physiologically plausible way.


- Why deleting the original RR < 0.3s AND the next value?

If merging the falsely detected short beat (RRi) with its neighbor (RRi+1) creates an impossibly long 
new beat, it implies that RRi+1 is itself likely an artifact (perhaps an unusually long compensatory pause 
that was also miscalculated). It's safer to discard this entire problematic two-beat segment (RRi and RRi+1) 
and treat it as a larger gap of missing data, then passed to the data imputation phase for reconstruction.
'''


## Helper for E10 calculation

def _calculate_e10(rr_intervals, current_index):
    """
    Calculates the mean relative deviation over the last 10 RR intervals.
    This corresponds to E10 in the paper's equations.
    """
    # Ensure we don't go out of bounds on the left
    start_index = max(0, current_index - 10)
    if start_index >= current_index:
        return 0.4 # Return a default high variability if not enough data

    # Get the last 10 intervals (or fewer if at the beginning)
    last_10_rrs = rr_intervals[start_index:current_index]
    
    if len(last_10_rrs) < 2:
        return 0.4 # Not enough data to compute deviation, assume high variability

    # Calculate successive differences and normalize by the previous RR interval
    successive_diffs = np.abs(np.diff(last_10_rrs))
    previous_rrs = last_10_rrs[:-1]
    
    # Avoid division by zero
    valid_indices = previous_rrs > 0
    
    if not np.any(valid_indices):
        return 0.4

    relative_deviations = successive_diffs[valid_indices] / previous_rrs[valid_indices]
    
    return np.mean(relative_deviations)



# ALGORITHM PART 1: ECTOPIC BEAT FILTERING

def filter_ectopic_beats(rr_intervals_ms, rpeak_times_s, min_rr_s=0.3, max_rr_s=1.3):
    """
    Implements the DVC filtering process from the paper.
    - Deletes RR intervals > max_rr_s.
    - Merges or deletes RR intervals < min_rr_s based on physiological conditions.

    Args:
        rr_intervals_ms (np.array): Array of RR intervals in milliseconds.
        rpeak_times_s (np.array): Array of R-peak timestamps in seconds.
        min_rr_s (float): Minimum RR interval duration to consider (default: 0.3s).
        max_rr_s (float): Maximum RR interval duration to consider (default: 1.3s).

    Returns:
        tuple: A tuple containing:
            - filtered_rr_s (np.array): The filtered RR intervals in seconds.
            - filtered_ts_s (np.array): The corresponding filtered timestamps in seconds.
    """
    print("--- Starting Ectopic Beat Filtering ---")
    
    # Convert inputs to seconds and create copies to work with
    rr_s = rr_intervals_ms / 1000.0
    # Note: The paper's logic implies timestamps correspond to the END of the RR interval.
    # rpeak_times_s aligns with this assumption.
    ts_s = np.copy(rpeak_times_s) 
    
    # Delete RR intervals > 1.3s (physiologically unlikely) -> creates gaps for imputation
    long_indices = np.where(rr_s > max_rr_s)[0]
    if len(long_indices) > 0:
        print(f"Found {len(long_indices)} RR intervals > {max_rr_s}s. Deleting them to create gaps.")
        # We delete in reverse to not mess up indices of subsequent elements
        for i in sorted(long_indices, reverse=True):
            rr_s = np.delete(rr_s, i)
            ts_s = np.delete(ts_s, i)

    # Handle RR intervals < 0.3s (likely false peak detections)
    # We iterate backwards through the array to handle deletions and modifications safely.
    i = len(rr_s) - 1
    while i > 0:
        if rr_s[i] < min_rr_s:
            print(f"Found short RR interval ({rr_s[i]:.3f}s) at index {i}. Analyzing merge options...")
            
            # Initialize merge possibilities
            right_merge_possible = True
            left_merge_possible = True
            
            # ATTEMPT RIGHT MERGE FIRST
            # Cannot right-merge the last element
            if i >= len(rr_s) - 1:
                right_merge_possible = False
            else:
                rr_r = rr_s[i] + rr_s[i+1]
                # Test physiological conditions for right merge (Table 1 in the paper)
                cond1_r = min_rr_s < rr_r < max_rr_s
                
                # Check deviation against past and future beats
                # E10 is the avg deviation of the 10 beats BEFORE the merge point
                e10 = _calculate_e10(rr_s, i)
                max_deviation = min(0.4, e10) # Max allowed deviation is 40% or E10
                
                # Deviation with the PREVIOUS beat (RRj-1)
                e_l = abs(rr_r - rr_s[i-1]) / rr_s[i-1] if i > 0 and rr_s[i-1] > 0 else float('inf')
                # Deviation with the FOLLOWING beat (RRj+1)
                # The "following" beat after the merge is at original index i+2
                e_r = abs(rr_s[i+2] - rr_r) / rr_r if i < len(rr_s) - 2 and rr_r > 0 else float('inf')
                
                cond2_r = e_l <= max_deviation
                cond3_r = e_r <= max_deviation
                
                if cond1_r and cond2_r and cond3_r:
                    # Successful Right Merge!
                    print(f"  > Right merge successful. New RR: {rr_r:.3f}s.")
                    rr_s[i+1] = rr_r # Update the next interval
                    # Delete the original short interval and its timestamp
                    rr_s = np.delete(rr_s, i)
                    ts_s = np.delete(ts_s, i)
                    i -= 1 # Continue to the next item
                    continue
                else:
                    right_merge_possible = cond1_r # Still possible if within bounds but high deviation

            # ATTEMPT LEFT MERGE (IF RIGHT FAILED)
            # Cannot left-merge the first element
            if i == 0:
                 left_merge_possible = False
            else:
                rr_l = rr_s[i] + rr_s[i-1]
                # Test physiological conditions for left merge
                cond1_l = min_rr_s < rr_l < max_rr_s
                
                e10 = _calculate_e10(rr_s, i - 1)
                max_deviation = min(0.4, e10)
                
                e_l = abs(rr_l - rr_s[i-2]) / rr_s[i-2] if i > 1 and rr_s[i-2] > 0 else float('inf')
                e_r = abs(rr_s[i+1] - rr_l) / rr_l if i < len(rr_s) - 1 and rr_l > 0 else float('inf')

                cond2_l = e_l <= max_deviation
                cond3_l = e_r <= max_deviation
                
                if cond1_l and cond2_l and cond3_l:
                    # Successful Left Merge!
                    print(f"  > Left merge successful. New RR: {rr_l:.3f}s.")
                    rr_s[i-1] = rr_l # Update the previous interval
                    # Delete the original short interval and its timestamp
                    rr_s = np.delete(rr_s, i)
                    ts_s = np.delete(ts_s, i)
                    i -= 2 # Skip the newly merged interval
                    continue
                else:
                    left_merge_possible = cond1_l
            
            # Case from paper: if both merges are > 1.3s, delete original and NEXT value
            if not right_merge_possible and not left_merge_possible:
                print(f"  > Both merges resulted in RR > {max_rr_s}s. Deleting short RR and the NEXT RR.")
                if i < len(rr_s) - 1:
                    # Delete i+1 first since we are iterating backwards
                    rr_s = np.delete(rr_s, i + 1)
                    ts_s = np.delete(ts_s, i + 1)
                # Delete i
                rr_s = np.delete(rr_s, i)
                ts_s = np.delete(ts_s, i)
                i -= 1
                continue
            
            # Fallback: if no merge happened, it's safest to just delete the artifact
            print("  > No merge met all conditions. Deleting the single short RR interval as an artifact.")
            rr_s = np.delete(rr_s, i)
            ts_s = np.delete(ts_s, i)
            
        i -= 1
        
    print(f"--- Filtering finished. Final series has {len(rr_s)} beats. ---")
    return rr_s, ts_s



# ALGORITHM PART 2: DATA IMPUTATION

def impute_gaps(filtered_rr_s, filtered_ts_s, min_rr_s=0.3, max_rr_s=1.3):
    """
    Identifies and fills gaps in the time series using the DVC's iterative
    Gaussian imputation method.
    
    Args:
        filtered_rr_s (np.array): Filtered RR intervals in seconds.
        filtered_ts_s (np.array): Corresponding filtered timestamps in seconds.
        
    Returns:
        tuple: A tuple containing:
            - imputed_rr_s (np.array): The final gap-filled RR intervals.
            - imputed_ts_s (np.array): The final corresponding timestamps.
    """
    print("\n--- Starting Gap Imputation ---")
    
    rr_final = list(filtered_rr_s)
    ts_final = list(filtered_ts_s)
    
    i = 0
    while i < len(ts_final) - 1:
        # The gap is identified by a large time jump between consecutive timestamps
        gap_duration = ts_final[i+1] - ts_final[i]
        
        # The RR interval representing this gap is at index `i` in the rr_final list
        if gap_duration > 1.3:
            print(f"Found a {gap_duration:.2f}s gap. The RR interval at index {i} will be replaced.")
            
            t_start = ts_final[i]
            t_end = ts_final[i+1]
            
            start_idx = max(0, i - 9)
            last_10_rrs = rr_final[start_idx:i+1] # Use up to the beat before the gap
            
            mu = np.mean(last_10_rrs) if last_10_rrs else 0.8
            sigma = np.std(last_10_rrs) if len(last_10_rrs) > 1 else 0.1

            newly_generated_rrs = []
            newly_generated_ts = []
            current_t = t_end
            
            while (current_t - t_start) > 1.3:
                generated = False
                for attempt in range(5):
                    new_rr = np.random.normal(mu, sigma)
                    if 0.3 < new_rr < 1.3 and (current_t - new_rr) > t_start:
                        new_t = current_t - new_rr
                        newly_generated_rrs.insert(0, new_rr)
                        newly_generated_ts.insert(0, new_t)
                        current_t = new_t
                        generated = True
                        break
                if not generated:
                    newly_generated_rrs, newly_generated_ts = [], []
                    break
            
            if newly_generated_rrs:
                last_rr = current_t - t_start
                if 0.3 < last_rr < 1.3:
                    print(f"  > Replacing 1 large RR with {len(newly_generated_rrs) + 1} new beats.")
                    newly_generated_rrs.insert(0, last_rr)
                    
                    # --- CRITICAL FIX AREA ---
                    # Correctly reassemble both lists
                    
                    # Timestamps: take the start, insert the new timestamps, add the end
                    ts_final = ts_final[:i+1] + newly_generated_ts + ts_final[i+1:]
                    
                    # RR Intervals: take the start, insert the new RRs, add the end.
                    # This replaces the single bad RR at index `i`.
                    rr_final = rr_final[:i] + newly_generated_rrs + rr_final[i+1:]
                    
                    i += len(newly_generated_rrs) -1 # Advance index past new beats
            else:
                print("  > Failed to fill the gap.")
        
        i += 1

    print("--- Gap Imputation finished. ---")
    return np.array(rr_final), np.array(ts_final)



# Outlier Handling Pipeline Wrapper

def process_dvc_for_session(session_data, min_rr_s=0.3, max_rr_s=1.3):
    """
    A wrapper function to apply the full DVC pipeline to a single session dict.

    Keys for the session list are: dict_keys(['patient_id', 'run_number', 'ecg_file', 'seizure_events', 
    'annotations', 'ecg_raw', 'ecg_signal', 'sampling_rate', 'duration_seconds', 'pantompkins_signals', 
    'pantompkins_info', 'rpeaks', 'rpeaks_times', 'rr_intervals_ms', 'heart_rate_mean', 'num_beats', 
    'processing_method', 'processed_successfully'])
    """
    if not session_data.get('processed_successfully', False):
        print(f"Skipping session {session_data['run_number']} for patient {session_data['patient_id']} due to initial processing failure.")
        return session_data
        
    print(f"\n--------------------------------------------------")
    print(f"Processing session {session_data['run_number']} for patient {session_data['patient_id']}")
    print(f"--------------------------------------------------")

    # Extract required data
    rr_intervals_ms = session_data['rr_intervals_ms']
    rpeaks_times = session_data['rpeaks_times']
    
    print(f"Initial state: {len(rr_intervals_ms)} RRs, {len(rpeaks_times)} timestamps.")

    filtered_rr, filtered_ts = filter_ectopic_beats(rr_intervals_ms, rpeaks_times, min_rr_s, max_rr_s)
    imputed_rr, imputed_ts = impute_gaps(filtered_rr, filtered_ts, min_rr_s, max_rr_s)
    
    # --- VERIFICATION STEP ---
    print("\n--- Verifying final output integrity ---")
    is_consistent = len(imputed_rr) == len(imputed_ts) - 1
    print(f"Final state: {len(imputed_rr)} RRs, {len(imputed_ts)} timestamps.")
    print(f"Is structure (n, n+1) preserved? {'Yes' if is_consistent else 'No!'}")

    if is_consistent and len(imputed_rr) > 5:
        # Check a few random intervals to be sure
        check_idx = 5
        calculated_rr = imputed_ts[check_idx+1] - imputed_ts[check_idx]
        stored_rr = imputed_rr[check_idx]
        print(f"Sample check at index {check_idx}:")
        print(f"  Stored RR     = {stored_rr:.4f} s")
        print(f"  Calculated RR = {calculated_rr:.4f} s (from ts[{check_idx+1}] - ts[{check_idx}])")
        if not np.isclose(calculated_rr, stored_rr):
            print("  !! Mismatch found !!")
        else:
            print("  Sample check passed.")
    
    if not is_consistent:
        print("!! WARNING: Final output does not respect the n and n+1 principle. !!")

    session_data['dvc_rr_intervals_s'] = imputed_rr
    session_data['dvc_rr_intervals_ms'] = imputed_rr * 1000
    session_data['dvc_rpeak_times_s'] = imputed_ts
    session_data['dvc_processed'] = True
    
    return session_data


'''
# Main loop
for patient_id, sessions in patients_data.items():
    for i, session in enumerate(sessions):
        
        processed_session = process_dvc_for_session(session)
        patients_data[patient_id][i] = processed_session
'''





