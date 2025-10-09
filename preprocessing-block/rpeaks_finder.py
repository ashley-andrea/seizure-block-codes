import neurokit2 as nk
import mne
import numpy as np
from typing import Dict, List, Optional
import logging
from ecgdetectors import Detectors
from biosppy.signals import ecg

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_ecg_rpeaks(patient_data: List[Dict], method: str = "pantompkins1985") -> List[Dict]:
    """
    Process ECG data using R-peaks detection for all runs of all patients.

    Args:
        patient_data (List[Dict]): List of patient run data from ECGFileLoader
        method (str): NeuroKit2 method for ECG processing (default: "pantompkins1985")
        We do not need the method if we do not use the neurokit2 function
        
    Returns:
        List[Dict]: List of processed ECG data with Pan-Tompkins results
    """
    processed_data = []
    
    for run_idx, run_data in enumerate(patient_data):
        logger.info(f"Processing {run_data['patient_id']} run {run_data['run_number']}")
        
        try:
            # Get ECG raw data
            ecg_raw = run_data['ecg_data']
            if ecg_raw is None:
                logger.warning(f"No ECG data for {run_data['patient_id']} run {run_data['run_number']}")
                continue
            
            # Extract ECG signal (first channel if multiple)
            ecg_signal = ecg_raw.get_data()[0]  # 1D array
            sampling_rate = ecg_raw.info['sfreq']
            
            logger.info(f"ECG signal shape: {ecg_signal.shape}, sampling rate: {sampling_rate} Hz")
            
            ##### TEST CLEANING #####
            #ecg_signal_clean = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate, method='neurokit')
            #print("cleaning done")

            # Apply Pan-Tompkins algorithm using NeuroKit2
            #signals, info = nk.ecg_peaks(ecg_signal_clean, sampling_rate=sampling_rate, method=method, correct_artifacts=True)

            # Extract R-peaks and other information
            #rpeaks = info["ECG_R_Peaks"]

            # PROVA CON ECGDETECTOR
            #detectors = Detectors(sampling_rate)
            #rpeaks = np.array(detectors.pan_tompkins_detector(ecg_signal))

            # PROVA CON biosppy
            out = ecg.engzee_segmenter(signal=ecg_signal, sampling_rate=sampling_rate)
            rpeaks = out['rpeaks']

            
            # Calculate additional metrics
            rr_intervals = np.diff(rpeaks) / sampling_rate * 1000  # RR intervals in milliseconds
            # np.diff() fa la differenza tra elemtni consecutivi

            
            # Create processed data entry
            processed_entry = {
                # Original data
                'patient_id': run_data['patient_id'],
                'run_number': run_data['run_number'],
                'seizure_events': run_data['seizure_events'],
                'annotations': run_data['annotations'],
                
                # Raw ECG data
                'ecg_signal': ecg_signal,
                'sampling_rate': sampling_rate,
                'duration_seconds': len(ecg_signal) / sampling_rate,
                
                # method specific results
                #'method_signals': signals,
                #'method_info': info,
                'rpeaks': rpeaks,
                'rpeaks_times': rpeaks / sampling_rate,  # R-peaks in seconds
                'rr_intervals_ms': rr_intervals,
                'num_beats': len(rpeaks),
                
                # Processing metadata
                'processing_method': "biosppy",
            }
            
            processed_data.append(processed_entry)
            
            logger.info(f"Successfully processed {run_data['patient_id']} run {run_data['run_number']}: "
                       f"{len(rpeaks)} R-peaks detected")
            
        except Exception as e:
            logger.error(f"Error processing {run_data['patient_id']} run {run_data['run_number']}: {e}")
            
            # Add failed entry with error info
            failed_entry = {
                'patient_id': run_data['patient_id'],
                'run_number': run_data['run_number'],
                'ecg_file': run_data['ecg_file'],
                'seizure_events': run_data['seizure_events'],
                'annotations': run_data['annotations'],
                'processing_method': method,
                'processed_successfully': False,
                'error_message': str(e)
            }
            processed_data.append(failed_entry)

    logger.info(f"Completed R-peaks detection processing for {len(processed_data)} runs")
    return processed_data



def process_all_patients_rpeaks(loader, patient_ids: List[str] = None, method: str = "pantompkins1985") -> Dict[str, List[Dict]]:
    """
    Process ECG data using R-peaks detection for multiple patients.

    Args:
        loader: ECGFileLoader instance
        patient_ids (List[str], optional): List of patient IDs to process. If None, processes all patients.
        method (str): NeuroKit2 method for ECG processing (default: "pantompkins1985")
        
    Returns:
        Dict[str, List[Dict]]: Dictionary with patient IDs as keys and processed run data as values
    """
    all_processed_data = {}
    
    # Get patient list
    if patient_ids is None:
        patient_ids = loader.get_patient_list()
    
    logger.info(f"Processing R-peaks detection for {len(patient_ids)} patients: {patient_ids}")
    
    for patient_id in patient_ids:
        logger.info(f"Loading data for patient {patient_id}")
        
        try:
            # Load patient data
            patient_data = loader.get_ecg_with_annotations(patient_id)

            # Process with R-peaks detection
            processed_runs = process_ecg_rpeaks(patient_data, method=method)
            
            all_processed_data[patient_id] = processed_runs
            
            logger.info(f"Completed processing for patient {patient_id}: {len(processed_runs)} runs")
            
        except Exception as e:
            logger.error(f"Error processing patient {patient_id}: {e}")
            all_processed_data[patient_id] = []
    
    return all_processed_data

