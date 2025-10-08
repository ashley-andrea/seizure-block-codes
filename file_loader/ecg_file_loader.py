import os
import pandas as pd
import mne
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import logging
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import tempfile
from urllib.parse import urlparse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ECGFileLoader:
    """
    A class to load ECG files with corresponding EEG annotations.
    
    This loader maps EEG seizure annotations to ECG data, allowing analysis
    of cardiac patterns during seizure events.
    """
    
    def __init__(self, base_path: str = "/Volumes/Seizury/ds005873", use_s3: bool = False):
        """
        Initialize the ECG file loader.
        
        Args:
            base_path (str): Base path to the dataset directory or S3 bucket path
            use_s3 (bool): If True, use S3 storage; if False, use local storage
        """
        self.use_s3 = use_s3
        if use_s3:
            self.base_path = "s3://seizury-data/ds005873"
            self.s3_client = boto3.client('s3')
            parsed = urlparse(self.base_path)
            self.bucket_name = parsed.netloc
            self.s3_prefix = parsed.path.lstrip('/')
        else:
            self.base_path = Path(base_path)
            self.s3_client = None
            self.bucket_name = None
            self.s3_prefix = None
        self.patients_data = {}
    
    def _list_s3_objects(self, prefix: str) -> List[str]:
        """List objects in S3 with given prefix."""
        try:
            response = self.s3_client.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
            return [obj['Key'] for obj in response.get('Contents', [])]
        except Exception as e:
            logger.error(f"Error listing S3 objects: {e}")
            return []
    
    def _download_s3_file(self, s3_key: str) -> str:
        """Download S3 file to temporary location and return local path."""
        try:
            # Create temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(s3_key)[1])
            self.s3_client.download_fileobj(self.bucket_name, s3_key, temp_file)
            temp_file.close()
            return temp_file.name
        except Exception as e:
            logger.error(f"Error downloading S3 file {s3_key}: {e}")
            return None
        
    def get_patient_list(self) -> List[str]:
        """
        Get list of all available patients in the dataset.
        
        Returns:
            List[str]: List of patient IDs (e.g., ['sub-001', 'sub-002', ...])
        """
        patient_dirs = []
        
        if self.use_s3:
            # List S3 objects with the dataset prefix
            objects = self._list_s3_objects(self.s3_prefix + '/')
            # Extract patient IDs from S3 keys
            for obj_key in objects:
                # Format: ds005873/sub-XXX/ses-01/...
                parts = obj_key.split('/')
                if len(parts) >= 2:
                    patient_part = parts[1]  # sub-XXX
                    if patient_part.startswith('sub-') and patient_part not in patient_dirs:
                        patient_dirs.append(patient_part)
        else:
            # Local file system
            if self.base_path.exists():
                for item in self.base_path.iterdir():
                    if item.is_dir() and item.name.startswith('sub-'):
                        patient_dirs.append(item.name)
        
        return sorted(patient_dirs)
    
    def get_patient_runs(self, patient_id: str) -> Dict[str, List[str]]:
        """
        Get all runs for a specific patient.
        
        Args:
            patient_id (str): Patient ID (e.g., 'sub-001')
            
        Returns:
            Dict[str, List[str]]: Dictionary with 'ecg' and 'eeg' keys containing run files
        """
        runs = {'ecg': [], 'eeg': [], 'annotations': []}
        
        if self.use_s3:
            # S3 paths
            ecg_prefix = f"{self.s3_prefix}/{patient_id}/ses-01/ecg/"
            eeg_prefix = f"{self.s3_prefix}/{patient_id}/ses-01/eeg/"
            
            # Get all objects for this patient
            patient_objects = self._list_s3_objects(f"{self.s3_prefix}/{patient_id}/")
            
            for obj_key in patient_objects:
                if '/ecg/' in obj_key and obj_key.endswith('.edf'):
                    runs['ecg'].append(f"s3://{self.bucket_name}/{obj_key}")
                elif '/eeg/' in obj_key and obj_key.endswith('.edf'):
                    runs['eeg'].append(f"s3://{self.bucket_name}/{obj_key}")
                elif '/eeg/' in obj_key and obj_key.endswith('events.tsv'):
                    runs['annotations'].append(f"s3://{self.bucket_name}/{obj_key}")
        else:
            # Local file system
            patient_path = self.base_path / patient_id / "ses-01"
            
            # Get ECG files
            ecg_path = patient_path / "ecg"
            if ecg_path.exists():
                for file in ecg_path.glob("*.edf"):
                    runs['ecg'].append(str(file))
            
            # Get EEG files and annotations
            eeg_path = patient_path / "eeg"
            if eeg_path.exists():
                for file in eeg_path.glob("*.edf"):
                    runs['eeg'].append(str(file))
                for file in eeg_path.glob("*events.tsv"):
                    runs['annotations'].append(str(file))
                
        return runs
    
    def load_eeg_annotations(self, annotation_file: str) -> pd.DataFrame:
        """
        Load EEG annotations from TSV file.
        
        Args:
            annotation_file (str): Path to the TSV annotation file (local or S3)
            
        Returns:
            pd.DataFrame: DataFrame with annotation data (onset, duration, trial_type, etc.)
        """
        try:
            if self.use_s3 and annotation_file.startswith('s3://'):
                # Download S3 file to temp location
                parsed = urlparse(annotation_file)
                s3_key = parsed.path.lstrip('/')
                temp_file = self._download_s3_file(s3_key)
                if temp_file is None:
                    return pd.DataFrame()
                annotations_df = pd.read_csv(temp_file, sep='\t')
                # Clean up temp file
                os.unlink(temp_file)
            else:
                annotations_df = pd.read_csv(annotation_file, sep='\t')
            
            #logger.info(f"Loaded annotations from {annotation_file}")
            #logger.info(f"Columns: {annotations_df.columns.tolist()}")
            #logger.info(f"Number of annotations: {len(annotations_df)}")
            return annotations_df
        except Exception as e:
            logger.error(f"Error loading annotations from {annotation_file}: {e}")
            return pd.DataFrame()
    
    def load_ecg_data(self, ecg_file: str) -> Optional[mne.io.Raw]:
        """
        Load ECG data from EDF file.
        
        Args:
            ecg_file (str): Path to the ECG EDF file (local or S3)
            
        Returns:
            Optional[mne.io.Raw]: MNE Raw object containing ECG data, None if error
        """
        try:
            if self.use_s3 and ecg_file.startswith('s3://'):
                # Download S3 file to temp location
                parsed = urlparse(ecg_file)
                s3_key = parsed.path.lstrip('/')
                temp_file = self._download_s3_file(s3_key)
                if temp_file is None:
                    return None
                raw_ecg = mne.io.read_raw_edf(temp_file, preload=True, verbose=False)
                # Clean up temp file
                os.unlink(temp_file)
            else:
                raw_ecg = mne.io.read_raw_edf(ecg_file, preload=True, verbose=False)
            
            #logger.info(f"Loaded ECG data from {ecg_file}")
            #logger.info(f"ECG channels: {raw_ecg.ch_names}")
            #logger.info(f"Sampling frequency: {raw_ecg.info['sfreq']} Hz")
            #logger.info(f"Duration: {raw_ecg.times[-1]:.2f} seconds")
            return raw_ecg
        except Exception as e:
            logger.error(f"Error loading ECG data from {ecg_file}: {e}")
            return None
    
    def load_eeg_data(self, eeg_file: str) -> Optional[mne.io.Raw]:
        """
        Load EEG data from EDF file.
        
        Args:
            eeg_file (str): Path to the EEG EDF file (local or S3)
            
        Returns:
            Optional[mne.io.Raw]: MNE Raw object containing EEG data, None if error
        """
        try:
            if self.use_s3 and eeg_file.startswith('s3://'):
                # Download S3 file to temp location
                parsed = urlparse(eeg_file)
                s3_key = parsed.path.lstrip('/')
                temp_file = self._download_s3_file(s3_key)
                if temp_file is None:
                    return None
                raw_eeg = mne.io.read_raw_edf(temp_file, preload=True, verbose=False)
                # Clean up temp file
                os.unlink(temp_file)
            else:
                raw_eeg = mne.io.read_raw_edf(eeg_file, preload=True, verbose=False)
            
            #logger.info(f"Loaded EEG data from {eeg_file}")
            return raw_eeg
        except Exception as e:
            logger.error(f"Error loading EEG data from {eeg_file}: {e}")
            return None
    
    def match_ecg_eeg_runs(self, patient_id: str) -> List[Dict]:
        """
        Match ECG and EEG files for the same runs and load their data with annotations.
        
        Args:
            patient_id (str): Patient ID (e.g., 'sub-001')
            
        Returns:
            List[Dict]: List of dictionaries containing matched ECG/EEG data and annotations
        """
        runs = self.get_patient_runs(patient_id)
        matched_data = []
        
        # Extract run numbers from filenames
        ecg_runs = {}
        eeg_runs = {}
        annotation_runs = {}
        
        # Parse ECG files
        for ecg_file in runs['ecg']:
            filename = Path(ecg_file).name
            if 'run-' in filename:
                run_num = filename.split('run-')[1].split('_')[0]
                ecg_runs[run_num] = ecg_file
        
        # Parse EEG files
        for eeg_file in runs['eeg']:
            filename = Path(eeg_file).name
            if 'run-' in filename:
                run_num = filename.split('run-')[1].split('_')[0]
                eeg_runs[run_num] = eeg_file
        
        # Parse annotation files
        for ann_file in runs['annotations']:
            filename = Path(ann_file).name
            if 'run-' in filename:
                run_num = filename.split('run-')[1].split('_')[0]
                annotation_runs[run_num] = ann_file
        
        # Match runs and load data
        for run_num in sorted(set(ecg_runs.keys()) & set(eeg_runs.keys())):
            logger.info(f"Processing {patient_id} run {run_num}")
            
            # Load ECG data
            ecg_data = self.load_ecg_data(ecg_runs[run_num])
            
            # Load EEG data (for reference/validation)
            eeg_data = self.load_eeg_data(eeg_runs[run_num])
            
            # Load annotations if available
            annotations = pd.DataFrame()
            if run_num in annotation_runs:
                annotations = self.load_eeg_annotations(annotation_runs[run_num])
            
            # Create matched data entry
            matched_entry = {
                'patient_id': patient_id,
                'run_number': run_num,
                'ecg_file': ecg_runs[run_num],
                'eeg_file': eeg_runs[run_num],
                'annotation_file': annotation_runs.get(run_num, None),
                'ecg_data': ecg_data,
                'eeg_data': eeg_data,
                'annotations': annotations,
                'seizure_events': self._extract_seizure_events(annotations) if not annotations.empty else []
            }
            
            matched_data.append(matched_entry)
        
        return matched_data
    
    def _extract_seizure_events(self, annotations_df: pd.DataFrame) -> List[Dict]:
        """
        Extract seizure events from annotations DataFrame.
        This version is adapted for OpenNeuro datasets where the seizure event column is 'eventType'.
        Args:
            annotations_df (pd.DataFrame): DataFrame with annotations
        Returns:
            List[Dict]: List of seizure events with onset, duration, and type
        """
        seizure_events = []
        if annotations_df.empty:
            return seizure_events
        # Accept both 'eventType' and 'trial_type' columns for seizure detection
        event_col = None
        for col in ['eventType', 'trial_type']:
            if col in annotations_df.columns:
                event_col = col
                break
        if event_col is None:
            return seizure_events
        # Look for seizure-related markers in the event column
        seizure_markers = ['seizure', 'sz', 'ictal', 'focal', 'generalized', 'tonic', 'clonic']
        for _, row in annotations_df.iterrows():
            event_type = str(row.get(event_col, '')).lower()
            is_seizure = any(marker in event_type for marker in seizure_markers)
            if is_seizure:
                event = {
                    'onset_time': float(row.get('onset', 0)),
                    'duration': float(row.get('duration', 0)),
                    'type': row.get(event_col, ''),
                    'lateralization': row.get('lateralization', None),
                    'localization': row.get('localization', None),
                    'vigilance': row.get('vigilance', None),
                }
                seizure_events.append(event)
        return seizure_events
    
    def load_all_patients(self) -> Dict[str, List[Dict]]:
        """
        Load data for all patients in the dataset.
        
        Returns:
            Dict[str, List[Dict]]: Dictionary with patient IDs as keys and their matched data as values
        """
        all_patients_data = {}
        patients = self.get_patient_list()
        
        logger.info(f"Found {len(patients)} patients: {patients}")
        
        for patient_id in patients:
            logger.info(f"Loading data for {patient_id}")
            patient_data = self.match_ecg_eeg_runs(patient_id)
            all_patients_data[patient_id] = patient_data
            
        return all_patients_data
    
    def get_ecg_with_annotations(self, patient_id: str, run_number: str = None) -> List[Dict]:
        """
        Get ECG data with mapped EEG annotations for a specific patient.
        
        Args:
            patient_id (str): Patient ID (e.g., 'sub-001')
            run_number (str, optional): Specific run number. If None, returns all runs.
            
        Returns:
            List[Dict]: ECG data with annotations
        """
        patient_data = self.match_ecg_eeg_runs(patient_id)
        
        if run_number:
            return [data for data in patient_data if data['run_number'] == run_number]
        
        return patient_data
    
    def save_processed_data(self, output_dir: str, patient_data: Dict[str, List[Dict]]):
        """
        Save processed data to files for later use.
        
        Args:
            output_dir (str): Directory to save processed data
            patient_data (Dict): Processed patient data
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        for patient_id, runs_data in patient_data.items():
            patient_dir = output_path / patient_id
            patient_dir.mkdir(exist_ok=True)
            
            for run_data in runs_data:
                run_info = {
                    'patient_id': run_data['patient_id'],
                    'run_number': run_data['run_number'],
                    'ecg_file': run_data['ecg_file'],
                    'eeg_file': run_data['eeg_file'],
                    'annotation_file': run_data['annotation_file'],
                    'seizure_events': run_data['seizure_events']
                }
                
                # Save run info as JSON
                run_file = patient_dir / f"run_{run_data['run_number']}_info.json"
                with open(run_file, 'w') as f:
                    json.dump(run_info, f, indent=2)
                
                # Save annotations as CSV if available
                if not run_data['annotations'].empty:
                    ann_file = patient_dir / f"run_{run_data['run_number']}_annotations.csv"
                    run_data['annotations'].to_csv(ann_file, index=False)

# Example usage
if __name__ == "__main__":
    # Initialize the loader
    loader = ECGFileLoader()
    
    # Get list of patients
    patients = loader.get_patient_list()
    print(f"Available patients: {patients}")
    
    # Load data for a specific patient
    if patients:
        patient_id = patients[0]  # Use first patient as example
        ecg_with_annotations = loader.get_ecg_with_annotations(patient_id)
        
        print(f"\nData for {patient_id}:")
        for run_data in ecg_with_annotations:
            print(f"Run {run_data['run_number']}: {len(run_data['seizure_events'])} seizure events")
            for event in run_data['seizure_events']:
                print(f"  - {event['trial_type']} at {event['onset_time']}s for {event['duration']}s")