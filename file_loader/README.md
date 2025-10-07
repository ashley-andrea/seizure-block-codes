# ECG File Loader

This module provides a comprehensive solution for loading ECG data with corresponding EEG seizure annotations from the seizure monitoring dataset.

## Overview

The `ECGFileLoader` class automatically:
1. Discovers all patients in the dataset
2. Matches ECG and EEG files for the same recording runs
3. Loads EEG seizure annotations from TSV files
4. Maps seizure timing information to ECG data
5. Returns ECG data with seizure event annotations

## Dataset Structure

The loader expects the following directory structure:
```
/Volumes/Seizury/ds005873/
├── sub-001/
│   └── ses-01/
│       ├── ecg/
│       │   ├── sub-001_ses-01_task-szMonitoring_run-01_ecg.edf
│       │   └── sub-001_ses-01_task-szMonitoring_run-01_ecg.json
│       └── eeg/
│           ├── sub-001_ses-01_task-szMonitoring_run-01_eeg.edf
│           ├── sub-001_ses-01_task-szMonitoring_run-01_eeg.json
│           └── sub-001_ses-01_task-szMonitoring_run-01_events.tsv
├── sub-002/
│   └── ses-01/
│       └── ... (same structure)
└── ...
```

## Installation

Install required dependencies:
```bash
pip install -r requirements.txt
```

Required packages:
- `mne` (for EDF file loading)
- `pandas` (for TSV annotation handling)
- `numpy` (for data processing)

## Usage

### Basic Usage

```python
from ecg_file_loader import ECGFileLoader

# Initialize the loader
loader = ECGFileLoader()

# Get available patients
patients = loader.get_patient_list()
print(f"Available patients: {patients}")

# Load ECG data with annotations for a specific patient
patient_data = loader.get_ecg_with_annotations('sub-001')

# Access the data
for run in patient_data:
    print(f"Run {run['run_number']}: {len(run['seizure_events'])} seizures")
    
    # ECG data (MNE Raw object)
    ecg_data = run['ecg_data']
    
    # Seizure events
    for event in run['seizure_events']:
        print(f"Seizure at {event['onset_time']}s, duration: {event['duration']}s")
```

### Advanced Usage

```python
# Load all patients at once
all_data = loader.load_all_patients()

# Get runs for a specific patient
runs_info = loader.get_patient_runs('sub-001')
print(f"ECG files: {runs_info['ecg']}")
print(f"EEG files: {runs_info['eeg']}")
print(f"Annotation files: {runs_info['annotations']}")

# Load only a specific run
specific_run = loader.get_ecg_with_annotations('sub-001', run_number='01')
```

## Data Structure

Each loaded run returns a dictionary with the following structure:

```python
{
    'patient_id': 'sub-001',
    'run_number': '01',
    'ecg_file': '/path/to/ecg.edf',
    'eeg_file': '/path/to/eeg.edf', 
    'annotation_file': '/path/to/events.tsv',
    'ecg_data': mne.io.Raw,  # MNE Raw object with ECG data
    'eeg_data': mne.io.Raw,  # MNE Raw object with EEG data
    'annotations': pd.DataFrame,  # Raw annotations from TSV
    'seizure_events': [  # Extracted seizure events
        {
            'onset_time': 120.5,
            'duration': 45.2,
            'trial_type': 'focal_seizure',
            'description': 'focal seizure left temporal',
            'value': 'seizure_onset'
        },
        # ... more events
    ]
}
```

## Seizure Event Detection

The loader automatically identifies seizure events by looking for keywords in the annotation files:
- 'seizure', 'sz', 'ictal'
- 'focal', 'generalized'
- 'tonic', 'clonic'

## Example Script

Run the example script to see the loader in action:

```bash
python example_usage.py
```

This will:
1. Load data for the first available patient
2. Display run information and seizure events
3. Show ECG data characteristics

## Error Handling

The loader includes comprehensive error handling:
- Missing files are logged but don't stop processing
- Corrupted EDF files are skipped with error messages
- Empty annotation files are handled gracefully

## Logging

The loader uses Python's logging module to provide detailed information:
- Patient discovery and processing status
- File loading success/failure
- Data characteristics (channels, sampling rates, durations)
- Seizure event extraction results

## Output

For each patient run, you get:
- **ECG data**: Raw ECG signals as MNE Raw objects
- **Seizure annotations**: Precise timing and characteristics of seizure events
- **Metadata**: File paths, patient IDs, run numbers
- **EEG data**: Original EEG data for reference/validation

This allows you to analyze cardiac patterns during seizure events by combining high-quality ECG recordings with precise seizure timing annotations from EEG analysis.