import neurokit2 as nk
import mne

# ecg_signal: 1D numpy array of ECG data
ecg_signal = "/Users/pietrosaveri/Desktop/Pietro/•StartUps/Seizury/scr/Open_Neuro/data/sub-001/ses-01/ecg/sub-001_ses-01_task-szMonitoring_run-01_ecg.edf"

raw = mne.io.read_raw_edf(ecg_signal, preload=True, verbose=False)


# Get the ECG data as a numpy array (channels x samples)
ecg_data = raw.get_data()

# If there is only one channel, flatten to 1D
ecg_1d = ecg_data[0]
print(ecg_1d.shape)

sampling_rate = raw.info['sfreq']
print(f"Sampling rate: {sampling_rate} Hz")

signals, info = nk.ecg_process(ecg_1d, sampling_rate=sampling_rate, method="pantompkins1985")
rpeaks = info["ECG_R_Peaks"]
print(rpeaks)