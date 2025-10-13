import pyhrv.frequency_domain as fd
import biosppy
import pyhrv
import numpy as np
import warnings
from scipy import signal
from scipy.signal import welch
from scipy.interpolate import Akima1DInterpolator, interp1d
from pyhrv.frequency_domain import _check_freq_bands, _compute_parameters, _plot_psd

warnings.filterwarnings(action="ignore", module="scipy")



# Helper to return list of features for each window
def extract_frequency_features(window_result, bands_of_interest=None):
    """
    Extract frequency features from a window result.
    
    Parameters:
    -----------
    window_result : dict
        Dictionary containing window data and computed features
    bands_of_interest : list, optional
        List of band names to extract: ['vlf', 'lf', 'hf']
        If None, extracts all bands (default)
    
    Returns:
    --------
    dict : Dictionary with extracted features (only for bands of interest)
    """
    params = window_result['features'][0]
    
    # Base features always present
    extracted = {
        'window_start': window_result['start_time_s'],
        'window_end': window_result['end_time_s'],
        'n_peaks': len(window_result['rpeaks_window']),
        'total_power': params['fft_total'],
    }
    
    # Determine which bands to extract
    if bands_of_interest is None:
        bands_to_extract = ['vlf', 'lf', 'hf']
    else:
        bands_to_extract = [b.lower() for b in bands_of_interest]
    
    # Map band names to indices (pyHRV always computes in this order)
    band_indices = {'vlf': 0, 'lf': 1, 'hf': 2}
    
    # Extract features for each band of interest
    for band in bands_to_extract:
        if band in band_indices:
            idx = band_indices[band]
            extracted[f'{band}_abs'] = params['fft_abs'][idx]
            extracted[f'{band}_rel'] = params['fft_rel'][idx]
            extracted[f'{band}_peak'] = params['fft_peak'][idx]
    
    # LF/HF ratio only if both bands are in bands of interest
    if 'lf' in bands_to_extract and 'hf' in bands_to_extract:
        extracted['lf_hf_ratio'] = params['fft_ratio']
    
    return extracted



# Custom implementation of Welch's method with enhanced interpolation and adaptive resolution
def custom_welch_psd(nni=None,
                     rpeaks=None,
                     fbands=None,
                     detrend=True,
                     window='hamming',
                     interpolation='makima',
                     hp_filter=True,
                     hp_cutoff=0.0033,
                     overlap=0.5,
                     override_nperseg=None,
                     override_nfft=None,
                     show=True,
                     show_param=True,
                     legend=True,
                     figsize=None,
                     mode='dev'):
    """
    Computes a Power Spectral Density (PSD) estimation from the NNI series using Welch's method
    with enhanced interpolation options and adaptive resolution based on window length.

    Parameters
    ----------
    nni : array
        NN-Intervals in [ms] or [s]

    rpeaks : array
        R-peak locations in [ms] or [s]

    fbands : dict, optional
        Dictionary with frequency bands (2-element tuples or list)
        Value format: (lower_freq_band_boundary, upper_freq_band_boundary)
        Keys:   'ulf'   Ultra low frequency     (default: none) optional
                'vlf'   Very low frequency      (default: (0.000Hz, 0.04Hz))
                'lf'    Low frequency           (default: (0.04Hz - 0.15Hz))
                'hf'    High frequency          (default: (0.15Hz - 0.4Hz))

        To compute only specific bands, include only those keys in the dictionary.
        Example: fbands={'ulf': None, 'vlf': None, 'lf': (0.04, 0.15), 'hf': (0.15, 0.4)} computes only LF and HF
        Example: fbands={'ulf': None, 'vlf': (0.0, 0.04), 'lf': None, 'hf': None} computes only VLF

    detrend : bool, optional
        If True, detrend NNI series by subtracting the mean NNI (default: True)

    window : scipy window function, optional
        Window function used for PSD estimation (default: 'hamming')

    interpolation : str, optional
        Interpolation method: 'cubic', 'akima', or 'makima' (default: 'makima')
        - 'cubic': scipy's interp1d with cubic splines
        - 'akima': Akima1DInterpolator (C1 continuous)
        - 'makima': Akima1DInterpolator with improved end conditions

    hp_filter : bool, optional
        If True, apply high-pass filter after interpolation (default: True)

    hp_cutoff : float, optional
        High-pass filter cutoff frequency in Hz (default: 0.0033 Hz)

    overlap : float, optional
        Overlap ratio for Welch's method (default: 0.5 for 50% overlap)

    override_nperseg : int, optional
        Override automatic nperseg calculation with custom value (default: None)

    override_nfft : int, optional
        Override automatic nfft calculation with custom value (default: None)

    show : bool, optional
        If true, show PSD plot (default: True)

    show_param : bool, optional
        If true, list all computed PSD parameters next to the plot (default: True)

    legend : bool, optional
        If true, add a legend with frequency bands to the plot (default: True)

    figsize : tuple, optional
        Matplotlib figure size (width, height) (default: None: (12, 4))

    mode : string, optional
        Return mode of the function; available modes:
        'normal'    Returns frequency domain parameters and PSD plot figure
        'dev'       Returns frequency domain parameters, frequency and power arrays, no plot
        'devplot'   Returns frequency domain parameters, frequency/power arrays, and plot

    Returns (biosppy.utils.ReturnTuple Object)
    ------------------------------------------
    results : biosppy.utils.ReturnTuple object
        All results of the Welch's method's PSD estimation

    Returned Parameters & Keys
    --------------------------
    .. Peak frequencies of all frequency bands in [Hz] (key: 'fft_peak')
    .. Absolute powers of all frequency bands in [ms^2] (key: 'fft_abs')
    .. Relative powers of all frequency bands [%] (key: 'fft_rel')
    .. Logarithmic powers of all frequency bands [-] (key: 'fft_log')
    .. Normalized powers of all frequency bands [-] (key: 'fft_norms')
    .. LF/HF ratio [-] (key: 'fft_ratio')
    .. Total power over all frequency bands in [ms^2] (key: 'fft_total')
    .. Interpolation method used (key: 'fft_interpolation')
    .. Resampling frequency (key: 'fft_resampling_frequency')
    .. Spectral window used (key: 'fft_spectral_window')
    .. High-pass filter applied (key: 'fft_hp_filter')
    .. HP filter cutoff frequency (key: 'fft_hp_cutoff')
    """
    # Check input values
    nn = pyhrv.utils.check_input(nni, rpeaks)

    # Verify or set default frequency bands
    fbands = _check_freq_bands(fbands)

    # Resampling frequency
    fs = 4
    
    # Calculate window length in seconds
    window_length_s = np.sum(nn) / 1000.0  # Convert from ms to seconds
    
    # INTERPOLATION
    t = np.cumsum(nn)
    t -= t[0]
    t_interpol = np.arange(t[0], t[-1], 1000./fs)
    
    # Apply selected interpolation method
    if interpolation == 'cubic':
        f_interpol = interp1d(t, nn, 'cubic')
        nn_interpol = f_interpol(t_interpol)
    elif interpolation == 'akima':
        f_interpol = Akima1DInterpolator(t, nn)
        nn_interpol = f_interpol(t_interpol)
    elif interpolation == 'makima':
        # Makima is Akima with improved end behavior (method=1)
        f_interpol = Akima1DInterpolator(t, nn, method='makima')
        nn_interpol = f_interpol(t_interpol)
    else:
        raise ValueError(f"Unknown interpolation method '{interpolation}'. Choose 'cubic', 'akima', or 'makima'.")
    
    # HIGH-PASS FILTERING (optional)
    hp_filter_applied = False
    if hp_filter:
        try:
            # Check if cutoff frequency is valid
            nyquist_freq = fs / 2.0
            if hp_cutoff >= nyquist_freq:
                warnings.warn(
                    f"HP cutoff frequency ({hp_cutoff} Hz) >= Nyquist frequency ({nyquist_freq} Hz). "
                    f"Skipping high-pass filter.",
                    stacklevel=2
                )
            else:
                # Design high-pass Butterworth filter (4th order)
                normalized_cutoff = hp_cutoff / nyquist_freq
                b, a = signal.butter(4, normalized_cutoff, btype='high', analog=False)
                
                # Apply zero-phase filtering
                nn_interpol = signal.filtfilt(b, a, nn_interpol)
                hp_filter_applied = True
        except Exception as e:
            warnings.warn(f"High-pass filter failed: {e}. Proceeding without filtering.", stacklevel=2)
    
    # DETRENDING (optional) 
    if detrend:
        nn_interpol = nn_interpol - np.mean(nn_interpol)
    
    # CALCULATE nperseg AND nfft
    if override_nperseg is not None:
        nperseg = override_nperseg
    else:
        # Adaptive nperseg based on window length
        if window_length_s <= 60:
            nperseg = 256
        elif window_length_s <= 120:
            nperseg = 256
        elif window_length_s <= 180:
            nperseg = 512
        elif window_length_s <= 300:
            nperseg = 1024
        else:
            nperseg = 1024  # windows > 5 min
    
    if override_nfft is not None:
        nfft = override_nfft
    else:
        # nfft is next power of 2 >= nperseg
        nfft = int(2 ** np.ceil(np.log2(nperseg)))
    
    # Calculate noverlap from overlap ratio
    noverlap = int(nperseg * overlap)
    
    # COMPUTE POWER SPECTRAL DENSITY
    frequencies, powers = welch(
        x=nn_interpol,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        scaling='density'
    )

    # STEP 6: PREPARE METADATA
    args = (
        nfft, 
        nperseg, 
        noverlap,
        window, 
        fs, 
        interpolation,
        hp_filter_applied,
        hp_cutoff if hp_filter_applied else None,
        window_length_s
    )
    names = (
        'fft_nfft', 
        'fft_nperseg',
        'fft_noverlap',
        'fft_window', 
        'fft_resampling_frequency', 
        'fft_interpolation',
        'fft_hp_filter',
        'fft_hp_cutoff',
        'fft_window_length_s'
    )
    meta = biosppy.utils.ReturnTuple(args, names)

    # STEP 7: MODE HANDLING
    if mode not in ['normal', 'dev', 'devplot']:
        warnings.warn("Unknown mode '%s'. Will proceed with 'normal' mode." % mode, stacklevel=2)
        mode = 'normal'

    # Normal Mode: Returns parameters and plot, no frequency/power arrays
    if mode == 'normal':
        params, freq_i = _compute_parameters('fft', frequencies, powers, fbands)
        figure = _plot_psd('fft', frequencies, powers, freq_i, params, show, show_param, legend, figsize)
        figure = biosppy.utils.ReturnTuple((figure, ), ('fft_plot', ))
        return pyhrv.utils.join_tuples(params, figure, meta)

    # Dev Mode: Returns parameters and arrays, no plot
    elif mode == 'dev':
        params, _ = _compute_parameters('fft', frequencies, powers, fbands)
        return pyhrv.utils.join_tuples(params, meta), frequencies, np.asarray((powers / 10 ** 6))

    # Devplot Mode: Returns parameters, arrays, and plot
    elif mode == 'devplot':
        params, freq_i = _compute_parameters('fft', frequencies, powers, fbands)
        figure = _plot_psd('fft', frequencies, powers, freq_i, params, show, show_param, legend, figsize)
        figure = biosppy.utils.ReturnTuple((figure, ), ('fft_plot', ))
        return pyhrv.utils.join_tuples(params, figure, meta), frequencies, np.asarray((powers / 10 ** 6))



# Main function to compute frequency features in windows
def compute_frequency_features(rpeaks, 
                               window_minutes=10, 
                               tolerance_ms=1300, 
                               use_custom_welch=False,
                               welch_params=None,
                               bands_of_interest=None,
                               print_features=True):
    """
    Compute frequency features in non-overlapping windows of specified length,
    ensuring no RR interval is broken (±tolerance around boundaries).

    Parameters:
    -----------
    rpeaks : list or np.ndarray
        Array of R-peak timestamps in seconds.
    window_minutes : float
        Desired window length in minutes.
    tolerance_ms : int
        Max allowed deviation (±) around window boundary in ms.
    use_custom_welch : bool, optional
        If True, use custom_welch_psd; if False, use fd.welch_psd (default: False)
    welch_params : dict, optional
        Dictionary of parameters to pass to the selected Welch function.
        
        For DEFAULT welch (fd.welch_psd):
        {
            'fbands': None,  
            'nfft': 2**12,
            'detrend': True,
            'window': 'hamming',
            'show': False,
            'show_param': False,
            'legend': True,
            'figsize': None,
            'mode': 'dev'
        }
        
        For CUSTOM welch (custom_welch_psd):
        {
            'detrend': True,
            'window': 'hamming',
            'interpolation': 'makima',
            'hp_filter': True,
            'hp_cutoff': 0.0033,
            'overlap': 0.5,
            'override_nperseg': None,
            'override_nfft': None,
            'show': False,
            'show_param': False,
            'legend': True,
            'figsize': None,
            'mode': 'dev'
        }
    bands_of_interest : list, optional
        List of bands to extract and display: ['vlf'], ['lf', 'hf'], or ['vlf', 'lf', 'hf']
        If None, all bands are extracted (default: None for all bands)
        Examples:
            ['vlf'] - only VLF features
            ['lf', 'hf'] - only LF and HF features
            ['vlf', 'lf', 'hf'] - all bands (same as None)
    print_features : bool, optional
        If True, print extracted features for each window (default: True)

    Returns:
    --------
    results_list : list of dict
        Each element contains:
        - 'start_time_s': window start time in seconds
        - 'end_time_s': window end time in seconds
        - 'rpeaks_window': array of R-peak timestamps (in seconds) within the window
        - 'features': raw HRV features tuple from Welch function (all bands)
        - 'extracted_features': dict with cleaned/organized feature values (filtered bands)
    """
    # Set default parameters based on which Welch method is used
    if use_custom_welch:
        default_welch_params = {
            'fbands': None,  # Always None - compute all bands
            'detrend': True,
            'window': 'hamming',
            'interpolation': 'makima',
            'hp_filter': True,
            'hp_cutoff': 0.0033,
            'overlap': 0.5,
            'override_nperseg': None,
            'override_nfft': None,
            'show': False,
            'show_param': False,
            'legend': True,
            'figsize': None,
            'mode': 'dev'
        }
    else:
        default_welch_params = {
            'fbands': None,  # Always None - compute all bands
            'nfft': 2**12,
            'detrend': True,
            'window': 'hamming',
            'show': False,
            'show_param': False,
            'legend': True,
            'figsize': None,
            'mode': 'dev'
        }
    
    # Update with user-provided parameters
    if welch_params is not None:
        # Remove 'fbands' if user provided it (we always use None)
        welch_params_clean = {k: v for k, v in welch_params.items() if k != 'fbands'}
        default_welch_params.update(welch_params_clean)
    
    # Force fbands to None to always compute all bands
    default_welch_params['fbands'] = None
    
    rpeaks = np.array(rpeaks)
    timestamps_ms = rpeaks * 1000  # Convert to milliseconds for processing
    total_duration = timestamps_ms[-1]

    window_length_ms = window_minutes * 60 * 1000  # convert to ms
    start_time = 0
    results_list = []

    welch_method = "CUSTOM" if use_custom_welch else "DEFAULT"
    print(f"Processing {window_minutes}-minute windows with tolerance ±{tolerance_ms}ms")
    print(f"Using {welch_method} Welch PSD method")
    
    if bands_of_interest is not None:
        bands_display = [b.upper() for b in bands_of_interest]
        print(f"Extracting and displaying only: {', '.join(bands_display)}")
    else:
        print(f"Extracting and displaying all bands: VLF, LF, HF")
    print()
    
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
            print(f"Window {len(results_list) + 1}: {start_time/1000:.2f}s - {end_time/1000:.2f}s ({len(rpeaks_window)} peaks)")
            
            # Select which Welch function to use (always computes all bands)
            if use_custom_welch:
                features = custom_welch_psd(
                    nni=None,
                    rpeaks=rpeaks_window,
                    **default_welch_params
                )
            else:
                features = fd.welch_psd(
                    nni=None,
                    rpeaks=rpeaks_window,
                    **default_welch_params
                )
            
            # Store window result
            window_result = {
                'start_time_s': start_time / 1000,
                'end_time_s': end_time / 1000,
                'rpeaks_window': rpeaks_window,
                'features': features
            }
            
            # Extract only bands of interest
            extracted = extract_frequency_features(window_result, bands_of_interest=bands_of_interest)
            window_result['extracted_features'] = extracted
            
            # Print features if requested (only for bands of interest)
            if print_features:
                if 'vlf_abs' in extracted:
                    print(f"  VLF: {extracted['vlf_abs']:.2f} ms² ({extracted['vlf_rel']:.2f}%) @ {extracted['vlf_peak']:.4f} Hz")
                if 'lf_abs' in extracted:
                    print(f"  LF:  {extracted['lf_abs']:.2f} ms² ({extracted['lf_rel']:.2f}%) @ {extracted['lf_peak']:.4f} Hz")
                if 'hf_abs' in extracted:
                    print(f"  HF:  {extracted['hf_abs']:.2f} ms² ({extracted['hf_rel']:.2f}%) @ {extracted['hf_peak']:.4f} Hz")
                if 'lf_hf_ratio' in extracted:
                    print(f"  LF/HF Ratio: {extracted['lf_hf_ratio']:.2f}")
                print(f"  Total Power: {extracted['total_power']:.2f} ms²")
                print()
            
            results_list.append(window_result)
        else:
            print(f"Skipping window {start_time/1000:.2f}s - {end_time/1000:.2f}s (only {len(rpeaks_window)} peaks)\n")

        # move to next window
        start_time = end_time

        if start_time + window_length_ms > total_duration:
            break

    print(f"Total windows processed: {len(results_list)}")
    return results_list


'''
## EXAMPLE USAGE:

# ===== Extract and display ALL bands =====
results_all = compute_frequency_features(
    rpeaks=filtered_rpeak_times,
    window_minutes=5,
    use_custom_welch=True
)

# ===== Extract and display ONLY VLF =====
results_vlf = compute_frequency_features(
    rpeaks=filtered_rpeak_times,
    window_minutes=10,
    use_custom_welch=True,
    welch_params={'overlap': 0.75},
    bands_of_interest=['vlf']
)

# ===== Extract and display ONLY LF and HF =====
results_lf_hf = compute_frequency_features(
    rpeaks=filtered_rpeak_times,
    window_minutes=5,
    use_custom_welch=True,
    welch_params={'hp_filter': True, 'hp_cutoff': 0.001},
    bands_of_interest=['lf', 'hf']
)

# ===== Use custom Welch with different settings for different bands =====
# Scenario: VLF over 10-min windows with 75% overlap
results_vlf_10min = compute_frequency_features(
    rpeaks=filtered_rpeak_times,
    window_minutes=10,
    use_custom_welch=True,
    welch_params={'overlap': 0.75, 'hp_cutoff': 0.001},
    bands_of_interest=['vlf']
)

# Scenario: LF & HF over 5-min windows with 50% overlap
results_lf_hf_5min = compute_frequency_features(
    rpeaks=filtered_rpeak_times,
    window_minutes=5,
    use_custom_welch=True,
    welch_params={'overlap': 0.5, 'interpolation': 'akima'},
    bands_of_interest=['lf', 'hf']
)

# ===== Access results - only contains bands of interest =====
for window in results_vlf:
    features = window['extracted_features']
    # Only VLF features will be present
    print(f"VLF Power: {features['vlf_abs']:.2f} ms²")
    # 'lf_abs', 'hf_abs' won't be in the dict

for window in results_lf_hf:
    features = window['extracted_features']
    # Only LF and HF features will be present
    print(f"LF Power: {features['lf_abs']:.2f} ms²")
    print(f"HF Power: {features['hf_abs']:.2f} ms²")
    print(f"LF/HF Ratio: {features['lf_hf_ratio']:.2f}")
    # 'vlf_abs' won't be in the dict
'''


# Sliding window TBD