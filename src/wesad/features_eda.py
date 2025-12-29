"""
Electrodermal Activity (EDA) feature extraction for stress detection.

Window-based feature extraction from chest EDA signals.
"""
import numpy as np
import neurokit2 as nk


def extract_eda_features(eda_signal, sampling_rate=700):
    """
    Extract EDA features from a signal window for stress detection.
    
    Robustness:
    - Returns 'valid_eda' flag (True/False).
    - If valid=False, all numeric features are 0.0.
    - If valid=True but calculating specific metrics fails (e.g. < 2 peaks), 
      returns 0.0 or a sentinel for those specific features.
    
    Args:
        eda_signal: Raw EDA signal array (1D)
        sampling_rate: Sampling rate in Hz
        
    Returns:
        Dictionary of EDA features + 'valid_eda' flag.
    """
    # Default invalid response
    features = {
        'scr_peaks_count': 0.0,
        'scr_amplitude_mean': 0.0,
        'scr_amplitude_max': 0.0,
        'scr_amplitude_var': 0.0,
        'scr_iei_mean': 0.0,
        'scr_iei_var': 0.0,
        'tonic_mean': 0.0,
        'tonic_std': 0.0,
        'tonic_var': 0.0,
        'tonic_slope': 0.0,
        'phasic_mean': 0.0,
        'phasic_std': 0.0,
        'eda_mean': 0.0,
        'eda_std': 0.0,
        'eda_min': 0.0,
        'eda_max': 0.0,
        'valid_eda': False
    }

    try:
        # Check for Hard Failures: input NaNs, empty, flatline (std=0)
        if len(eda_signal) < sampling_rate:  # Too short (< 1 sec)
            return features
        
        if np.isnan(eda_signal).any():
            return features
            
        if np.std(eda_signal) < 1e-9: # Flatline
             return features

        # --- EDA Processing ---
        
        # Clean EDA signal
        eda_cleaned = nk.eda_clean(eda_signal, sampling_rate=sampling_rate)
        
        # Decompose into phasic (SCR) and tonic components
        eda_decomposed = nk.eda_phasic(eda_cleaned, sampling_rate=sampling_rate)
        
        # Find SCR peaks in phasic component
        peak_signal, peak_info = nk.eda_peaks(
            eda_decomposed['EDA_Phasic'], 
            sampling_rate=sampling_rate,
            method='neurokit'
        )
        
        # --- Feature Calculation ---
        
        # 1. SCR (Phasic) Peaks
        scr_peaks = peak_info['SCR_Peaks']
        scr_peaks_count = len(scr_peaks)
        
        scr_amplitude_mean = 0.0
        scr_amplitude_max = 0.0
        scr_amplitude_var = 0.0
        
        if scr_peaks_count > 0:
            # Amplitudes at peak locations
            scr_amplitudes = eda_decomposed['EDA_Phasic'].iloc[scr_peaks].values
            scr_amplitude_mean = np.mean(scr_amplitudes)
            scr_amplitude_max = np.max(scr_amplitudes)
            if scr_peaks_count > 1:
                scr_amplitude_var = np.var(scr_amplitudes)
            # Else var stays 0.0 (Benign)
            
        # 2. SCR Inter-Event Interval (IEI)
        scr_iei_mean = 0.0
        scr_iei_var = 0.0
        
        if scr_peaks_count > 1:
            peak_times = scr_peaks / sampling_rate
            iei = np.diff(peak_times)
            scr_iei_mean = np.mean(iei)
            scr_iei_var = np.var(iei)
        else:
            # Benign Failure: < 2 peaks
            # Use window length as sentinel for mean IEI to indicate "very sparse"
            window_length_sec = len(eda_signal) / sampling_rate
            scr_iei_mean = window_length_sec
            # Var stays 0.0
            
        # 3. Tonic Features
        tonic = eda_decomposed['EDA_Tonic'].values
        # Slope
        t_seconds = np.arange(len(tonic)) / sampling_rate
        try:
            # Fit 1D line: slope = polyfit(t, y, 1)[0]
            tonic_slope = np.polyfit(t_seconds, tonic, 1)[0]
        except Exception:
            tonic_slope = 0.0 # Benign fallback
            
        # 4. Global Statistics
        phasic = eda_decomposed['EDA_Phasic'].values
        
        features.update({
            'scr_peaks_count': float(scr_peaks_count),
            'scr_amplitude_mean': float(scr_amplitude_mean),
            'scr_amplitude_max': float(scr_amplitude_max),
            'scr_amplitude_var': float(scr_amplitude_var),
            'scr_iei_mean': float(scr_iei_mean),
            'scr_iei_var': float(scr_iei_var),
            
            'tonic_mean': float(np.mean(tonic)),
            'tonic_std': float(np.std(tonic)),
            'tonic_var': float(np.var(tonic)),
            'tonic_slope': float(tonic_slope),
            
            'phasic_mean': float(np.mean(phasic)),
            'phasic_std': float(np.std(phasic)),
            
            'eda_mean': float(np.mean(eda_cleaned)),
            'eda_std': float(np.std(eda_cleaned)),
            'eda_min': float(np.min(eda_cleaned)),
            'eda_max': float(np.max(eda_cleaned)),
            
            'valid_eda': True
        })
        
        return features
    
    except Exception as e:
        # Hard Failure (Detection error, etc)
        # Return default 0.0 dict with valid=False
        return features


def compute_scr_rate(eda_signal, sampling_rate=700):
    """
    Compute Skin Conductance Response (SCR) rate per minute.
    Returns np.nan on failure (legacy helper, mostly unused).
    """
    try:
        if np.isnan(eda_signal).any():
            return np.nan
            
        features = extract_eda_features(eda_signal, sampling_rate)
        if not features['valid_eda']:
            return np.nan
            
        duration_min = len(eda_signal) / sampling_rate / 60.0
        if duration_min > 0:
            return features['scr_peaks_count'] / duration_min
        return 0.0
    
    except Exception:
        return np.nan
