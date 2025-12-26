"""
Electrodermal Activity (EDA) feature extraction for stress detection.

Window-based feature extraction from chest EDA signals.
"""
import numpy as np
import neurokit2 as nk


def extract_eda_features(eda_signal, sampling_rate=700):
    """
    Extract EDA features from a signal window for stress detection.
    
    Designed for window-based analysis (e.g., 60-second windows).
    Extracts both phasic (SCR) and tonic features.
    
    Args:
        eda_signal: Raw EDA signal array (1D, typically 42000 samples for 60s @ 700Hz)
        sampling_rate: Sampling rate in Hz (default: 700 for WESAD chest)
        
    Returns:
        Dictionary of EDA features:
          - scr_peaks_count: Number of SCR peaks detected
          - scr_amplitude_mean: Mean amplitude of SCR peaks
          - scr_amplitude_max: Maximum SCR amplitude
          - tonic_mean: Mean tonic (baseline) level
          - tonic_std: Standard deviation of tonic level
          - phasic_mean: Mean phasic activity
          - phasic_std: Standard deviation of phasic activity
          - eda_mean: Mean raw EDA
          - eda_std: Standard deviation of raw EDA
          - eda_min: Minimum EDA value
          - eda_max: Maximum EDA value
    """
    try:
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
        
        # Extract SCR peak features
        scr_peaks = peak_info['SCR_Peaks']
        scr_peaks_count = len(scr_peaks)
        
        if scr_peaks_count > 0:
            # Get amplitudes at peak locations
            scr_amplitudes = eda_decomposed['EDA_Phasic'].iloc[scr_peaks].values
            scr_amplitude_mean = np.mean(scr_amplitudes)
            scr_amplitude_max = np.max(scr_amplitudes)
        else:
            scr_amplitude_mean = 0.0
            scr_amplitude_max = 0.0
        
        # Extract tonic and phasic statistics
        tonic = eda_decomposed['EDA_Tonic'].values
        phasic = eda_decomposed['EDA_Phasic'].values
        
        return {
            'scr_peaks_count': scr_peaks_count,
            'scr_amplitude_mean': scr_amplitude_mean,
            'scr_amplitude_max': scr_amplitude_max,
            'tonic_mean': np.mean(tonic),
            'tonic_std': np.std(tonic),
            'phasic_mean': np.mean(phasic),
            'phasic_std': np.std(phasic),
            'eda_mean': np.mean(eda_cleaned),
            'eda_std': np.std(eda_cleaned),
            'eda_min': np.min(eda_cleaned),
            'eda_max': np.max(eda_cleaned),
        }
    
    except Exception as e:
        # Return NaN features if extraction fails
        return {
            'scr_peaks_count': np.nan,
            'scr_amplitude_mean': np.nan,
            'scr_amplitude_max': np.nan,
            'tonic_mean': np.nan,
            'tonic_std': np.nan,
            'phasic_mean': np.nan,
            'phasic_std': np.nan,
            'eda_mean': np.nan,
            'eda_std': np.nan,
            'eda_min': np.nan,
            'eda_max': np.nan,
        }


def compute_scr_rate(eda_signal, sampling_rate=700):
    """
    Compute Skin Conductance Response (SCR) rate per minute.
    
    Args:
        eda_signal: Raw EDA signal array
        sampling_rate: Sampling rate in Hz
        
    Returns:
        SCR rate (peaks per minute) as float
    """
    try:
        # Extract features (includes SCR peak count)
        features = extract_eda_features(eda_signal, sampling_rate)
        
        # Calculate duration in minutes
        duration_min = len(eda_signal) / sampling_rate / 60.0
        
        # SCR rate = peaks per minute
        scr_rate = features['scr_peaks_count'] / duration_min if duration_min > 0 else 0.0
        
        return scr_rate
    
    except Exception:
        return np.nan
