"""
Heart Rate Variability (HRV) feature extraction.
"""
import numpy as np
import neurokit2 as nk


def extract_hrv_features(ecg_signal, sampling_rate=700):
    """
    Extract HRV features from ECG signal.
    
    Args:
        ecg_signal: Raw ECG signal array
        sampling_rate: Sampling rate in Hz (WESAD chest ECG is 700Hz)
        
    Returns:
        Dictionary of HRV features (time-domain, frequency-domain)
    """
    # Clean ECG signal
    ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
    
    # Find R-peaks
    _, rpeaks = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate)
    
    # Extract HRV features
    hrv_time = nk.hrv_time(rpeaks, sampling_rate=sampling_rate)
    hrv_freq = nk.hrv_frequency(rpeaks, sampling_rate=sampling_rate)
    
    return {
        'time_domain': hrv_time,
        'frequency_domain': hrv_freq
    }


def compute_rmssd(rr_intervals):
    """
    Compute Root Mean Square of Successive Differences (RMSSD).
    
    Args:
        rr_intervals: Array of RR intervals in ms
        
    Returns:
        RMSSD value
    """
    diff_rr = np.diff(rr_intervals)
    return np.sqrt(np.mean(diff_rr ** 2))
