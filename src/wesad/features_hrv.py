"""
Heart Rate Variability (HRV) feature extraction for stress detection.

Window-based feature extraction from ECG signals.
"""
import numpy as np
import neurokit2 as nk
import pandas as pd
from scipy import stats


def extract_hrv_features(ecg_signal, sampling_rate=700):
    """
    Extract HRV features from ECG signal window for stress detection.
    
    Designed for window-based analysis (e.g., 60-second windows).
    Extracts time-domain and frequency-domain HRV metrics.
    
    Args:
        ecg_signal: Raw ECG signal array (1D, typically 42000 samples for 60s @ 700Hz)
        sampling_rate: Sampling rate in Hz (default: 700 for WESAD chest ECG)
        
    Returns:
        Dictionary of flattened HRV features:
          Time-domain:
            - hrv_mean_rr: Mean RR interval (ms)
            - hrv_sdnn: Standard deviation of RR intervals
            - hrv_rmssd: Root mean square of successive differences
            - hrv_pnn50: Percentage of successive RR intervals differing by >50ms
          Frequency-domain:
            - hrv_lf: Low frequency power (0.04-0.15 Hz)
            - hrv_hf: High frequency power (0.15-0.4 Hz)
            - hrv_lf_hf_ratio: LF/HF ratio
          Additional:
            - hr_mean: Mean heart rate (bpm)
    """
    try:
        # Clean ECG signal
        ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
        
        # Find R-peaks
        _, rpeaks_dict = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate)
        
        # Check if we have enough peaks for HRV analysis
        # Need at least 5 R-peaks for meaningful HRV
        rpeaks = rpeaks_dict['ECG_R_Peaks']
        if len(rpeaks) < 5:
            # Not enough peaks - return NaN features
            return _nan_hrv_features()
        
        # Extract time-domain HRV features
        hrv_time = nk.hrv_time(rpeaks, sampling_rate=sampling_rate, show=False)
        
        # Extract frequency-domain HRV features
        # This requires longer windows (>60s ideal), so may fail on shorter windows
        try:
            hrv_freq = nk.hrv_frequency(rpeaks, sampling_rate=sampling_rate, show=False)
        except Exception:
            hrv_freq = None
        
        # Compute mean heart rate from RR intervals
        rr_intervals = np.diff(rpeaks) / sampling_rate * 1000  # Convert to ms
        
        if len(rr_intervals) > 1:
            hr_mean = 60000 / np.mean(rr_intervals)
            
            # IBI CV (Coefficient of Variation)
            ibi_mean = np.mean(rr_intervals)
            ibi_std = np.std(rr_intervals)
            ibi_cv = ibi_std / ibi_mean if ibi_mean > 1e-6 else 0.0
            
            # HR Slope
            # Instantaneous HR at each beat
            inst_hr = 60000 / rr_intervals
            # Times of beats (using the second peak of each pair for timestamp)
            beat_times = rpeaks[1:] / sampling_rate
            try:
                hr_slope = np.polyfit(beat_times, inst_hr, 1)[0]
            except Exception:
                hr_slope = 0.0
                
            # IBI Entropy
            # Histogram with 10 bins
            try:
                counts, _ = np.histogram(rr_intervals, bins=10)
                # Compute entropy
                ibi_entropy = stats.entropy(counts)
            except Exception:
                ibi_entropy = 0.0
                
            # RMSSD Subwindow Variance
            # Split signal into 3 subwindows and compute RMSSD for each
            num_subwindows = 3
            subwin_len = len(ecg_cleaned) // num_subwindows
            subwin_rmssds = []
            
            for i in range(num_subwindows):
                start_idx = i * subwin_len
                end_idx = start_idx + subwin_len
                sub_signal = ecg_cleaned[start_idx:end_idx]
                
                # We need peaks in this subwindow to compute RMSSD
                # We can filter the existing peaks or re-find them. Re-finding might be cleaner 
                # but slower. Let's filter existing peaks.
                sub_peaks = rpeaks[(rpeaks >= start_idx) & (rpeaks < end_idx)]
                if len(sub_peaks) > 1:
                    sub_rr = np.diff(sub_peaks) / sampling_rate * 1000
                    subwin_rmssds.append(compute_rmssd(sub_rr))
            
            if len(subwin_rmssds) >= 2:
                rmssd_subwin_var = np.var(subwin_rmssds)
            else:
                rmssd_subwin_var = 0.0
                
        else:
            hr_mean = np.nan
            ibi_cv = 0.0
            hr_slope = 0.0
            ibi_entropy = 0.0
            rmssd_subwin_var = 0.0
        
        # Flatten outputs into single dict
        features = {
            # Time-domain
            'hrv_mean_rr': hrv_time['HRV_MeanNN'].values[0] if 'HRV_MeanNN' in hrv_time else np.nan,
            'hrv_sdnn': hrv_time['HRV_SDNN'].values[0] if 'HRV_SDNN' in hrv_time else np.nan,
            'hrv_rmssd': hrv_time['HRV_RMSSD'].values[0] if 'HRV_RMSSD' in hrv_time else np.nan,
            'hrv_pnn50': hrv_time['HRV_pNN50'].values[0] if 'HRV_pNN50' in hrv_time else np.nan,
            
            # Heart rate and Regularity
            'hr_mean': hr_mean,
            'hr_slope': hr_slope,
            'ibi_cv': ibi_cv,
            'rmssd_subwin_var': rmssd_subwin_var,
            'ibi_entropy': ibi_entropy,
        }
        
        # Add frequency-domain if available
        if hrv_freq is not None:
            features.update({
                'hrv_lf': hrv_freq['HRV_LF'].values[0] if 'HRV_LF' in hrv_freq else np.nan,
                'hrv_hf': hrv_freq['HRV_HF'].values[0] if 'HRV_HF' in hrv_freq else np.nan,
                'hrv_lf_hf_ratio': hrv_freq['HRV_LFHF'].values[0] if 'HRV_LFHF' in hrv_freq else np.nan,
            })
        else:
            features.update({
                'hrv_lf': np.nan,
                'hrv_hf': np.nan,
                'hrv_lf_hf_ratio': np.nan,
            })
        
        return features
    
    except Exception as e:
        # Return NaN features if extraction fails
        return _nan_hrv_features()


def _nan_hrv_features():
    """Helper to return NaN-filled HRV feature dict when extraction fails."""
    return {
        'hrv_mean_rr': np.nan,
        'hrv_sdnn': np.nan,
        'hrv_rmssd': np.nan,
        'hrv_pnn50': np.nan,
        'hr_mean': np.nan,
        'hr_slope': np.nan,
        'ibi_cv': np.nan,
        'rmssd_subwin_var': np.nan,
        'ibi_entropy': np.nan,
        'hrv_lf': np.nan,
        'hrv_hf': np.nan,
        'hrv_lf_hf_ratio': np.nan,
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
