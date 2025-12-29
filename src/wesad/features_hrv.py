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
    
    Robustness:
    - Returns 'valid_hrv' flag (True/False).
    - Hard failures (input NaNs, exception, not enough peaks) -> valid=False, all 0.0.
    - Benign failures -> valid=True, missing metrics are 0.0.
    - All divisions are epsilon-guarded.
    
    Args:
        ecg_signal: Raw ECG signal array (1D)
        sampling_rate: Sampling rate in Hz
        
    Returns:
        Dictionary of HRV features + 'valid_hrv' flag.
    """
    # Default numeric values
    features = {
        'hrv_mean_rr': 0.0,
        'hrv_sdnn': 0.0,
        'hrv_rmssd': 0.0,
        'hrv_pnn50': 0.0,
        'hr_mean': 0.0,
        'hr_slope': 0.0,
        'ibi_cv': 0.0,
        'rmssd_subwin_var': 0.0,
        'ibi_entropy': 0.0,
        'hrv_lf': 0.0,
        'hrv_hf': 0.0,
        'hrv_lf_hf_ratio': 0.0,
        'valid_hrv': False
    }
    
    eps = 1e-9 # Epsilon for safe division

    try:
        # Check Hard Failures: input NaNs, empty
        if len(ecg_signal) < sampling_rate: 
            return features
        if np.isnan(ecg_signal).any():
            return features
        if np.std(ecg_signal) < eps:
            return features

        # Clean ECG signal
        ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate)
        
        # Find R-peaks
        _, rpeaks_dict = nk.ecg_peaks(ecg_cleaned, sampling_rate=sampling_rate)
        rpeaks = rpeaks_dict['ECG_R_Peaks']
        
        # Hard Check: Need minimal peaks to be meaningful
        if len(rpeaks) < 5:
            return features
            
        # Compute RR Intervals (ms)
        rr_intervals = np.diff(rpeaks) / sampling_rate * 1000 
        # Filter artifacts (physiologically impossible RRs, e.g. > 2000ms or < 300ms)
        # Simple bounds check for validity
        valid_rr = rr_intervals[(rr_intervals >= 300) & (rr_intervals <= 2000)]
        
        if len(valid_rr) < 3:
            return features # Not enough valid beats
            
        # --- Time Domain Features ---
        features['hrv_mean_rr'] = float(np.mean(valid_rr))
        features['hrv_sdnn'] = float(np.std(valid_rr))
        
        diff_rr = np.diff(valid_rr)
        features['hrv_rmssd'] = float(np.sqrt(np.mean(diff_rr**2))) if len(diff_rr) > 0 else 0.0
        
        if len(diff_rr) > 0:
            nn50 = np.sum(np.abs(diff_rr) > 50)
            features['hrv_pnn50'] = float(nn50 / (len(diff_rr) + eps)) * 100
        
        # --- Heart Rate & Regularity ---
        if features['hrv_mean_rr'] > eps:
            features['hr_mean'] = float(60000 / features['hrv_mean_rr'])
        
        # IBI CV
        ibi_mean = np.mean(valid_rr)
        ibi_std = np.std(valid_rr)
        features['ibi_cv'] = float(ibi_std / (ibi_mean + eps))
        
        # IBI Entropy
        try:
            counts, _ = np.histogram(valid_rr, bins=10)
            features['ibi_entropy'] = float(stats.entropy(counts + eps))
        except:
            features['ibi_entropy'] = 0.0
            
        # HR Slope
        # Instantaneous HR at each valid beat pair
        # Corresponding times
        try:
            # Re-map valid RRs to times. This is approximate if we filtered beats.
            # Using original RRs for slope is often safer if noise isn't extreme, 
            # but let's use the filtered list for consistency, assuming index linearity
            # or just use start/end of signal.
            # Better: Fit HR vs Time on the *original* peaks that generated these Intervals
            inst_hr = 60000 / (rr_intervals + eps)
            beat_times = rpeaks[1:] / sampling_rate
            # Filter matches
            mask = (rr_intervals >= 300) & (rr_intervals <= 2000)
            if np.sum(mask) > 2:
                features['hr_slope'] = float(np.polyfit(beat_times[mask], inst_hr[mask], 1)[0])
        except:
            features['hr_slope'] = 0.0
            
        # RMSSD Subwindow Variance
        try:
            num_subs = 3
            sub_len = len(ecg_cleaned) // num_subs
            sub_vals = []
            for i in range(num_subs):
                s = i * sub_len
                e = s + sub_len
                # Peaks in this subwindow
                sub_p = rpeaks[(rpeaks >= s) & (rpeaks < e)]
                if len(sub_p) > 2:
                    sub_rr = np.diff(sub_p) / sampling_rate * 1000
                    sub_diff = np.diff(sub_rr)
                    val = np.sqrt(np.mean(sub_diff**2))
                    sub_vals.append(val)
            if len(sub_vals) > 1:
                features['rmssd_subwin_var'] = float(np.var(sub_vals))
        except:
            features['rmssd_subwin_var'] = 0.0

        # --- Frequency Domain ---
        # Neurokit often throws errors on short windows
        try:
            # We pass the R-peaks directly to be robust
            hrv_freq = nk.hrv_frequency(rpeaks, sampling_rate=sampling_rate, show=False)
            if not hrv_freq.empty:
                lf = hrv_freq['HRV_LF'].values[0]
                hf = hrv_freq['HRV_HF'].values[0]
                
                # Check for NaNs output by library
                if not np.isnan(lf): features['hrv_lf'] = float(lf)
                if not np.isnan(hf): features['hrv_hf'] = float(hf)
                
                if not np.isnan(lf) and not np.isnan(hf) and hf > eps:
                    features['hrv_lf_hf_ratio'] = float(lf / hf)
        except:
            pass # Keep 0.0 defaults

        # If we got here, we have valid time-domain features at least
        features['valid_hrv'] = True
        return features

    except Exception:
        # Return default failure dict
        return features

def _nan_hrv_features():
    """Deprecated: Helper to return NaN-filled HRV feature dict."""
    # Kept for backward compatibility if imported elsewhere, but internal logic uses the dict above
    d = {
        'hrv_mean_rr': np.nan, 'hrv_sdnn': np.nan, 'hrv_rmssd': np.nan, 'hrv_pnn50': np.nan,
        'hr_mean': np.nan, 'hr_slope': np.nan, 'ibi_cv': np.nan, 'rmssd_subwin_var': np.nan,
        'ibi_entropy': np.nan, 'hrv_lf': np.nan, 'hrv_hf': np.nan, 'hrv_lf_hf_ratio': np.nan
    }
    return d

def compute_rmssd(rr_intervals):
    """Compute Root Mean Square of Successive Differences (RMSSD)."""
    diff_rr = np.diff(rr_intervals)
    return np.sqrt(np.mean(diff_rr ** 2))
