"""
Accelerometer (ACC) feature extraction for stress detection.

Window-based feature extraction from ACC signals.
"""
import numpy as np
import pandas as pd


def extract_acc_features(acc_signal, sampling_rate=700, rolling_window_sec=1.0, stillness_threshold=0.01):
    """
    Extract ACC features from a signal window.
    
    Robustness:
    - Returns 'valid_acc' flag (True/False).
    - Hard failures (input NaNs, exceptions) -> valid=False, all 0.0.
    - No NaNs returned in features.
    
    Args:
        acc_signal: ACC signal array (N samples x 3 axes)
        sampling_rate: Sampling rate in Hz
        rolling_window_sec: Window size for rolling variance (seconds)
        stillness_threshold: Variance threshold for stillness detection
        
    Returns:
        Dictionary of ACC features + 'valid_acc' flag.
    """
    # Default numeric failure features
    features = {
        'acc_mag_mean': 0.0,
        'acc_mag_var': 0.0,
        'still_ratio': 0.0,
        'valid_acc': False
    }

    try:
        # Input Validation
        acc_signal = np.array(acc_signal)
        
        # Hard Check: input NaNs
        if np.isnan(acc_signal).any():
            return features
        
        # Calculate magnitude: sqrt(x^2 + y^2 + z^2)
        if acc_signal.ndim == 2 and acc_signal.shape[1] == 3:
            acc_mag = np.sqrt(np.sum(acc_signal**2, axis=1))
        elif acc_signal.ndim == 1:
            acc_mag = acc_signal # Already magnitude?
        else:
            if acc_signal.shape[0] == 3:
                 acc_mag = np.sqrt(np.sum(acc_signal.T**2, axis=1))
            else:
                 return features # Hard fail on shape

        # Mean and Variance
        acc_mag_mean = np.mean(acc_mag)
        acc_mag_var = np.var(acc_mag)
        
        # Stillness Ratio
        # Rolling variance
        window_samples = int(rolling_window_sec * sampling_rate)
        if window_samples < 1: window_samples = 1
            
        mag_series = pd.Series(acc_mag)
        rolling_var = mag_series.rolling(window=window_samples).var()
        
        # Drop NaNs from the start of rolling window so we only count valid computed stats
        valid_rolling_var = rolling_var.dropna()
        
        if len(valid_rolling_var) > 0:
            still_count = np.sum(valid_rolling_var < stillness_threshold)
            still_ratio = still_count / len(valid_rolling_var)
        else:
            still_ratio = 0.0
            
        features.update({
            'acc_mag_mean': float(acc_mag_mean),
            'acc_mag_var': float(acc_mag_var),
            'still_ratio': float(still_ratio),
            'valid_acc': True
        })
        return features

    except Exception:
        return features
