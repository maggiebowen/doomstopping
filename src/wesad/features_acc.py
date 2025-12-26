"""
Accelerometer (ACC) feature extraction for stress detection.

Window-based feature extraction from ACC signals.
"""
import numpy as np
import pandas as pd


def extract_acc_features(acc_signal, sampling_rate=700, rolling_window_sec=1.0, stillness_threshold=0.01):
    """
    Extract ACC features from a signal window.
    
    Args:
        acc_signal: ACC signal array (N samples x 3 axes)
        sampling_rate: Sampling rate in Hz
        rolling_window_sec: Window size for rolling variance (seconds)
        stillness_threshold: Variance threshold for stillness detection
        
    Returns:
        Dictionary of ACC features:
          - acc_mag_mean: Mean magnitude
          - acc_mag_var: Variance of magnitude
          - still_ratio: Fraction of window spent in 'stillness'
    """
    try:
        # Ensure signal is numpy array
        acc_signal = np.array(acc_signal)
        
        # Calculate magnitude: sqrt(x^2 + y^2 + z^2)
        # Assuming input shape is (N, 3) or similar
        if acc_signal.ndim == 2 and acc_signal.shape[1] == 3:
            acc_mag = np.sqrt(np.sum(acc_signal**2, axis=1))
        elif acc_signal.ndim == 1:
            # Fallback if only magnitude is passed
            acc_mag = acc_signal
        else:
            # If shape is (3, N) transpose or handle
            if acc_signal.shape[0] == 3:
                 acc_mag = np.sqrt(np.sum(acc_signal.T**2, axis=1))
            else:
                 raise ValueError("Invalid ACC signal shape")

        # Mean and Variance of magnitude
        acc_mag_mean = np.mean(acc_mag)
        acc_mag_var = np.var(acc_mag)
        
        # Stillness Ratio
        # Rolling variance of magnitude
        window_samples = int(rolling_window_sec * sampling_rate)
        if window_samples < 1:
            window_samples = 1
            
        # Use pandas for efficient rolling calculation
        mag_series = pd.Series(acc_mag)
        rolling_var = mag_series.rolling(window=window_samples).var()
        
        # Fraction where rolling var < threshold
        # nan values from start of rolling window are ignored (or counted as not still)
        # We drop NaNs to compute ratio over valid computed windows
        valid_rolling_var = rolling_var.dropna()
        if len(valid_rolling_var) > 0:
            still_count = np.sum(valid_rolling_var < stillness_threshold)
            still_ratio = still_count / len(valid_rolling_var)
        else:
            still_ratio = 0.0
            
        return {
            'acc_mag_mean': acc_mag_mean,
            'acc_mag_var': acc_mag_var,
            'still_ratio': still_ratio
        }

    except Exception as e:
        return {
            'acc_mag_mean': np.nan,
            'acc_mag_var': np.nan,
            'still_ratio': np.nan
        }
