
import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.abspath('src'))

from wesad.features_eda import extract_eda_features
from wesad.features_hrv import extract_hrv_features
from wesad.features_acc import extract_acc_features

def test_eda():
    print("Testing EDA features...")
    # Generate synthetic EDA: tonic + scr
    fs = 700
    duration = 60
    t = np.arange(duration * fs) / fs
    
    # Tonic: Linear drift
    tonic = 0.1 * t + 5.0
    
    # Phasic: Gaussian pulses
    phasic = np.zeros_like(t)
    peaks = [10, 30, 50]
    for p in peaks:
        phasic += 2.0 * np.exp(-0.5 * ((t - p) / 1.0)**2)
        
    eda = tonic + phasic + np.random.normal(0, 0.01, len(t))
    
    feats = extract_eda_features(eda, fs)
    print("EDA Features:", list(feats.keys()))
    
    expected = ['tonic_var', 'tonic_slope', 'scr_amplitude_var', 'scr_iei_mean', 'scr_iei_var']
    for k in expected:
        assert k in feats, f"Missing {k}"
        print(f"  {k}: {feats[k]}")
    print("EDA Test Passed\n")

def test_hrv():
    print("Testing HRV features...")
    # Synthetic ECG
    fs = 700
    duration = 60
    # Simulate R-peaks
    rr_mean = 0.8  # 800ms
    r_times = np.cumsum(np.random.normal(rr_mean, 0.05, int(duration/rr_mean) + 10))
    r_times = r_times[r_times < duration]
    
    ecg = np.zeros(int(duration * fs))
    # Simple peaks
    for rt in r_times:
        idx = int(rt * fs)
        if idx < len(ecg):
            ecg[idx] = 1.0
            
    # Neurokit expects signal, so this synthetic might be too simple for it to detect peaks?
    # WESAD code currently uses nk.ecg_peaks. NK is robust but expects some morphology.
    # Let's try nk.ecg_simulate if available or build better shape.
    try:
        import neurokit2 as nk
        ecg = nk.ecg_simulate(duration=duration, sampling_rate=fs, heart_rate=75)
    except:
        print("Neurokit simulate not available, using simple peaks (might fail info extraction)")
        pass

    feats = extract_hrv_features(ecg, fs)
    print("HRV Features:", list(feats.keys()))
    
    expected = ['hr_slope', 'ibi_cv', 'rmssd_subwin_var', 'ibi_entropy']
    for k in expected:
         # It refers directly to keys in dict
         # Note: features_hrv returns flattened dict
         assert k in feats, f"Missing {k}"
         print(f"  {k}: {feats[k]}")
    print("HRV Test Passed\n")

def test_acc():
    print("Testing ACC features...")
    fs = 700
    duration = 60
    # Random ACC
    acc = np.random.normal(0, 1, (duration * fs, 3))
    
    feats = extract_acc_features(acc, fs)
    print("ACC Features:", list(feats.keys()))
    
    expected = ['acc_mag_mean', 'acc_mag_var', 'still_ratio']
    for k in expected:
        assert k in feats, f"Missing {k}"
        print(f"  {k}: {feats[k]}")
    print("ACC Test Passed\n")

if __name__ == "__main__":
    try:
        test_eda()
        test_hrv()
        test_acc()
        print("ALL TESTS PASSED")
    except Exception as e:
        print(f"TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
