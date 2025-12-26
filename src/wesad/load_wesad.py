"""
WESAD dataset loader and preprocessor.
Handles loading both E4 wristband data and chest sensor pickle files.
"""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd


def load_subject_e4(subject_dir):
    """
    Load E4 wristband data for one WESAD subject.
    
    Args:
        subject_dir: Path to subject folder (e.g., .../WESAD/S2/)
        
    Returns:
        Dictionary with E4 signals:
          signals["EDA"], ["BVP"], ["ACC"], ["TEMP"], ["HR"] -> {"time": np.ndarray, "data": np.ndarray, "fs": float, "start": float}
          signals["IBI"] -> pd.DataFrame with columns ["t_rel", "ibi", "time"]
          signals["tags"] -> pd.DataFrame with column ["time"]
          signals["_paths"] -> {"subject_dir": Path, "e4_dir": Path}
    """
    subject_dir = Path(subject_dir).expanduser().resolve()
    if not subject_dir.exists():
        raise FileNotFoundError(f"Subject directory not found: {subject_dir}")

    # Find *_E4_data (case-insensitive)
    e4_candidates = [p for p in subject_dir.iterdir()
                     if p.is_dir() and p.name.lower().endswith("_e4_data")]
    if not e4_candidates:
        contents = [p.name for p in subject_dir.iterdir()]
        raise FileNotFoundError(
            f"No '*_E4_data' folder found inside: {subject_dir}\n"
            f"Contents: {contents}"
        )
    if len(e4_candidates) > 1:
        e4_candidates.sort()
    e4_dir = e4_candidates[0]

    def _load_standard_signal(csv_name, expect_cols=None):
        p = e4_dir / csv_name
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")

        raw = pd.read_csv(p, header=None)
        if raw.shape[0] < 3:
            raise ValueError(f"File too short: {p} (rows={raw.shape[0]})")

        start = float(raw.iloc[0, 0])
        fs = float(raw.iloc[1, 0])
        data = raw.iloc[2:].to_numpy(dtype=float)

        if expect_cols is not None and data.shape[1] != expect_cols:
            raise ValueError(f"{p.name}: expected {expect_cols} columns, got {data.shape[1]}")

        t = start + np.arange(data.shape[0], dtype=float) / fs
        return {"time": t, "data": data, "fs": fs, "start": start}

    signals = {
        "_paths": {"subject_dir": subject_dir, "e4_dir": e4_dir}
    }

    # Standard sampled signals
    signals["EDA"] = _load_standard_signal("EDA.csv", expect_cols=1)
    signals["BVP"] = _load_standard_signal("BVP.csv", expect_cols=1)
    signals["TEMP"] = _load_standard_signal("TEMP.csv", expect_cols=1)
    signals["HR"] = _load_standard_signal("HR.csv", expect_cols=1)
    signals["ACC"] = _load_standard_signal("ACC.csv", expect_cols=3)

    # IBI: first row start unix time; remaining rows are [t_rel, ibi]
    ibi_path = e4_dir / "IBI.csv"
    if not ibi_path.exists():
        raise FileNotFoundError(f"Missing file: {ibi_path}")

    ibi_raw = pd.read_csv(ibi_path, header=None)
    if ibi_raw.shape[0] < 2 or ibi_raw.shape[1] < 2:
        raise ValueError(f"Unexpected IBI.csv shape: {ibi_raw.shape} at {ibi_path}")

    ibi_start = float(ibi_raw.iloc[0, 0])
    ibi = ibi_raw.iloc[1:, :2].copy()
    ibi.columns = ["t_rel", "ibi"]
    ibi["t_rel"] = pd.to_numeric(ibi["t_rel"], errors="coerce")
    ibi["ibi"] = pd.to_numeric(ibi["ibi"], errors="coerce")
    ibi = ibi.dropna().reset_index(drop=True)
    ibi["time"] = ibi_start + ibi["t_rel"].to_numpy(dtype=float)
    signals["IBI"] = ibi

    # tags: each row is an absolute unix timestamp
    tags_path = e4_dir / "tags.csv"
    if not tags_path.exists():
        raise FileNotFoundError(f"Missing file: {tags_path}")

    tags = pd.read_csv(tags_path, header=None, names=["time"])
    tags["time"] = pd.to_numeric(tags["time"], errors="coerce")
    tags = tags.dropna().reset_index(drop=True)
    signals["tags"] = tags

    return signals


def load_subject_data(subject_id, data_dir='data/raw'):
    """
    Load chest sensor data and labels from WESAD pickle file.
    
    Args:
        subject_id: Subject ID (e.g., 'S2', 'S3')
        data_dir: Path to WESAD data directory
        
    Returns:
        Dictionary containing subject data with keys:
          - 'signal': dict with chest sensor data (ECG, EDA, Resp, etc.)
          - 'label': array of labels (sampled at 700Hz)
          - 'subject': subject ID string
          - other metadata fields
    """
    filepath = Path(data_dir) / subject_id / f'{subject_id}.pkl'
    if not filepath.exists():
        raise FileNotFoundError(f"Pickle file not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data


def extract_chest_signals(subject_data):
    """
    Extract chest sensor signals from pickle data into structured format.
    
    Args:
        subject_data: Dictionary from load_subject_data()
        
    Returns:
        Dictionary with chest sensor signals:
          - 'ACC': 3-axis accelerometer (Nx3)
          - 'ECG': electrocardiogram (Nx1)
          - 'EDA': electrodermal activity (Nx1)
          - 'EMG': electromyogram (Nx1)
          - 'Resp': respiration (Nx1)
          - 'Temp': temperature (Nx1)
        All sampled at 700Hz
    """
    if 'signal' not in subject_data:
        raise KeyError("Subject data missing 'signal' key")
    
    chest_data = subject_data['signal']['chest']
    
    # Extract each signal type
    signals = {}
    for key in ['ACC', 'ECG', 'EDA', 'EMG', 'Resp', 'Temp']:
        if key in chest_data:
            signals[key] = np.array(chest_data[key])
    
    return signals


def extract_labels(subject_data):
    """
    Extract stress labels from pickle data.
    
    Args:
        subject_data: Dictionary from load_subject_data()
        
    Returns:
        NumPy array of labels (sampled at 700Hz, aligned with chest sensors)
        Label scheme:
          0 = transition / not-worn / between blocks
          1 = Baseline
          2 = TSST stress
          3 = Amusement
          4 = Meditation
          6 = sRead (self-report/reading)
          7 = fRead (final reading)
    """
    if 'label' not in subject_data:
        raise KeyError("Subject data missing 'label' key")
    
    return np.array(subject_data['label']).astype(int)
