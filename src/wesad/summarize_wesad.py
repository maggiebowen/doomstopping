"""
Batch summarizer for WESAD labels across all subjects.
Outputs two CSVs to data/processed:
  - wesad_label_counts.csv
  - wesad_label_segments.csv
Run: python3 scripts/summarize_wesad.py
"""
from pathlib import Path
import pickle
import numpy as np
import csv


RAW_ROOT = Path("data/raw")
OUTPUT_DIR = Path("data/processed")
LABEL_FS = 700.0  # labels aligned to chest sampling rate


def load_labels(subject_dir: Path) -> np.ndarray:
    pkl_path = subject_dir / f"{subject_dir.name}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"Missing pkl for subject {subject_dir.name}: {pkl_path}")
    with pkl_path.open("rb") as f:
        obj = pickle.load(f, encoding="latin1")
    return np.asarray(obj["label"]).astype(int)


def run_length_segments(labels: np.ndarray, fs: float):
    segments = []
    start = 0
    cur = int(labels[0])
    for idx, val in enumerate(labels[1:], 1):
        v = int(val)
        if v != cur:
            seg_len = idx - start
            segments.append(
                {
                    "label": cur,
                    "start_idx": start,
                    "len_samples": seg_len,
                    "dur_s": seg_len / fs,
                }
            )
            cur, start = v, idx
    seg_len = len(labels) - start
    segments.append(
        {
            "label": cur,
            "start_idx": start,
            "len_samples": seg_len,
            "dur_s": seg_len / fs,
        }
    )
    return segments


def main():
    subjects = sorted(p for p in RAW_ROOT.iterdir() if p.is_dir() and p.name.startswith("S"))
    if not subjects:
        raise SystemExit(f"No subject folders found in {RAW_ROOT}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    count_rows = []
    segment_rows = []

    for subj_dir in subjects:
        try:
            labels = load_labels(subj_dir)
        except FileNotFoundError as e:
            print(f"Skipping {subj_dir.name}: {e}")
            continue

        vals, counts = np.unique(labels, return_counts=True)
        for v, c in zip(vals.tolist(), counts.tolist()):
            count_rows.append(
                {
                    "subject": subj_dir.name,
                    "label": int(v),
                    "count": int(c),
                    "dur_s": c / LABEL_FS,
                }
            )

        for seg in run_length_segments(labels, LABEL_FS):
            segment_rows.append({"subject": subj_dir.name, **seg})

    count_rows.sort(key=lambda r: (r["subject"], r["label"]))
    counts_path = OUTPUT_DIR / "wesad_label_counts.csv"
    with counts_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["subject", "label", "count", "dur_s"])
        writer.writeheader()
        writer.writerows(count_rows)

    segment_rows.sort(key=lambda r: (r["subject"], r["start_idx"]))
    segments_path = OUTPUT_DIR / "wesad_label_segments.csv"
    with segments_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["subject", "label", "start_idx", "len_samples", "dur_s"]
        )
        writer.writeheader()
        writer.writerows(segment_rows)

    print(f"Saved label counts to {counts_path}")
    print(f"Saved label segments to {segments_path}")


if __name__ == "__main__":
    main()
