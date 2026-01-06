# Doomstopping

A stress detection and intervention system combining physiological signals (WESAD dataset) and real-time facial emotion analysis.

## Features

### Real-time Webcam Monitoring
- Capture webcam frames
- Analyze facial emotions using DeepFace
- Compute distress scores from emotion probabilities
- Trigger interventions based on sustained high distress

### Interventions
- Visual overlay UI showing current distress level
- Guided breathing exercises (4-4-6 pattern)
- Configurable thresholds and triggers

## Setup

### 1. Python Environment Setup

First, create a virtual environment to manage dependencies:

```bash
# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

Next, install the required packages:

```bash
pip install -r requirements.txt
```

### 2. Chrome Extension Installation

To enable the breathing intervention on YouTube:

1. Open Chrome and navigate to `chrome://extensions/`.
2. Toggle **Developer mode** in the top right corner.
3. Click **Load unpacked**.
4. Select the `chrome-extension` directory from this project.
5. The "Breathing Overlay" extension should now be active.

**Testing the Extension:**
- Open any YouTube video.
- The extension runs in the background and listens for the distress state from the Python script.
- The overlay will automatically appear when the "INTERVENTION" state is triggered.

## Usage

### Running the Distress Detector

Start the real-time emotion detection script:

```bash
python3 src/webcam/real-time-emotion-detection.py
```

This script will:
1. Open your webcam.
2. Analyze your facial expressions in real-time.
3. Start a local server at `http://127.0.0.1:8765/state` to communicate with the Chrome extension.

> **Note:** To stop the script and close the webcam, click on the webcam window to focus it and press **`q`**.

### Adjusting Thresholds

You can customize the sensitivity of the intervention in `src/webcam/real-time-emotion-detection.py`:

```python
# --- Distress State Machine Config ---
DISTRESS_ENTER_THRESHOLD = 0.60  # Distress score (0.0 - 1.0) required to start accumulating
ACCUM_REQUIRED_SECONDS   = 5.0   # Duration (seconds) of sustained distress to trigger intervention
```

### Distress Score Calculation

The distress score logic is located in `src/webcam/distress_score.py`. It calculates a weighted sum of negative emotions detected by DeepFace. You can adjust the weights to prioritize certain emotions:

```python
# Weights derived from arousal/severity
weights = {
    # Distress Emotions (Higher Weight)
    'sad': 1.5,
    'angry': 1.5,
    'fear': 1.5,
    'disgust': 1.5,
    
    # Non-Distress Emotions
    'neutral': 1.0,
    'happy': 1.0,
    'surprise': 1.0
}
```

The system calculates a **Weighted Distress Ratio**: 
`Score = (DistressSum) / (DistressSum + NonDistressSum)`

This score is then **baseline-corrected** by subtracting the user's calibrated resting neutral baseline to ensure sensitivity to *changes* in expression rather than absolute facial structure.

### Breathing Exercise

The guided breathing script is adapted from **"The Practice of Mindful Consumption"** by **Plum Village Monastery**. 

- **Source**: [the-practice-of-mindful-consumption](https://plumvillage.app/the-practice-of-mindful-consumption/)
- **Timing**: The duration of each breath cycle is defined in `chrome-extension/breathing_exercise.js`. You can increase the `duration` values in the `sequence` array for a longer, more relaxing pace. (Timings have been shortened slightly for demonstration purposes).

## Project Structure

```
doomstopping/
  ├── .gitignore
  ├── README.md                 # Project documentation
  ├── requirements.txt          # Python dependencies
  ├── chrome-extension/         # Chrome Extension for Youtube overlay
  │   ├── assets/
  │   ├── background.js
  │   ├── breathing_exercise.html
  │   ├── breathing_exercise.js
  │   ├── content.js
  │   └── manifest.json
  ├── data/
  │   ├── images/
  │   ├── processed/
  │   ├── raw/
  │   └── wesad_readme.pdf
  ├── models/
  │   ├── stress_rf_model.pkl
  │   └── wesad_linear_svm_3class.joblib
  ├── notebooks/                # Jupyter notebooks for model development
  │   ├── 01_wesad_stress_model.ipynb
  │   ├── 01_wesad_stress_model_binary.ipynb
  │   ├── 02_model_inference_test.ipynb
  │   ├── 03_deepface_stored_photo.ipynb
  │   ├── 03_deepface_webcam copy.ipynb
  │   └── fix_nb.py
  └── src/
      ├── ui/
      │   └── breathing_exercise.html
      ├── webcam/               # Real-time monitoring logic
      │   ├── distress_score.py
      │   └── real-time-emotion-detection.py
      └── wesad/                # WESAD dataset processing logic
          ├── features_acc.py
          ├── features_eda.py
          ├── features_hrv.py
          ├── load_wesad.py
          ├── normalization.py
          └── summarize_wesad.py
```

## Future Work

### Multimodal Integration

The machine learning models developed in the `notebooks/` directory (trained on the WESAD dataset) are designed for future integration into the real-time pipeline. The goal is to transition from facial-only detection to a multimodal system incorporating physiological biosignals from the **CareLab's Embrace Plus smartwatch**.

The `src/wesad/` directory contains the core feature extraction logic required for this transition:
- **`features_hrv.py`**: Extracts time-domain (RMSSD, SDNN) and frequency-domain (LF, HF) Heart Rate Variability features from BVP signals.
- **`features_eda.py`**: Decomposes Electrodermal Activity into tonic and phasic components to identify skin conductance responses (SCRs) associated with sympathetic nervous system arousal.
- **`features_acc.py`**: Processes 3-axis accelerometer data to calculate statistical movement features, used to filter motion artifacts from physiological sensors.
- **`normalization.py`**: Implements subject-specific baseline correction, ensuring that the models respond to physiological *shifts* rather than individual biological baselines.

In the future, real-time streams of HRV and EDA data from the Embrace Plus will be processed through these scripts and fed into the pre-trained `models/stress_rf_model.pkl`. This will allow the system to cross-validate facial distress with internal physiological states, significantly reducing false positives and providing a more robust trigger for interventions.
