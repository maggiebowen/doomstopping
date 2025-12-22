# Doomstopping

A stress detection and intervention system combining physiological signals (WESAD dataset) and real-time facial emotion analysis.

## Project Structure

```
project-root/
  README.md                 # This file
  docs/                     # Slides, UML diagrams, references
  data/                     # Gitignored data directory
    raw/                    # Raw WESAD dataset
    processed/              # Processed features
  models/                   # Gitignored trained models
  notebooks/                # Jupyter notebooks for experiments
    01_wesad_stress_model.ipynb
    02_deepface_webcam.ipynb
  src/                      # Source code
    wesad/                  # WESAD dataset processing
      load_wesad.py         # Dataset loader
      features_hrv.py       # HRV feature extraction
      features_eda.py       # EDA feature extraction
      train_eval.py         # Model training and evaluation
    webcam/                 # Real-time webcam analysis
      capture.py            # Webcam capture
      distress_score.py     # Distress scoring from emotions
      trigger.py            # Intervention triggering logic
    ui/                     # User interface
      overlay_app.py        # Distress monitor overlay
      breathing_script.py   # Breathing exercise guidance
  requirements.txt          # Python dependencies
  .gitignore               # Git ignore rules
```

## Features

### WESAD Stress Detection
- Load and process WESAD physiological dataset
- Extract HRV (Heart Rate Variability) features from ECG
- Extract EDA (Electrodermal Activity) features
- Train machine learning models for stress classification

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

1. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download WESAD dataset** (optional):
   - Visit: https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/
   - Place extracted data in `data/raw/WESAD/`

## Usage

### Training a Stress Model
See `notebooks/01_wesad_stress_model.ipynb` for a complete example.

```python
from src.wesad.load_wesad import load_subject_data
from src.wesad.train_eval import train_stress_model

# Load data
data = load_subject_data('S2', data_dir='data/raw/WESAD')

# Extract features and train
# ... (see notebook for details)
```

### Real-time Webcam Monitoring
See `notebooks/02_deepface_webcam.ipynb` for a complete example.

```python
from src.webcam.capture import WebcamCapture
from src.webcam.distress_score import analyze_emotion, compute_distress_score
from src.webcam.trigger import DistressTrigger

# Initialize
trigger = DistressTrigger(threshold=60)

# Monitor in real-time
with WebcamCapture() as webcam:
    frame = webcam.read_frame()
    emotions = analyze_emotion(frame)
    score = compute_distress_score(emotions)
    trigger.add_score(score)
    
    if trigger.should_trigger():
        print("Intervention needed!")
```

### Running the UI Demo
```bash
python src/ui/overlay_app.py
```

### Testing Breathing Exercise
```bash
python src/ui/breathing_script.py
```

## Dependencies

Key libraries:
- **Data processing**: `numpy`, `pandas`, `scipy`
- **Machine learning**: `scikit-learn`, `tensorflow` (or `pytorch`)
- **Signal processing**: `neurokit2`, `heartpy`
- **Computer vision**: `opencv-python`, `deepface`
- **UI**: `tkinter` (built-in)

See `requirements.txt` for complete list.

## License

[Add your license here]

## Acknowledgments

- WESAD dataset: Schmidt et al. (2018)
- DeepFace library for emotion recognition
