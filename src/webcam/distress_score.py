import numpy as np

# --- Constants ---

# Emotion Categories
DISTRESS_EMOTIONS = ["sad", "angry", "fear", "disgust"]
NEUTRAL_EMOTIONS = ["neutral", "happy"]

# Weights: Reflect severity/arousal of distress
EMOTION_WEIGHTS = {
    "sad": 1.0,      # Simplifed weights to keep it "true to percentages"
    "angry": 1.0,
    "fear": 1.0,
    "disgust": 1.0,
    "neutral": 0.0,
    "happy": 0.0
}

EPSILON = 1e-6

def normalize_emotions(emotions):
    """
    Ensure all emotion values are in [0, 1].
    DeepFace sometimes returns percentages (0-100).
    """
    normalized = {}
    for emo, val in emotions.items():
        if val > 1.5:
            normalized[emo] = float(val) / 100.0
        else:
            normalized[emo] = float(val)
        
        normalized[emo] = np.clip(normalized[emo], 0.0, 1.0)
    return normalized

def calculate_weighted_distress(emotions):
    """
    Compute Raw Distress Sum:
    Score = Sum(Weight * Value) for Distress Emotions
    
    This provides a direct reflection of the 'percentage' of distress detected.
    """
    # 1. Normalize inputs
    norm_emotions = normalize_emotions(emotions)
    
    # 2. Compute Mass
    mass_distress = 0.0
    for emo in DISTRESS_EMOTIONS:
        val = norm_emotions.get(emo, 0.0)
        weight = EMOTION_WEIGHTS.get(emo, 1.0)
        mass_distress += (val * weight)
        
    return np.clip(mass_distress, 0.0, 1.0)

def apply_baseline_correction(current_score, baseline_mean):
    """
    Rescale the score based on the user's calibration baseline.
    
    Revised Strategy: Simple Subtraction with Scaling
    This is more sensitive than the previous ratio.
    """
    if baseline_mean is None:
        return 0.0 # Not calibrated yet
        
    # Subtract baseline
    corrected = current_score - baseline_mean
    
    # If below baseline, it's 0
    if corrected < 0:
        return 0.0
        
    # Scaling: Map the remaining range [0, 1-baseline] to [0, 1]
    # Denominator
    denom = 1.0 - baseline_mean
    if denom < 0.05: denom = 0.05 # Prevent divide by zero/noise
        
    final_score = corrected / denom
    
    return np.clip(final_score, 0.0, 1.0)
