import numpy as np

# --- Constants ---
# Emotion Categories
DISTRESS_EMOTIONS = ["sad", "angry", "fear", "disgust"]
NON_DISTRESS_EMOTIONS = ["neutral", "happy", "surprise"]

# Weights: Reflect severity/arousal of distress
EMOTION_WEIGHTS = {
    # Distress
    "sad": 1.25,
    "angry": 1.25,
    "fear": 1.4,     # Higher weight for high-arousal/urgency signals 
    "disgust": 1.4,
    # Non-Distress
    "neutral": 1.0,  # Standard anchor
    "happy": 1.0,
    "surprise": 1.0
}

# prevent division by zero
EPSILON = 1e-6

def normalize_emotions(emotions) -> dict:
    """
    Ensure all emotion values are in [0, 1].
    DeepFace sometimes returns percentages (0-100)? --> maybe not necessary, but just to standardize
    """
    normalized = {}
    for emo, val in emotions.items():
        if val > 1.5:
            normalized[emo] = float(val) / 100.0
        else:
            normalized[emo] = float(val)
        
        normalized[emo] = np.clip(normalized[emo], 0.0, 1.0)
    return normalized

def calculate_weighted_distress(emotions) -> float:
    """
    Compute Raw Distress Ratio:
    S_D = Sum(Weight * Value) for Distress Emotions
    S_N = Sum(Weight * Value) for Non-Distress Emotions
    
    Raw Score = S_D / (S_D + S_N + Epsilon)
    
    This answers: "How much of the model's confidence is on distress vs neutral/happy?"
    """
    # 1. Normalize inputs
    norm_emotions = normalize_emotions(emotions)
    
    # 2. Compute Weighted Sums
    s_d = 0.0
    for emo in DISTRESS_EMOTIONS:
        val = norm_emotions.get(emo, 0.0)
        weight = EMOTION_WEIGHTS.get(emo, 1.0)
        s_d += (val * weight)
        
    s_n = 0.0
    for emo in NON_DISTRESS_EMOTIONS:
        val = norm_emotions.get(emo, 0.0)
        weight = EMOTION_WEIGHTS.get(emo, 1.0)
        s_n += (val * weight)

    # 3. Compute Ratio
    raw_score = s_d / (s_d + s_n + EPSILON)
        
    return np.clip(raw_score, 0.0, 1.0)

def apply_baseline_correction(current_score, baseline_mean) -> float:
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
