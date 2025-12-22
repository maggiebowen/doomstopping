"""
Distress score computation from facial analysis.
"""
from deepface import DeepFace
import numpy as np


def analyze_emotion(frame):
    """
    Analyze emotions from face in frame using DeepFace.
    
    Args:
        frame: Image frame (BGR format from OpenCV)
        
    Returns:
        Dictionary with emotion probabilities
    """
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        return result[0]['emotion'] if isinstance(result, list) else result['emotion']
    except Exception as e:
        print(f"Error analyzing emotion: {e}")
        return None


def compute_distress_score(emotion_dict):
    """
    Compute overall distress score from emotion probabilities.
    
    Args:
        emotion_dict: Dictionary with emotion probabilities from DeepFace
        
    Returns:
        Distress score (0-100, higher = more distress)
    """
    if emotion_dict is None:
        return 0
    
    # Weight negative emotions higher
    distress_weights = {
        'angry': 1.0,
        'fear': 1.0,
        'sad': 0.8,
        'disgust': 0.7,
        'neutral': 0.0,
        'happy': -0.5,
        'surprise': 0.3
    }
    
    score = 0
    for emotion, prob in emotion_dict.items():
        weight = distress_weights.get(emotion, 0)
        score += prob * weight
    
    # Normalize to 0-100 range
    return max(0, min(100, score))


def get_stress_level(distress_score):
    """
    Categorize distress score into stress levels.
    
    Args:
        distress_score: Distress score (0-100)
        
    Returns:
        String: 'low', 'medium', or 'high'
    """
    if distress_score < 30:
        return 'low'
    elif distress_score < 60:
        return 'medium'
    else:
        return 'high'
