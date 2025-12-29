import cv2
import numpy as np
from deepface import DeepFace
from collections import deque
import logging
import time

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EmotionDetector")

# Constants
FRAME_SKIP = 10        # Analyze emotion every N frames
WINDOW_SIZE = 30      # Rolling window size for smoothing (approx 2-3s)
DISTRESS_THRESH = 50.0 # Just for visual coloring (optional)

# Distress is sum of these negative emotions
DISTRESS_EMOTIONS = ['sad', 'angry', 'fear', 'disgust']

def run_realtime_emotion():
    """
    Main loop for real-time webcam emotion detection.
    """
    # 1. Initialize Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Could not open webcam.")
        return

    logger.info("Webcam opened. Press 'q' to quit.")

    # 2. State Variables
    frame_count = 0
    scores_deque = deque(maxlen=WINDOW_SIZE)
    
    # Defaults for display
    current_distress = 0.0
    current_dominant = "Neutral"
    current_emotions = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to read frame.")
            break
        
        # Flip for mirror effect
        frame = cv2.flip(frame, 1)

        # 3. DeepFace Analysis (Every N frames)
        if frame_count % FRAME_SKIP == 0:
            try:
                # analyze() returns a list of result dicts (one per face)
                results = DeepFace.analyze(
                    img_path=frame, 
                    actions=['emotion'], 
                    enforce_detection=False,
                    silent=True # Suppress frequent logging
                )

                if results:
                    # Take the largest face (width * height)
                    face = max(results, key=lambda x: x['region']['w'] * x['region']['h'])
                    
                    # Extract Data
                    current_emotions = face['emotion']
                    current_dominant = face['dominant_emotion']
                    
                    # Calculate Distress Score
                    # Sum percentages of negative emotions
                    raw_score = sum(current_emotions.get(e, 0.0) for e in DISTRESS_EMOTIONS)
                    
                    # Update History
                    scores_deque.append(raw_score)
                    
                    # Update Smoothed Score
                    current_distress = np.mean(scores_deque)
            
            except Exception as e:
                # If analysis fails (e.g. face detection error inside DeepFace), just log and skip update
                # This keeps the last known valid state on screen 
                # logger.debug(f"Analysis failed: {e}")
                pass

        # 4. Draw Overlay
        # Setup Text
        text_color = (0, 255, 0) # Green by default
        if current_distress > 50:
            text_color = (0, 0, 255) # Red if high distress
        
        # Status Bar Background
        cv2.rectangle(frame, (0, 0), (400, 120), (50, 50, 50), -1)
        
        # Display Text
        cv2.putText(frame, f"Distress Score: {current_distress:.1f}%", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2)
        
        cv2.putText(frame, f"Dominant: {current_dominant}", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Display Webcam View
        cv2.imshow('Real-Time Emotion', frame)

        # 5. Quit Check
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    logger.info("Webcam released.")

if __name__ == "__main__":
    run_realtime_emotion()