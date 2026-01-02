import cv2
import numpy as np
from deepface import DeepFace
from collections import deque
import logging
import time
import sys

# Robust Scoring Logic
import distress_score

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("EmotionDetector")

# Constants
ANALYSIS_INTERVAL = 0.2  # Analyze emotion every 0.2 seconds (5 FPS)
WINDOW_SIZE = 10         # Rolling window size (10 samples @ 0.2s = 2s history)
CALIBRATION_SECONDS = 12.0
MIN_BASELINE_SAMPLES = 5 

def get_centered_coords(text, font, scale, thickness, img_width, img_height):
    text_size = cv2.getTextSize(text, font, scale, thickness)[0]
    text_x = (img_width - text_size[0]) // 2
    text_y = (img_height + text_size[1]) // 2
    return text_x, text_y

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

    # Get frame dimensions for centering
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 2. State Variables
    scores_deque = deque(maxlen=WINDOW_SIZE)
    last_analysis_time = 0
    
    # Baseline State
    baseline_start_time = time.time()
    baseline_samples = []
    baseline_mean = None # None = Calibrating, Float = Running
    
    # Defaults for display
    smoothed_score = 0.0 # Raw Smoothed (0-1)
    final_score = 0.0    # Baseline Corrected (0-1)
    current_dominant = "Neutral"
    current_emotions = {} # Store raw emotion dict
    raw_sum = 0.0        # Current frame raw score
    raw_sum = 0.0        # Current frame raw score
    distress_z = 0.0     # Z-score for triggers
    baseline_std = 1.0   # Default to 1.0 to avoid divide-by-zero

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Failed to read frame.")
            break
        
        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        current_time = time.time()

        # 3. DeepFace Analysis (Time-Based)
        if (current_time - last_analysis_time) >= ANALYSIS_INTERVAL:
            last_analysis_time = current_time
            try:
                # analyze() returns a list of result dicts (one per face)
                results = DeepFace.analyze(
                    img_path=frame, 
                    actions=['emotion'], 
                    enforce_detection=False,
                    silent=True
                )

                if results:
                    # Take the largest face (width * height)
                    face = max(results, key=lambda x: x['region']['w'] * x['region']['h'])
                    emotions = face['emotion']
                    current_emotions = emotions # Save for display
                    current_dominant = face['dominant_emotion']
                    
                    # --- SCORING LOGIC ---
                    
                    # 1. Compute Raw Weighted Sum (Direct Percentage)
                    raw_sum = distress_score.calculate_weighted_distress(emotions)
                    
                    # 2. Smooth (Rolling Window)
                    scores_deque.append(raw_sum)
                    smoothed_score = np.mean(scores_deque)
                    
                    # 3. Calibration / Baseline Logic
                    
                    if baseline_mean is None:
                        # CALIBRATION PHASE
                        baseline_samples.append(smoothed_score)
                        
                        elapsed = current_time - baseline_start_time
                        remaining = max(0, CALIBRATION_SECONDS - elapsed)
                        
                        if elapsed >= CALIBRATION_SECONDS and len(baseline_samples) >= MIN_BASELINE_SAMPLES:
                            baseline_mean = np.mean(baseline_samples)
                            baseline_std = np.std(baseline_samples)
                            if baseline_std < 0.01: baseline_std = 0.01 # Prevent zero std
                            logger.info(f"BASELINE LOCKED: Mean={baseline_mean*100:.1f}%, Std={baseline_std:.4f}")
                        
                        # Output Status (Scrolling Print)
                        print(f"CALIBRATING | {remaining:.1f}s left | smooth={smoothed_score*100:.1f}% | dominant={current_dominant}")
                        final_score = 0.0
                        
                    else:
                        # RUNNING PHASE
                        final_score = distress_score.apply_baseline_correction(smoothed_score, baseline_mean)
                        
                        # Calculate Z-Score
                        distress_z = (smoothed_score - baseline_mean) / baseline_std
                        
                        # Explicitly show the math for the user
                        # Format: Raw (Current Frame) | Smooth (Window) | Final (Corrected)
                        print(
                            f"Raw: {raw_sum*100:.1f}% | "
                            f"Smooth: {smoothed_score*100:.1f}% | "
                            f"Final: {final_score*100:.1f}% | "
                            f"Dominant: {current_dominant}"
                        )
            
            except Exception as e:
                # logger.error(f"Error: {e}")
                pass

        # 4. Draw Overlay
        
        if baseline_mean is None:
            # --- CALIBRATION SCREEN ---
            # Darken background
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # Countdown
            elapsed = time.time() - baseline_start_time
            remaining = max(0, CALIBRATION_SECONDS - elapsed)
            
            # Centered Text 1: Title
            title = "CALIBRATION MODE"
            tx, ty = get_centered_coords(title, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3, width, height)
            cv2.putText(frame, title, (tx, ty - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

            # Centered Text 2: Instruction
            instr = "Keep a Neutral Expression"
            tx, ty = get_centered_coords(instr, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2, width, height)
            cv2.putText(frame, instr, (tx, ty - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Centered Text 3: Countdown
            count_text = f"{remaining:.1f}s"
            tx, ty = get_centered_coords(count_text, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 5, width, height)
            cv2.putText(frame, count_text, (tx, ty + 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 5)
            
        else:
            # --- RUNNING SCREEN ---
            
            # Status Bar Background (Expanded for bars)
            cv2.rectangle(frame, (0, 0), (450, 400), (40, 40, 40), -1)
            
            # Phase
            cv2.putText(frame, "MONITORING ACTIVE", (20, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Final Score Logic for Colors
            # Score is 0.0 - 1.0
            score_pct = final_score * 100
            score_color = (0, 255, 0) # Green (< 50)
            if score_pct > 75: 
                score_color = (0, 0, 255) # Red
            elif score_pct > 50:
                score_color = (0, 255, 255) # Yellow
                
            cv2.putText(frame, f"Distress: {score_pct:.1f}%", (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, score_color, 3)
            
            # Dominant Emotion
            cv2.putText(frame, f"Dominant: {current_dominant}", (20, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

            # Draw Raw Emotion Bars
            # Sort order: Non-Distress first, then Distress
            display_order = ["neutral", "happy", "sad", "angry", "fear", "disgust", "surprise"]
            bar_y = 160
            
            for emo in display_order:
                # Get value (0-100 or 0-1 depending on DeepFace, normalize just in case)
                val = current_emotions.get(emo, 0.0)
                # DeepFace usually gives 0-100 percentages. If small, assume it's 0-1 and scale.
                # However, our distress_score helper assumes checks. 
                # For display, we want pure 0-100.
                if val <= 1.0 and val > 0.001: 
                    # If it looks normalized, scale up
                    val *= 100
                
                # Bar Logic
                bar_width = int(val * 2.5) # Scale 100% -> 250px
                bar_color = (200, 200, 200) # Gray
                
                if emo in ["sad", "angry", "fear", "disgust"]:
                    bar_color = (0, 100, 255) # Orange/Red
                
                # Label
                cv2.putText(frame, f"{emo.capitalize()}: {val:.1f}%", (20, bar_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Bar Fill
                cv2.rectangle(frame, (120, bar_y - 10), (120 + bar_width, bar_y), bar_color, -1)
                # Bar Outline
                cv2.rectangle(frame, (120, bar_y - 10), (120 + 250, bar_y), (100, 100, 100), 1)
                
                bar_y += 30
        
        # Display Webcam View
        cv2.imshow('Real-Time Emotion', frame)

        # 5. Quit Check
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    logger.info("Webcam released.")

if __name__ == "__main__":
    run_realtime_emotion()
