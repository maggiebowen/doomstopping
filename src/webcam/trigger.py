"""
Trigger intervention based on distress levels.
"""
import time
from collections import deque


class DistressTrigger:
    """Monitor distress levels and trigger interventions."""
    
    def __init__(self, threshold=60, window_size=10, consecutive_high=3):
        """
        Initialize distress trigger.
        
        Args:
            threshold: Distress score threshold for triggering
            window_size: Number of recent scores to track
            consecutive_high: Number of consecutive high scores needed to trigger
        """
        self.threshold = threshold
        self.window_size = window_size
        self.consecutive_high = consecutive_high
        self.scores = deque(maxlen=window_size)
        self.last_trigger_time = 0
        self.cooldown_seconds = 300  # 5 minutes between triggers
        
    def add_score(self, distress_score):
        """
        Add a new distress score to the monitoring window.
        
        Args:
            distress_score: Current distress score (0-100)
        """
        self.scores.append(distress_score)
        
    def should_trigger(self):
        """
        Check if intervention should be triggered.
        
        Returns:
            Boolean indicating if trigger condition is met
        """
        # Check cooldown
        if time.time() - self.last_trigger_time < self.cooldown_seconds:
            return False
        
        # Need enough data points
        if len(self.scores) < self.consecutive_high:
            return False
        
        # Check if last N scores are all above threshold
        recent_scores = list(self.scores)[-self.consecutive_high:]
        all_high = all(score >= self.threshold for score in recent_scores)
        
        if all_high:
            self.last_trigger_time = time.time()
            return True
        
        return False
    
    def reset(self):
        """Reset the trigger state."""
        self.scores.clear()
        self.last_trigger_time = 0
