"""
Breathing exercise guidance script.
"""
import time
import threading


class BreathingExercise:
    """Guide user through breathing exercises."""
    
    def __init__(self, inhale_sec=4, hold_sec=4, exhale_sec=6):
        """
        Initialize breathing exercise parameters.
        
        Args:
            inhale_sec: Seconds to inhale
            hold_sec: Seconds to hold breath
            exhale_sec: Seconds to exhale
        """
        self.inhale_sec = inhale_sec
        self.hold_sec = hold_sec
        self.exhale_sec = exhale_sec
        self.is_running = False
        
    def run_cycle(self, callback=None):
        """
        Run a single breathing cycle.
        
        Args:
            callback: Optional function to call with status updates
                     (phase_name, seconds_remaining)
        """
        phases = [
            ("Breathe In", self.inhale_sec),
            ("Hold", self.hold_sec),
            ("Breathe Out", self.exhale_sec),
        ]
        
        for phase_name, duration in phases:
            for remaining in range(duration, 0, -1):
                if callback:
                    callback(phase_name, remaining)
                else:
                    print(f"{phase_name}: {remaining}s")
                time.sleep(1)
                
    def start_continuous(self, cycles=5, callback=None):
        """
        Run breathing exercise for multiple cycles.
        
        Args:
            cycles: Number of breathing cycles to complete
            callback: Optional callback for status updates
        """
        self.is_running = True
        
        for cycle in range(1, cycles + 1):
            if not self.is_running:
                break
                
            print(f"\n--- Cycle {cycle}/{cycles} ---")
            self.run_cycle(callback)
            
        self.is_running = False
        print("\nBreathing exercise complete!")
        
    def start_async(self, cycles=5, callback=None):
        """
        Start breathing exercise in background thread.
        
        Args:
            cycles: Number of breathing cycles
            callback: Optional callback for status updates
        """
        thread = threading.Thread(
            target=self.start_continuous,
            args=(cycles, callback)
        )
        thread.daemon = True
        thread.start()
        return thread
        
    def stop(self):
        """Stop the breathing exercise."""
        self.is_running = False


if __name__ == "__main__":
    print("Starting 4-4-6 breathing exercise...\n")
    exercise = BreathingExercise(inhale_sec=4, hold_sec=4, exhale_sec=6)
    exercise.start_continuous(cycles=3)
