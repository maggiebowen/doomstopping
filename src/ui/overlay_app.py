"""
Overlay UI application demo.
"""
import tkinter as tk
from tkinter import ttk


class OverlayApp:
    """Simple overlay UI for distress monitoring."""
    
    def __init__(self):
        """Initialize the overlay application."""
        self.root = tk.Tk()
        self.root.title("Distress Monitor")
        self.root.geometry("300x200")
        
        # Make window stay on top
        self.root.attributes('-topmost', True)
        
        self.setup_ui()
        
    def setup_ui(self):
        """Set up UI elements."""
        # Title
        title = tk.Label(self.root, text="Distress Monitor", 
                        font=("Arial", 16, "bold"))
        title.pack(pady=10)
        
        # Distress score display
        self.score_label = tk.Label(self.root, text="Score: --", 
                                    font=("Arial", 24))
        self.score_label.pack(pady=10)
        
        # Status indicator
        self.status_label = tk.Label(self.root, text="Status: Monitoring", 
                                     font=("Arial", 12))
        self.status_label.pack(pady=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.root, length=250, mode='determinate')
        self.progress.pack(pady=10)
        
    def update_score(self, score):
        """
        Update displayed distress score.
        
        Args:
            score: Distress score (0-100)
        """
        self.score_label.config(text=f"Score: {score:.1f}")
        self.progress['value'] = score
        
        # Update color based on score
        if score < 30:
            color = "green"
        elif score < 60:
            color = "orange"
        else:
            color = "red"
        self.score_label.config(fg=color)
        
    def run(self):
        """Start the UI application."""
        self.root.mainloop()


if __name__ == "__main__":
    app = OverlayApp()
    app.run()
