"""
Webcam capture and frame processing.
"""
import cv2


class WebcamCapture:
    """Handle webcam video capture."""
    
    def __init__(self, camera_id=0):
        """
        Initialize webcam capture.
        
        Args:
            camera_id: Camera device ID (0 for default)
        """
        self.cap = cv2.VideoCapture(camera_id)
        self.is_running = False
        
    def start(self):
        """Start capturing frames."""
        self.is_running = True
        
    def stop(self):
        """Stop capturing frames."""
        self.is_running = False
        
    def read_frame(self):
        """
        Read a single frame from webcam.
        
        Returns:
            Frame as numpy array, or None if failed
        """
        if not self.cap.isOpened():
            return None
            
        ret, frame = self.cap.read()
        return frame if ret else None
        
    def release(self):
        """Release webcam resource."""
        self.cap.release()
        
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        self.release()
