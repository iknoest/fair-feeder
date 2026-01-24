import cv2
import numpy as np

class MotionDetector:
    def __init__(self, history=500, var_threshold=16, detect_shadows=True):
        """
        Initialize Motion Detector using Background Subtraction.
        """
        self.back_sub = cv2.createBackgroundSubtractorMOG2(
            history=history, 
            varThreshold=var_threshold, 
            detectShadows=detect_shadows
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def detect_motion(self, frame, min_area=500):
        """
        Detects motion in the frame.
        Returns:
        - has_motion: bool
        - mask: binary mask of motion
        - contours: list of contours where motion was detected
        """
        # Apply background subtraction
        fg_mask = self.back_sub.apply(frame)
        
        # Remove shadows (gray pixels) by thresholding
        _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
        
        # Clean up noise
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_detected = False
        significant_contours = []
        
        for contour in contours:
            if cv2.contourArea(contour) > min_area:
                motion_detected = True
                significant_contours.append(contour)
                
        return motion_detected, fg_mask, significant_contours
