import cv2
import numpy as np

class CatIdentifier:
    def __init__(self):
        pass
        
    def identify(self, frame, bbox):
        """
        Analyzes the ROI defined by bbox to distinguish Dan from Sanbo.
        Returns: 'Sanbo' (Calico), 'Dan' (Tuxedo), or 'Unknown'
        """
        x, y, w, h = bbox
        
        # Ensure bbox is within frame
        h_img, w_img, _ = frame.shape
        x = max(0, x)
        y = max(0, y)
        w = min(w, w_img - x)
        h = min(h, h_img - y)
        
        if w < 10 or h < 10:
            return "Unknown"
            
        roi = frame[y:y+h, x:x+w]
        
        # --- LOGIC SELECTION ---
        import config
        if getattr(config, 'SIMULATE_IR', False):
            return self._identify_ir_pattern(roi)
        else:
            return self._identify_color(roi, w, h)

    def _identify_color(self, roi, w, h):
        """Original Color Logic (Orange Detection)"""
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_orange1 = np.array([0, 50, 50])
        upper_orange1 = np.array([25, 255, 255])
        lower_orange2 = np.array([160, 50, 50])
        upper_orange2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv_roi, lower_orange1, upper_orange1)
        mask2 = cv2.inRange(hsv_roi, lower_orange2, upper_orange2)
        orange_mask = cv2.bitwise_or(mask1, mask2)
        
        orange_pixels = cv2.countNonZero(orange_mask)
        total_pixels = w * h
        orange_ratio = orange_pixels / total_pixels
        
        ORANGE_THRESHOLD = 0.02
        if orange_ratio > ORANGE_THRESHOLD:
            return "Sanbo"
        else:
            return "Dan"

    def _identify_ir_pattern(self, roi):
        """
        Refined IR/Grayscale Logic: "Black Dominance"
        
        The previous "Mid-Gray" logic confused blue swaters/shadows with Sanbo.
        New Strategy:
        - Dan (Tuxedo): Has significant patches of TRUE BLACK/DARK (< 50).
        - Sanbo (Calico): Is mostly White + Orange. Orange maps to Gray (approx 100-150).
        
        Hypothesis: If the ROI has a high percentage of VERY DARK pixels, it is likely Dan.
        """
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 1. Focus on the Center of the bounding box (to avoid background edges)
        h, w = gray_roi.shape
        cy, cx = h // 2, w // 2
        chy, chx = h // 3, w // 3 # Center height/width (1/3rd of box)
        center_crop = gray_roi[cy-chy:cy+chy, cx-chx:cx+chx]
        
        if center_crop.size == 0:
            center_crop = gray_roi
            
        # 2. Count DARK pixels (Dan's fur)
        # Threshold 50 might need tuning based on lighting.
        # In the user's photo, Dan is quite dark.
        DARK_THRESHOLD = 60
        dark_mask = cv2.inRange(center_crop, 0, DARK_THRESHOLD)
        dark_count = cv2.countNonZero(dark_mask)
        total_pixels = center_crop.size
        
        dark_ratio = dark_count / total_pixels
        
        # If > 30% of the center is dark, assume Dan.
        DAN_BLACK_THRESHOLD = 0.3
        
        if dark_ratio > DAN_BLACK_THRESHOLD:
            return "Dan"
        else:
            return "Sanbo"


    def debug_view(self, frame, bbox):
        """
        Returns the ROI with the mask overlay for debugging.
        """
        x, y, w, h = bbox
        h_img, w_img, _ = frame.shape
        x = max(0, x)
        y = max(0, y)
        w = min(w, w_img - x)
        h = min(h, h_img - y)
        
        if w < 10 or h < 10:
            return frame
            
        roi = frame[y:y+h, x:x+w]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_orange1 = np.array([0, 50, 50])
        upper_orange1 = np.array([25, 255, 255])
        lower_orange2 = np.array([160, 50, 50])
        upper_orange2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv_roi, lower_orange1, upper_orange1)
        mask2 = cv2.inRange(hsv_roi, lower_orange2, upper_orange2)
        orange_mask = cv2.bitwise_or(mask1, mask2)
        
        # Overlay mask on ROI
        input_roi = roi.copy()
        input_roi[orange_mask > 0] = (0, 0, 255) # Paint orange detectors Red
        
        frame[y:y+h, x:x+w] = input_roi
        return frame
