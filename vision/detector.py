import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np

class CatDetector:
    def __init__(self, model_path='efficientdet_lite2.tflite', min_detection_confidence=0.4):
        """
        Initialize MediaPipe Object Detection using Tasks API.
        """
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.ObjectDetectorOptions(
            base_options=base_options,
            score_threshold=min_detection_confidence,
            max_results=5  # We only need top results
        )
        self.detector = vision.ObjectDetector.create_from_options(options)

    def detect(self, image, filter_cats=True):
        """
        Detects objects in the image.
        Returns a list of dicts.
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create MP Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        
        # Detect
        detection_result = self.detector.detect(mp_image)
        
        detections = []
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            score = detection.categories[0].score
            category_name = detection.categories[0].category_name
            
            # Filter for 'cat' class
            if filter_cats and category_name != 'cat':
                continue

            detections.append({
                    'bbox': (bbox.origin_x, bbox.origin_y, bbox.width, bbox.height),
                    'score': score,
                    'class': category_name,
                    'raw': detection
                })
                
        return detections
