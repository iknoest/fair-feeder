import cv2
import numpy as np
import mediapipe as mp
import vision.detector
import vision.identifier

def test_imports():
    print("Testing imports...")
    print(f"OpenCV Version: {cv2.__version__}")
    print(f"Numpy Version: {np.__version__}")
    print(f"MediaPipe Version: {mp.__version__}")
    return True

def test_detector_init():
    print("Testing Detector Initialization...")
    try:
        det = vision.detector.CatDetector()
        print("Detector initialized successfully.")
    except Exception as e:
        print(f"Detector initialization failed: {e}")
        return False
    return True

def test_identifier_init():
    print("Testing Identifier Initialization...")
    try:
        ident = vision.identifier.CatIdentifier()
        print("Identifier initialized successfully.")
    except Exception as e:
        print(f"Identifier initialization failed: {e}")
        return False
    return True

if __name__ == "__main__":
    if test_imports() and test_detector_init() and test_identifier_init():
        print("ENVIRONMENT VERIFICATION PASSED")
    else:
        print("ENVIRONMENT VERIFICATION FAILED")
