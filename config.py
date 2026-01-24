# Configuration for Fair-Feeder

# Camera Settings
# CAMERA_INDEX = 0  # 0 for default laptop webcam
# For Tapo Camera, we use the RTSP URL
# Camera Settings
# CAMERA_INDEX = 0  # 0 for default laptop webcam

FRAME_WIDTH = 1280 # Tapo Stream 1 is usually HD
FRAME_HEIGHT = 720

# System Mode: 'legacy' (Webcam) or 'tapo' (RTSP Stream)
# Set to 'tapo' to use the camera at CAMERA_IP
SYSTEM_MODE = 'tapo'

# Tapo Camera Configuration
# Note: It is best to load these from environment variables or a .env file for security.
import os
TAPO_IP = "192.168.1.246"
TAPO_USER = os.getenv("TAPO_USER", "c210RTSP")    # CHANGE or set env var
TAPO_PASS = os.getenv("TAPO_PASS", "mkeihfCCP@198964") # CHANGE or set env var

# RTSP Stream URL (Stream 1 is HD, Stream 2 is SD)
RTSP_URL = f"rtsp://{TAPO_USER}:{TAPO_PASS}@{TAPO_IP}:554/stream1"

# Detection Settings
CONFIDENCE_THRESHOLD = 0.4
MODEL_PATH = 'efficientdet_lite2.tflite'  # Larger model, better accuracy

# Identification Settings
# Thresholds for Histogram analysis to distinguish Dan (B&W) from Sanbo (Calico)
# These may need tuning.
CONTRAST_THRESHOLD = 50 

# Simulation Settings
SIMULATE_IR = True # Force input to grayscale to simulate Night Vision
