# Configuration for Fair-Feeder

# Camera Settings
# CAMERA_INDEX = 0  # 0 for default laptop webcam
# For Tapo Camera, we use the RTSP URL
# Camera Settings
# CAMERA_INDEX = 0  # 0 for default laptop webcam

FRAME_WIDTH = 1280 # Tapo Stream 1 is usually HD
FRAME_HEIGHT = 720

import os
from pathlib import Path
from typing import Any, Dict

# System Mode: 'rtsp' (Tapo) or 'usb' (Logitech)
# Can also be set via environment variable CAMERA_TYPE
CAMERA_TYPE = os.getenv('CAMERA_TYPE', 'rtsp')

# USB Camera Configuration
USB_CAMERA_INDEX = int(os.getenv('USB_CAMERA_INDEX', 0))
LOGITECH_DRIVE_FOLDER_ID = "14yBPCZvjrztIqxI5l-ckgZYkC7D0ZTdS"

# rclone sync destination path or folder ID
RCLONE_REMOTE = os.getenv('RCLONE_REMOTE', 'gdrive-randomdice:')
RCLONE_DEST_PATH = os.getenv('RCLONE_DEST_PATH', '') # Leave empty for root, or set to a folder ID/name

# Authoritative camera staging and destination mapping
CAMERA_TARGETS: Dict[str, Dict[str, Any]] = {
    "rtsp": {
        "camera_type": "rtsp",
        "rclone_remote": RCLONE_REMOTE,
        "rclone_dest_path": "",
        "staging_dir": Path("/home/pi5/Pictures/gdrive-randomdice-sync"),
        "dev_staging_dir": Path("/tmp/gdrive-randomdice-sync"),
    },
    "usb": {
        "camera_type": "usb",
        "rclone_remote": RCLONE_REMOTE,
        "rclone_dest_path": LOGITECH_DRIVE_FOLDER_ID,
        "staging_dir": Path("/home/pi5/Pictures/usb-camera-sync"),
        "dev_staging_dir": Path("/tmp/usb-camera-sync"),
    },
}

def get_camera_target_for_path(path: Any) -> Dict[str, Any]:
    """Resolves camera target config from staging directory or file path."""
    p_str = str(path)
    if "usb" in p_str:
        return CAMERA_TARGETS["usb"]
    if "gdrive-randomdice" in p_str:
        return CAMERA_TARGETS["rtsp"]
    cam = os.getenv("CAMERA_TYPE", "rtsp")
    return CAMERA_TARGETS.get(cam, CAMERA_TARGETS["rtsp"])

# Tapo Camera Configuration
# Credentials are loaded from Infisical (Colab) or environment variables (local).

def _load_tapo_credentials():
    """Load Tapo credentials from Infisical (Colab) or env vars (local)."""
    try:
        # Attempt Infisical load for Colab environments
        from infisical_sdk import InfisicalSDKClient
        from google.colab import userdata

        client = InfisicalSDKClient(host="https://app.infisical.com")
        client.auth.universal_auth.login(
            client_id=userdata.get('INFISICAL_ID'),
            client_secret=userdata.get('INFISICAL_SECRET'),
        )
        proj_id = userdata.get('INFISICAL_PROJECT_ID')

        def get_secret(name):
            return client.secrets.get_secret_by_name(
                secret_name=name,
                project_id=proj_id,
                environment_slug="dev",
                secret_path="/",
            ).secretValue

        return get_secret('TAPO_IP'), get_secret('TAPO_USER'), get_secret('TAPO_PASS')
    except Exception:
        # Fallback to environment variables (local, non-Colab use)
        return (
            os.getenv('TAPO_IP', '<YOUR_CAMERA_IP>'),
            os.getenv('TAPO_USER', '<YOUR_CAMERA_USER>'),
            os.getenv('TAPO_PASS', '<YOUR_CAMERA_PASSWORD>'),
        )

TAPO_IP, TAPO_USER, TAPO_PASS = _load_tapo_credentials()

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
