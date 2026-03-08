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
# Credentials are loaded from Infisical (Colab) or environment variables (local).
import os

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
            os.getenv('TAPO_IP', '192.168.1.246'),
            os.getenv('TAPO_USER', 'c210RTSP'),
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
