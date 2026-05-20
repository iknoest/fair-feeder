import cv2
import time

def test_usb_camera(index=0):
    print(f"Attempting to open USB camera at index {index}...")
    cap = cv2.VideoCapture(index)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera at index {index}")
        return

    # Try to set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Camera opened successfully!")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")

    print("Reading 10 frames...")
    for i in range(10):
        ret, frame = cap.read()
        if ret:
            print(f"Frame {i} read successfully (shape: {frame.shape})")
        else:
            print(f"Error: Could not read frame {i}")
        time.sleep(0.1)

    cap.release()
    print("Camera released.")

if __name__ == "__main__":
    test_usb_camera(0)
