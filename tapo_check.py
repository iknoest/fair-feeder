import cv2
import sys

def test_tapo_connection():
    print("--- Tapo Camera Connection Test ---")
    print("Note: You must create a 'Camera Account' in the Tapo App (Advanced Settings).")
    print("This is DIFFERENT from your TP-Link login email.")
    
    username = input("Enter Camera Username: ").strip()
    password = input("Enter Camera Password: ").strip()
    ip_address = input("Enter Camera IP Address (e.g., 192.168.1.50): ").strip()
    
    # Tapo C210 RTSP URL Format
    # stream1 = High Quality (1080p/2K)
    # stream2 = Low Quality (360p)
    rtsp_url = f"rtsp://{username}:{password}@{ip_address}:554/stream1"
    
    print(f"\nAttempting to connect to:\n{rtsp_url.replace(password, '******')}")
    
    cap = cv2.VideoCapture(rtsp_url)
    
    if not cap.isOpened():
        print("\n[ERROR] Could not open video stream.")
        print("Check:")
        print("1. Is the IP address correct?")
        print("2. Is the username/password correct (Camera Account)?")
        print("3. Is the computer on the same Wi-Fi network as the camera?")
        return

    print("\n[SUCCESS] Connected to camera!")
    print("Press 'q' to quit the preview.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to receive frame.")
            break
            
        # Resize for display if 2K is too big
        display_frame = cv2.resize(frame, (800, 600))
        cv2.imshow('Tapo C210 Stream Test', display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        test_tapo_connection()
    except KeyboardInterrupt:
        pass
