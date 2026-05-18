import cv2
import sys

def capture(index=0, output='capture.jpg'):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f'Error: Could not open camera {index}')
        sys.exit(1)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Warm up camera
    for _ in range(5):
        cap.read()
        
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(output, frame)
        print(f'Successfully saved to {output}')
    else:
        print('Error: Could not capture frame')
        sys.exit(1)
    
    cap.release()

if __name__ == '__main__':
    # Default to first USB camera
    capture(0, 'capture_usb.jpg')
