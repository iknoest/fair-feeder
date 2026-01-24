import cv2
import config
from vision.detector import CatDetector
from vision.identifier import CatIdentifier
from vision.motion import MotionDetector

def main():
    print("Initializing Fair-Feeder Cat Monitor...")
    print("Press 'd' to toggle Debug Mode (shows all detections)")
    
    debug_mode = False
    
    # Initialize Vision Modules
    detector = CatDetector(min_detection_confidence=config.CONFIDENCE_THRESHOLD, model_path=config.MODEL_PATH)
    identifier = CatIdentifier()
    motion_detector = MotionDetector()
    
    # Initialize Camera
    # Initialize Camera
    if getattr(config, 'SYSTEM_MODE', 'legacy') == 'tapo':
        print(f"Connecting to Tapo Camera at {config.TAPO_IP}...")
        # Provide instructions if password is missing
        if not config.TAPO_PASS:
            print("ERROR: TAPO_PASS is not set. Please set it in config.py or environment variables.")
            return
        source = config.RTSP_URL
    else:
        print(f"Connecting to local webcam {config.CAMERA_INDEX}...")
        source = config.CAMERA_INDEX
        
    cap = cv2.VideoCapture(source)
    
    # Only set properties for local webcam, RTSP streams usually ignore these or error out
    if isinstance(source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera with index {config.CAMERA_INDEX}")
        return

    print("Camera started. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            

            
        # 1. Motion Detection
        has_motion, motion_mask, motion_contours = motion_detector.detect_motion(frame)

        # 2. Object Detection
        # If 'd' was toggled, we might need a way to store state. 
        # For simplicity, let's use a mutable config or a local variable outside the loop.
        # Ideally, main() should have a `debug_mode` local var.
        
        # Let's fix the scope. We'll declare it before the loop
        
        detections = detector.detect(frame, filter_cats=not debug_mode)
        
        for det in detections:
            bbox = det['bbox']
            score = det['score']
            x, y, w, h = bbox
            
            # Identify Cat
            cat_name = identifier.identify(frame, bbox)
            
            # Draw Bounding Box and Label
            color = (0, 255, 0) # Green for Dan
            if cat_name == "Sanbo":
                color = (0, 165, 255) # Orange for Sanbo
            elif cat_name == "Unknown":
                color = (0, 0, 255) # Red for Unknown
                
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            label = f"{cat_name} ({score:.2f})"
            if debug_mode:
                label = f"{det['class']} ({score:.2f})"
            
            # If it's NOT a cat, but it IS moving, warn the user
            if det['class'] != 'cat':
                # Check if this bbox has motion inside it
                roi_motion = motion_mask[y:y+h, x:x+w]
                if roi_motion.size > 0:
                    motion_ratio = cv2.countNonZero(roi_motion) / (w * h)
                    if motion_ratio > 0.05: # >5% of the object is moving
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2) # Yellow
                        label = f"MOVING {det['class']}?"
                
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            # Debug view (Warning: this modifies frame in place in the current Identifier impl)
            # identifier.debug_view(frame, bbox) 

            # identifier.debug_view(frame, bbox) 

        # If no cats found, but significant motion, draw contours
        if not detections and has_motion:
             cv2.drawContours(frame, motion_contours, -1, (255, 0, 255), 1) # Pink contours for raw motion
             if debug_mode:
                 cv2.putText(frame, "Motion Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        # Display Result
        cv2.imshow('Fair-Feeder', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            debug_mode = not debug_mode
            print(f"Debug Mode: {debug_mode}")
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
