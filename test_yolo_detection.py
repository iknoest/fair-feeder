"""
Quick test: YOLOv8n with LOW threshold on Raspberry Pi 5
"""
import sys, time, cv2

try:
    import config as _cfg
    CAMERA_IP, CAMERA_USER, CAMERA_PASS = _cfg.TAPO_IP, _cfg.TAPO_USER, _cfg.TAPO_PASS
except Exception:
    import os
    CAMERA_IP   = os.getenv('TAPO_IP')
    CAMERA_USER = os.getenv('TAPO_USER')
    CAMERA_PASS = os.getenv('TAPO_PASS')

RTSP_URL = f'rtsp://{CAMERA_USER}:{CAMERA_PASS}@{CAMERA_IP}:554/stream1'
SAVE_DIR = '/home/pi5/Feeder/fair-feeder/test_frames'

def main():
    import os
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("=" * 55)
    print("  YOLOv8n LOW THRESHOLD Test (conf=0.10)")
    print("=" * 55)

    from ultralytics import YOLO
    model = YOLO('yolov8n.pt')
    print("  ✅ Model loaded!")

    cap = cv2.VideoCapture(RTSP_URL + '?rtsp_transport=tcp')
    if not cap.isOpened():
        cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        print("  ❌ Cannot connect"); sys.exit(1)
    w, h = int(cap.get(3)), int(cap.get(4))
    print(f"  ✅ Connected! {w}x{h}")

    for _ in range(10):
        cap.read()

    print("\n  Capturing 5 frames (conf=0.10)...")
    print("-" * 55)

    cats_found = 0
    for frame_num in range(1, 6):
        ret, frame = cap.read()
        if not ret: continue

        cv2.imwrite(f"{SAVE_DIR}/low_frame_{frame_num}_raw.jpg", frame)

        t0 = time.time()
        results = model(frame, imgsz=640, conf=0.10, verbose=False)
        ms = (time.time() - t0) * 1000

        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                name = model.names[cls_id]
                bbox = box.xyxy[0].tolist()
                detections.append((name, conf, cls_id, bbox))

        # Draw ALL detections
        annotated = frame.copy()
        for name, conf, cls_id, bbox in detections:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            if name == 'cat':
                color = (0, 255, 0)
                thickness = 4
            elif name == 'dog':
                color = (0, 165, 255)
                thickness = 3
            elif name == 'person':
                color = (255, 0, 0)
                thickness = 3
            else:
                color = (128, 128, 128)
                thickness = 2
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            label = f"{name} {conf:.0%}"
            cv2.putText(annotated, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.imwrite(f"{SAVE_DIR}/low_frame_{frame_num}_annotated.jpg", annotated)

        has_cat = any(d[0] == 'cat' for d in detections)
        if has_cat: cats_found += 1

        # Show ALL detections, not just top 5
        print(f"\n  Frame {frame_num} | {ms:.0f}ms | {len(detections)} objects")
        for name, conf, cls_id, bbox in sorted(detections, key=lambda x: -x[1]):
            emoji = "🐱" if name == "cat" else "🐕" if name == "dog" else "🧑" if name == "person" else "📦"
            marker = " <<<< CAT!" if name == "cat" else ""
            print(f"    {emoji} {name} — {conf:.1%}{marker}")

        if frame_num < 5: time.sleep(2)

    cap.release()
    print(f"\n{'='*55}")
    print(f"  Cats found in: {cats_found}/5 frames")
    if cats_found > 0:
        print(f"  ✅ CAT DETECTION WORKING! 🐱")
    print(f"{'='*55}")

if __name__ == '__main__':
    main()
