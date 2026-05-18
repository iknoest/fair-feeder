# USB Camera Monitor Guide (Logitech C925e)

This guide explains how to set up a second instance of the cat monitor using a USB camera (like the Logitech C925e) alongside your existing Tapo monitor.

## 1. Prerequisites
- Logitech C925e (or any USB webcam) plugged into the Pi 5.
- `rclone` configured with a remote named `gdrive-randomdice:`.

## 2. Testing the Camera
Run the test script to ensure the Pi recognizes the USB camera:
```bash
/home/pi5/Feeder/fair-feeder/.venv/bin/python scripts/test_usb_camera.py
```
It should report `Resolution: 1280.0x720.0` and read frames successfully.

## 3. Deployment as a Service
We have created a specific service file for the USB camera: `deploy/usb-monitor.service`.

### Copy the service file:
```bash
sudo cp deploy/usb-monitor.service /etc/systemd/system/
```

### Enable and Start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable usb-monitor.service
sudo systemctl start usb-monitor.service
```

### Check Status:
```bash
sudo systemctl status usb-monitor.service
journalctl -u usb-monitor.service -f
```

## 4. Configuration Details
The `usb-monitor.service` uses environment variables to override defaults:
- `CAMERA_TYPE=usb`: Switches from RTSP to USB mode.
- `USB_CAMERA_INDEX=0`: Uses `/dev/video0`.
- `RCLONE_DEST_PATH=14yBPCZvjrztIqxI5l-ckgZYkC7D0ZTdS`: Uploads directly to your specified Google Drive folder.

If you need to change these, edit the service file:
```bash
sudo nano /etc/systemd/system/usb-monitor.service
# Change values, then:
sudo systemctl daemon-reload
sudo systemctl restart usb-monitor.service
```

## 5. Running Both Monitors
You can run both `cat-monitor.service` (Tapo) and `usb-monitor.service` (Logitech) simultaneously. They use different temporary folders and different upload paths.
- Tapo clips: `gdrive-randomdice:` (Root or as configured in `.env`)
- USB clips: `gdrive-randomdice:14yBPCZvjrztIqxI5l-ckgZYkC7D0ZTdS`
