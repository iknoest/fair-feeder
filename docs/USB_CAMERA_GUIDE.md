# USB Camera Monitor Guide (Logitech C925e)

This guide explains how to set up and manage a second instance of the cat monitor using a USB camera (like the Logitech C925e) alongside your existing Tapo monitor on the Raspberry Pi 5.

## 1. Prerequisites
- Logitech C925e (or any USB webcam) plugged into the Pi 5.
- `rclone` configured with a remote named `gdrive-randomdice:`.

## 2. Testing the Camera
Run the diagnostic script to ensure the Pi recognizes the USB camera:
```bash
/home/pi5/Feeder/fair-feeder/.venv/bin/python scripts/test_usb_camera.py
```
It should report `Resolution: 1280.0x720.0` and read frames successfully.

## 3. Deployment as a Service
The Logitech camera runs as its own independent service: `usb-monitor.service`.

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

### Monitor Logs:
```bash
journalctl -u usb-monitor.service -f
```

## 4. Telegram Commands
The Telegram bot now handles both cameras intelligently:

*   **/status**: Returns a combined report. The Logitech report is labeled **"LOGITECH 🎥"** and shows the last time a cat was specifically identified.
*   **/streaming_logitech**: Captures and sends a fresh 5-second video clip directly from the Logitech camera. This does not interrupt 24/7 monitoring.
*   **/lastclip**: Sends the most recent cat video from **both** cameras.
*   **/help**: Shows the command list (only the Tapo instance responds to keep the chat clean).

## 5. Parallel Operation Details
- **Independence**: Both monitors run in parallel. Tapo uses RTSP, Logitech uses USB `/dev/video0`.
- **Storage**: Logitech uses a separate temporary folder (`recordings_usb_temp`) and a dedicated Google Drive folder ID (`14yBPCZvjrztIqxI5l-ckgZYkC7D0ZTdS`).
- **Bowl Monitoring**: The Logitech camera *will* monitor the bowl and report its status in `/status`, but **Telegram alerts are suppressed** for this camera to prevent unnecessary noise.

## 6. Auto-Cleanup
The cleanup script (`/home/pi5/Feeder/fair-feeder/sync_cleanup.sh`) has been updated to automatically purge old videos from both cameras and empty the system trash bin every night at 3:00 AM.
