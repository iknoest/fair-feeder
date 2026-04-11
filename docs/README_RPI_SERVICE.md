# How to run the Cat Monitor 24/7 on your Raspberry Pi

This guide will walk you through setting up `motion_recorder.py` to run automatically as a background service whenever your Raspberry Pi turns on.

## 1. Move the Service File
Copy the `cat-monitor.service` file to the systemd directory:
```bash
sudo cp cat-monitor.service /etc/systemd/system/
```

## 2. Set Environment Variables
If you aren't storing `TAPO_IP`, `TAPO_USER`, and `TAPO_PASS` in a local `config.py` on the Pi (for security reasons), you can add them directly to the service file.

To edit it securely:
```bash
sudo nano /etc/systemd/system/cat-monitor.service
```
Add these lines in the `[Service]` section:
```ini
Environment="TAPO_IP=192.168.1.50"
Environment="TAPO_USER=YOUR_CAMERA_USER"
Environment="TAPO_PASS=YOUR_CAMERA_PASSWORD"
```
*(Save and exit `nano` by pressing `Ctrl+X`, then `Y`, then `Enter`)*

## 3. Reload and Enable the Service
Tell systemd about the new file, then set it to launch at boot:
```bash
sudo systemctl daemon-reload
sudo systemctl enable cat-monitor.service
```

## 4. Start the Service
You can start it right now without having to reboot:
```bash
sudo systemctl start cat-monitor.service
```

## 5. Check the Status & View Logs
Because it runs in the background, you won't see the usual terminal outputs. To look at what the script is doing (the `print()` statements):
```bash
# See live logs
journalctl -u cat-monitor.service -f

# Check the service status
sudo systemctl status cat-monitor.service
```

## 6. To Stop or Restart
```bash
sudo systemctl stop cat-monitor.service
sudo systemctl restart cat-monitor.service
```
