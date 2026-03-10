# Setup Instructions After Pulling from Git

Since sensitive information (IP addresses, passwords) has been removed for security before committing to Git, you need to re-add them locally to get the system running.

## 1. Environment Variables (Recommended)

The easiest way is to set these in your local terminal session before running the scripts:

```powershell
$env:TAPO_IP = "192.168.1.246"
$env:TAPO_USER = "c210RTSP"
$env:TAPO_PASS = "<YOUR_CAMERA_PASSWORD>"
```

## 2. Hardcoding Locally (If not using env vars)

You can manually edit the following files and replace `<YOUR_CAMERA_IP>`, `<YOUR_CAMERA_USER>`, and `<YOUR_CAMERA_PASSWORD>` with your actual credentials:

- `motion_recorder.py`
- `check_onvif.py`
- `motion_recorder.ipynb` (Cell 2)

## 3. Verify Connection

After setting your credentials, run the check script to ensure everything is working:

```powershell
python check_onvif.py
```

## Security Reminder

**Never commit your real password to Git.** These files are already listed in `.gitignore` if they contain local compiled data or temporary results, but your main scripts should always use environment variables or local edits that aren't pushed back to a public repository.
