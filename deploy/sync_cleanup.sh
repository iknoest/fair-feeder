#!/bin/bash
# /home/pi5/Feeder/fair-feeder/sync_cleanup.sh
# Deletes local MP4s older than 3 days. Google Drive files are unaffected.

# 1. Purge Tapo sync folder
find /home/pi5/Pictures/gdrive-randomdice-sync/ -name "*.mp4" -type f -mtime +3 -exec rm -f {} \;

# 2. Purge Logitech sync folder
find /home/pi5/Pictures/usb-camera-sync/ -name "*.mp4" -type f -mtime +3 -exec rm -f {} \;

# 3. Clean up any stuck files in temporary folders (older than 1 day)
find /home/pi5/Feeder/fair-feeder/recordings_temp/ -name "*.mp4" -type f -mtime +1 -exec rm -f {} \;
find /home/pi5/Feeder/fair-feeder/recordings_usb_temp/ -name "*.mp4" -type f -mtime +1 -exec rm -f {} \;

# 4. Empty the GUI Trash bin (just in case files were deleted via Desktop)
rm -rf /home/pi5/.local/share/Trash/files/*
rm -rf /home/pi5/.local/share/Trash/info/*

echo "[$(date)] Auto-purged old local videos and emptied trash." >> /home/pi5/Feeder/fair-feeder/cleanup.log
