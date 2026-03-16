# How to Access Your Raspberry Pi 5 via SSH

Secure Shell (SSH) allows you to securely access your Raspberry Pi's terminal remotely from your Windows computer.

## Connecting from Windows

### Option 1: Using PowerShell (Recommended)
Windows 10 and 11 come with an SSH client built-in.

1. **Open PowerShell or Command Prompt**.
2. **Type the connection command**:
   ```bash
   ssh pi5@pi5
   ```
   *Note: if your Pi hostname is different, you can also use your Pi's local IP address (e.g., `ssh pi5@192.168.1.x`).*
3. **Accept the Host Key** (First time only): 
   If it asks `Are you sure you want to continue connecting (yes/no)?`, type `yes` and press Enter.
4. **Enter your password**: 
   Type your Pi 5 user password. Security feature: **Characters will not appear on the screen as you type.** Just type it carefully and press Enter.

### Option 2: Using VS Code (Best for Editing Files)
If you want to edit code on the Pi directly, use the "Remote - SSH" extension in VS Code.

1. Install the **Remote - SSH** extension by Microsoft in VS Code.
2. Press `F1` (or `Ctrl+Shift+P`) and type `Remote-SSH: Connect to Host...`
3. Select `Add New SSH Host...`
4. Enter `ssh pi5@pi5` or `ssh pi5@<IP_ADDRESS>`.
5. Select the configuration file to update (usually `C:\Users\YourUser\.ssh\config`).
6. Click **Connect** in the bottom right corner.
7. Enter your password when prompted.
8. Once connected, open the project folder (e.g., `/home/pi5/Feeder/fair-feeder`).

## Common Troubleshooting

*   **`ssh: Could not resolve hostname pi5`**: Your Windows PC cannot find the Pi by its hostname. Use the Pi's local IP address instead (find it in your router's admin page).
*   **`Host key verification failed`**: The SSH key for the Pi has changed (e.g., if you reinstalled the OS). To fix this, open `C:\Users\YourUser\.ssh\known_hosts` in Notepad, delete the line containing `pi5` or the IP address, save the file, and try SSHing again. OR use the following command: `ssh -o StrictHostKeyChecking=no pi5@pi5` to bypass the check.
*   **`Connection timed out`**: The Pi might be turned off, disconnected from Wi-Fi, or you are on a different network (e.g., Pi is on 2.4GHz, PC is on 5GHz, and your router isolates them).

## Useful SSH Commands Once Connected

*   **View live logs of the cat monitor service:**
    ```bash
    journalctl -u cat-monitor.service -f
    ```
*   **Stop the monitor service:**
    ```bash
    sudo systemctl stop cat-monitor.service
    ```
*   **Start the monitor service:**
    ```bash
    sudo systemctl start cat-monitor.service
    ```
*   **Check disk space (to ensure videos aren't filling up the SD card):**
    ```bash
    df -h
    ```
