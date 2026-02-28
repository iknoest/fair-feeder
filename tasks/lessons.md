# Lessons Learned

After every correction, this file is updated to prevent repeating mistakes.
Review this file at the start of each session.

---

| # | Lesson | Context |
|---|--------|---------|
| 1 | `.ipynb` files cannot be edited with editor tools — use a Python script to modify the JSON | Tried to use `multi_replace_file_content` on notebook, got blocked |
| 2 | Never hardcode camera credentials in fallback defaults — use `<PLACEHOLDER>` patterns | Tapo password was committed in source via `os.getenv('TAPO_PASS', 'real_password')` |
| 3 | `create_pullpoint_manager()` requires `subscription_lost_callback` keyword argument | ONVIF library TypeError when `subscription_lost_callback` was positional |
| 4 | Tapo ONVIF sends motion events in bursts with 1-3s gaps — never use instantaneous `motion_detected` flag for stop decisions | Recording kept stopping and restarting during continuous motion |
| 5 | Windows PowerShell redirect (`>`) produces UTF-16LE files — use Python `open()` with explicit `encoding='utf-8'` instead | `view_file` tool rejected UTF-16LE text files |
| 6 | Synchronous external commands (like file uploads) block the Python event loop and cause dropped frames | Video uploads prevented the script from resetting for the next capture; used `subprocess.Popen` instead for fire-and-forget sync |
| 7 | Cross-platform hardcoded paths break deployment — use `platform.system()` checks to conditionally set paths | Windows uses drive letters (`H:\`), Pi uses Linux mount points (`/home/pi5/...`) |
