# Phase B Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make Fair Feeder reliable, observable, and self-reporting — fixing bugs, adding two-way Telegram health check, and automating the morning kibble report via GitHub Actions.

**Architecture:** Bug fixes touch existing files minimally. Telegram health check adds one new class to `motion_recorder.py`. GitHub Actions workflow runs the existing `smoketest.ipynb` via papermill after downloading new Drive videos via a Google service account. A feeding log CSV grows over time for weekly digests.

**Tech Stack:** Python 3.11, opencv-python, ultralytics YOLOv8n, Telegram Bot API (polling), GitHub Actions (cron), papermill, google-auth + google-api-python-client, requests.

---

## Task 1: Fix hardcoded camera password

**Files:**
- Modify: `config.py:48`

**Step 1: Verify the problem**

```bash
grep -n "mkeihfCCP" config.py
```
Expected: line 48 shows the real password as a fallback default.

**Step 2: Fix it**

In `config.py`, change line 48 from:
```python
os.getenv('TAPO_PASS', '<YOUR_CAMERA_PASSWORD>'),
```
to:
```python
os.getenv('TAPO_PASS', '<YOUR_CAMERA_PASSWORD>'),
```

**Step 3: Verify fix**

```bash
grep -n "mkeihfCCP" config.py
```
Expected: no output (grep finds nothing).

**Step 4: Commit**

```bash
git add config.py
git commit -m "fix: remove hardcoded camera password from config.py fallback"
```

---

## Task 2: Fix RTSP reconnect to use TCP

**Files:**
- Modify: `motion_recorder.py:257`

**Step 1: Verify the problem**

In `motion_recorder.py`, find the `_read_loop` method. Line ~257 reads:
```python
self.cap = cv2.VideoCapture(self.url)
```
This reconnects using UDP. The original TCP fix (line 230) is bypassed on reconnect.

**Step 2: Fix it**

Change line 257 from:
```python
self.cap = cv2.VideoCapture(self.url)
```
to:
```python
self.cap = cv2.VideoCapture(self.url + '?rtsp_transport=tcp')
```

**Step 3: Verify fix**

```bash
grep -n "rtsp_transport=tcp" motion_recorder.py
```
Expected: two lines (initial connect + reconnect both use TCP).

**Step 4: Commit**

```bash
git add motion_recorder.py
git commit -m "fix: use TCP transport on RTSP reconnect in motion_recorder.py"
```

---

## Task 3: Add TelegramCommandListener to motion_recorder.py

**Files:**
- Modify: `motion_recorder.py`
- Test: manual test from phone (no automated test possible for Telegram polling)

**Step 1: Add START_TIME global**

After the logging setup block (around line 72), add one line:
```python
START_TIME = time.time()
```

**Step 2: Add the TelegramCommandListener class**

Add this class after the `RecordingController` class (before `# ── MAIN ──`):

```python
class TelegramCommandListener:
    """Polls Telegram Bot API for commands and replies with live Pi stats."""
    POLL_INTERVAL = 2  # seconds between polls

    def __init__(self, bot_token, chat_id, controller):
        self.bot_token = bot_token
        self.chat_id = str(chat_id)
        self.controller = controller
        self.running = False
        self._last_update_id = 0

    def start(self):
        self.running = True
        threading.Thread(target=self._poll_loop, daemon=True).start()
        log.info('📲 Telegram command listener started')

    def _poll_loop(self):
        while self.running:
            try:
                self._check_messages()
            except Exception as e:
                log.warning(f'Telegram poll error: {e}')
            time.sleep(self.POLL_INTERVAL)

    def _check_messages(self):
        url = f'https://api.telegram.org/bot{self.bot_token}/getUpdates'
        r = requests.get(url, params={'offset': self._last_update_id + 1, 'timeout': 0}, timeout=5)
        if r.status_code != 200:
            return
        for update in r.json().get('result', []):
            self._last_update_id = update['update_id']
            msg = update.get('message', {})
            text = msg.get('text', '').strip()
            sender_id = str(msg.get('chat', {}).get('id', ''))
            if sender_id == self.chat_id and text.startswith('/'):
                self._handle_command(text.split()[0].lower())

    def _handle_command(self, cmd):
        dispatch = {'/status': self._cmd_status, '/lastclip': self._cmd_lastclip, '/help': self._cmd_help}
        handler = dispatch.get(cmd)
        if handler:
            handler()

    def _cmd_status(self):
        uptime_secs = int(time.time() - START_TIME)
        h, rem = divmod(uptime_secs, 3600)
        m = rem // 60
        try:
            disk = shutil.disk_usage(str(DRIVE_OUTPUT_DIR))
            free_gb = disk.free / (1024 ** 3)
            disk_str = f'{free_gb:.1f} GB free'
        except Exception:
            disk_str = 'unknown'
        last_motion = self.controller.listener.last_motion_time
        motion_str = last_motion.strftime('%H:%M:%S') if last_motion else 'none yet'
        self._send(
            f'✅ Fair Feeder Status\n'
            f'Uptime: {h}h {m}m\n'
            f'Clips saved: {self.controller.clips_saved}\n'
            f'Clips deleted: {self.controller.clips_deleted}\n'
            f'Drive space: {disk_str}\n'
            f'Last motion: {motion_str}'
        )

    def _cmd_lastclip(self):
        clips = sorted(DRIVE_OUTPUT_DIR.glob('*.mp4'), key=lambda p: p.stat().st_mtime)
        if not clips:
            self._send('No clips saved yet.')
            return
        latest = clips[-1]
        size_mb = latest.stat().st_size / (1024 * 1024)
        if size_mb > 50:
            self._send(f'Latest clip too large to send ({size_mb:.0f} MB): {latest.name}')
            return
        url = f'https://api.telegram.org/bot{self.bot_token}/sendVideo'
        try:
            with open(latest, 'rb') as f:
                requests.post(url, data={'chat_id': self.chat_id}, files={'video': f}, timeout=60)
        except Exception as e:
            self._send(f'Failed to send clip: {e}')

    def _cmd_help(self):
        self._send(
            '🐱 Fair Feeder Commands\n'
            '/status — uptime, clips, disk space\n'
            '/lastclip — send most recent cat clip\n'
            '/help — this message'
        )

    def _send(self, text):
        url = f'https://api.telegram.org/bot{self.bot_token}/sendMessage'
        try:
            requests.post(url, json={'chat_id': self.chat_id, 'text': text}, timeout=5)
        except Exception as e:
            log.warning(f'Telegram reply failed: {e}')
```

**Step 3: Wire into the main block**

In `if __name__ == "__main__":`, after `send_telegram_alert(...)` and before the main `while True:` loop, add:

```python
# Start Telegram command listener (two-way health check)
cmd_bot_token = os.getenv('TelegramBotToken')
cmd_chat_id = os.getenv('TelegramChatId')
if cmd_bot_token and cmd_chat_id:
    cmd_listener = TelegramCommandListener(cmd_bot_token, cmd_chat_id, controller)
    cmd_listener.start()
else:
    log.info('No Telegram credentials — command listener disabled')
```

> The credentials may already be loaded into env by the `.env` file via the
> systemd `EnvironmentFile`. If not, the listener silently disables itself.

**Step 4: Verify on Pi**

```bash
cd /home/pi5/Feeder/fair-feeder
source .venv/bin/activate
python motion_recorder.py
```

From your phone, send `/help` to the Telegram bot.
Expected reply: the help message within 2–4 seconds.

Send `/status`.
Expected reply: uptime, clip counts, disk space, last motion time.

**Step 5: Commit**

```bash
git add motion_recorder.py
git commit -m "feat: add TelegramCommandListener for two-way Pi health check (/status, /lastclip, /help)"
```

---

## Task 4: Google Service Account Setup (one-time, no code)

**This is infrastructure setup, not coding. Do it once.**

**Step 1: Create Google Cloud project**

1. Go to https://console.cloud.google.com
2. Create new project (name it e.g. `fair-feeder`)
3. Enable the **Google Drive API** for this project:
   - APIs & Services → Enable APIs → search "Google Drive API" → Enable

**Step 2: Create service account**

1. IAM & Admin → Service Accounts → Create Service Account
2. Name: `fair-feeder-automation`
3. Skip optional role steps — click Done
4. Click the service account → Keys → Add Key → JSON → Download

**Step 3: Share Drive folder with service account**

1. Open the downloaded JSON key, copy the `client_email` value
   (looks like: `fair-feeder-automation@fair-feeder-XXXXX.iam.gserviceaccount.com`)
2. In Google Drive, right-click your video upload folder
   (`TAPO_autoupload` or similar) → Share
3. Share with the `client_email` address, Editor access

**Step 4: Store secrets in GitHub Actions**

In your GitHub repo: Settings → Secrets and variables → Actions → New repository secret

| Secret name | Value |
|-------------|-------|
| `GDRIVE_SERVICE_ACCOUNT_KEY` | Full contents of the downloaded JSON key file |
| `GDRIVE_FOLDER_ID` | The folder ID from the Drive URL (the long string after `/folders/`) |
| `TELEGRAM_BOT_TOKEN` | Your Telegram bot token |
| `TELEGRAM_CHAT_ID` | Your Telegram chat ID |

**Step 5: Verify folder ID**

Open the Drive folder in browser. The URL looks like:
`https://drive.google.com/drive/folders/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74`
The folder ID is `1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74`.

---

## Task 5: Add environment detection to smoketest.ipynb

**Files:**
- Create: `patch_smoketest.py` (temporary script to modify notebook JSON)
- Modify: `smoketest.ipynb` (via the script — never edit .ipynb directly)

**Context:** `smoketest.ipynb` currently uses `drive.mount('/content/drive')` (Colab only).
GitHub Actions uses a service account instead. We need one cell that detects which
environment it's in and sets `SOURCE_DIR` accordingly.

**Step 1: Read the current first cell of smoketest.ipynb**

```bash
python -c "
import json
nb = json.load(open('smoketest.ipynb'))
for i, cell in enumerate(nb['cells'][:3]):
    print(f'--- Cell {i} ({cell[\"cell_type\"]}) ---')
    print(''.join(cell['source']))
    print()
"
```

Note which cell sets `SOURCE_DIR` and which cell mounts Drive. You need these cell indices.

**Step 2: Write the environment detection cell code**

The new cell to inject at the TOP of the notebook (before any Drive mount):

```python
# ── Environment Detection (Colab vs GitHub Actions) ──────────────────
import os

RUNNING_IN_CI = os.getenv('GITHUB_ACTIONS') == 'true'

if RUNNING_IN_CI:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    import io, json, pathlib, tempfile

    _key = json.loads(os.environ['GDRIVE_SERVICE_ACCOUNT_KEY'])
    _creds = service_account.Credentials.from_service_account_info(
        _key, scopes=['https://www.googleapis.com/auth/drive.readonly']
    )
    _drive = build('drive', 'v3', credentials=_creds)

    GDRIVE_FOLDER_ID = os.environ['GDRIVE_FOLDER_ID']
    CI_DOWNLOAD_DIR = pathlib.Path(tempfile.mkdtemp()) / 'videos'
    CI_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # Download all .mp4 files from Drive folder
    results = _drive.files().list(
        q=f"'{GDRIVE_FOLDER_ID}' in parents and name contains '.mp4' and trashed=false",
        fields='files(id, name, modifiedTime)',
        orderBy='modifiedTime desc'
    ).execute()
    _files = results.get('files', [])
    print(f'Found {len(_files)} video(s) on Drive')
    for _f in _files:
        _dest = CI_DOWNLOAD_DIR / _f['name']
        if not _dest.exists():
            _req = _drive.files().get_media(fileId=_f['id'])
            with open(_dest, 'wb') as _fh:
                _dl = MediaIoBaseDownload(_fh, _req)
                _done = False
                while not _done:
                    _, _done = _dl.next_chunk()
            print(f'  Downloaded: {_f["name"]}')

    SOURCE_DIR = CI_DOWNLOAD_DIR
    print(f'CI mode: SOURCE_DIR = {SOURCE_DIR}')
else:
    from google.colab import drive
    drive.mount('/content/drive')
    SOURCE_DIR = '/content/drive/MyDrive/Fun Project/Cat monitor/TAPO_autoupload'
    print(f'Colab mode: SOURCE_DIR = {SOURCE_DIR}')
```

**Step 3: Write the patch script**

Create `patch_smoketest.py`:

```python
"""
Patches smoketest.ipynb:
1. Inserts environment-detection cell at position 0
2. Removes the old drive.mount() cell (finds it by content)
3. Adds feeding_log.csv append cell before the final Telegram send cell
"""
import json
import re
from pathlib import Path

NB_PATH = Path('smoketest.ipynb')
nb = json.loads(NB_PATH.read_text(encoding='utf-8'))

ENV_CELL_SOURCE = r"""# ── Environment Detection (Colab vs GitHub Actions) ──────────────────
import os

RUNNING_IN_CI = os.getenv('GITHUB_ACTIONS') == 'true'

if RUNNING_IN_CI:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    import io, json, pathlib, tempfile

    _key = json.loads(os.environ['GDRIVE_SERVICE_ACCOUNT_KEY'])
    _creds = service_account.Credentials.from_service_account_info(
        _key, scopes=['https://www.googleapis.com/auth/drive.readonly']
    )
    _drive = build('drive', 'v3', credentials=_creds)

    GDRIVE_FOLDER_ID = os.environ['GDRIVE_FOLDER_ID']
    CI_DOWNLOAD_DIR = pathlib.Path(tempfile.mkdtemp()) / 'videos'
    CI_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    results = _drive.files().list(
        q=f"'{GDRIVE_FOLDER_ID}' in parents and name contains '.mp4' and trashed=false",
        fields='files(id, name, modifiedTime)',
        orderBy='modifiedTime desc'
    ).execute()
    _files = results.get('files', [])
    print(f'Found {len(_files)} video(s) on Drive')
    for _f in _files:
        _dest = CI_DOWNLOAD_DIR / _f['name']
        if not _dest.exists():
            _req = _drive.files().get_media(fileId=_f['id'])
            with open(_dest, 'wb') as _fh:
                _dl = MediaIoBaseDownload(_fh, _req)
                _done = False
                while not _done:
                    _, _done = _dl.next_chunk()
            print(f'  Downloaded: {_f["name"]}')

    SOURCE_DIR = CI_DOWNLOAD_DIR
    print(f'CI mode: SOURCE_DIR = {SOURCE_DIR}')
else:
    from google.colab import drive
    drive.mount('/content/drive')
    SOURCE_DIR = '/content/drive/MyDrive/Fun Project/Cat monitor/TAPO_autoupload'
    print(f'Colab mode: SOURCE_DIR = {SOURCE_DIR}')
""".splitlines(keepends=True)

CSV_CELL_SOURCE = r"""# ── Append to feeding_log.csv on Drive ───────────────────────────────
import csv
from datetime import date

LOG_FILE = Path('feeding_log.csv')
FIELDNAMES = ['date', 'dan_kibble', 'sanbo_kibble', 'hand_feeding', 'compensation', 'video_count']

row = {
    'date': str(date.today()),
    'dan_kibble': getattr(tracker, 'dan_kibble_total', 0),
    'sanbo_kibble': getattr(tracker, 'sanbo_kibble_total', 0),
    'hand_feeding': getattr(tracker, 'hand_feeding_count', 0),
    'compensation': getattr(tracker, 'compensation', 0),
    'video_count': len(video_summaries) if 'video_summaries' in dir() else 0,
}

write_header = not LOG_FILE.exists()
with open(LOG_FILE, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    if write_header:
        writer.writeheader()
    writer.writerow(row)

print(f'Appended to {LOG_FILE}: {row}')

# Upload CSV back to Drive if running in CI
if RUNNING_IN_CI:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    import json as _json

    _key = _json.loads(os.environ['GDRIVE_SERVICE_ACCOUNT_KEY'])
    _creds = service_account.Credentials.from_service_account_info(
        _key, scopes=['https://www.googleapis.com/auth/drive']
    )
    _drive_rw = build('drive', 'v3', credentials=_creds)

    # Check if CSV already exists on Drive
    _existing = _drive_rw.files().list(
        q=f"'{os.environ['GDRIVE_FOLDER_ID']}' in parents and name='feeding_log.csv'",
        fields='files(id)'
    ).execute().get('files', [])

    _media = MediaFileUpload(str(LOG_FILE), mimetype='text/csv')
    if _existing:
        _drive_rw.files().update(fileId=_existing[0]['id'], media_body=_media).execute()
        print('Updated feeding_log.csv on Drive')
    else:
        _drive_rw.files().create(
            body={'name': 'feeding_log.csv', 'parents': [os.environ['GDRIVE_FOLDER_ID']]},
            media_body=_media
        ).execute()
        print('Created feeding_log.csv on Drive')
""".splitlines(keepends=True)

def make_code_cell(source_lines):
    return {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': source_lines,
    }

# 1. Remove old drive.mount() cell if present
nb['cells'] = [
    c for c in nb['cells']
    if not any('drive.mount' in line for line in c.get('source', []))
]

# 2. Insert env detection cell at position 0
nb['cells'].insert(0, make_code_cell(ENV_CELL_SOURCE))

# 3. Find the Telegram send cell and insert CSV cell before it
telegram_idx = next(
    (i for i, c in enumerate(nb['cells'])
     if any('sendMessage' in line or 'output-and-telegram' in str(c.get('metadata', {}))
            for line in c.get('source', []))),
    len(nb['cells']) - 1
)
nb['cells'].insert(telegram_idx, make_code_cell(CSV_CELL_SOURCE))

NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
print(f'Patched {NB_PATH} — {len(nb["cells"])} cells total')
print(f'  + Env detection cell at position 0')
print(f'  + CSV append cell at position {telegram_idx}')
```

**Step 4: Run the patch script**

```bash
cd C:\Users\AVAVAVA\.gemini\antigravity\scratch\fair-feeder
python patch_smoketest.py
```

Expected output:
```
Patched smoketest.ipynb — N cells total
  + Env detection cell at position 0
  + CSV append cell at position N-2
```

**Step 5: Verify in Colab**

Open `smoketest.ipynb` in Colab. The first cell should show the environment detection block. Run it — it should fall into the `else` branch and call `drive.mount()` exactly as before.

**Step 6: Clean up patch script and commit**

```bash
del patch_smoketest.py
git add smoketest.ipynb
git commit -m "feat: add CI environment detection and feeding_log.csv append to smoketest.ipynb"
```

---

## Task 6: Create GitHub Actions workflow

**Files:**
- Create: `.github/workflows/morning-report.yml`

**Step 1: Create the workflow file**

```yaml
name: Fair Feeder Morning Report

on:
  schedule:
    - cron: '45 23 * * *'   # 06:45 Thailand time (UTC+7)
  workflow_dispatch:          # Manual trigger for testing

jobs:
  morning-report:
    runs-on: ubuntu-latest
    timeout-minutes: 60

    steps:
      - name: Checkout repo
        uses: actions/checkout@v4

      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install dependencies
        run: |
          pip install \
            ultralytics \
            easyocr \
            opencv-python-headless \
            papermill \
            ipykernel \
            google-auth \
            google-api-python-client \
            requests \
            nbformat

      - name: Run analysis notebook
        env:
          GITHUB_ACTIONS: 'true'
          GDRIVE_SERVICE_ACCOUNT_KEY: ${{ secrets.GDRIVE_SERVICE_ACCOUNT_KEY }}
          GDRIVE_FOLDER_ID: ${{ secrets.GDRIVE_FOLDER_ID }}
          TelegramBotToken: ${{ secrets.TELEGRAM_BOT_TOKEN }}
          TelegramChatId: ${{ secrets.TELEGRAM_CHAT_ID }}
        run: |
          papermill smoketest.ipynb /tmp/smoketest_output.ipynb \
            --no-progress-bar \
            --kernel python3

      - name: Upload output notebook (always, for debugging)
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: smoketest-output-${{ github.run_id }}
          path: /tmp/smoketest_output.ipynb
          retention-days: 7
```

**Step 2: Test the workflow manually**

Push to GitHub, then go to:
Actions → Fair Feeder Morning Report → Run workflow → Run workflow

Watch the logs. Expected: notebook runs, Telegram message received.

If it fails, download the `smoketest-output` artifact — it shows which cell failed and why.

**Step 3: Commit**

```bash
git add .github/workflows/morning-report.yml
git commit -m "feat: add GitHub Actions morning report workflow (cron 6:45am Thailand)"
```

---

## Task 7: Add weekly digest job

**Files:**
- Create: `weekly_digest.py`
- Modify: `.github/workflows/morning-report.yml`

**Step 1: Write weekly_digest.py**

```python
"""
Reads feeding_log.csv from Google Drive and sends a weekly Telegram digest.
Run via GitHub Actions every Monday 7:00am Thailand time.
"""
import csv
import json
import os
import tempfile
import requests
from pathlib import Path
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload


def download_csv(drive_service, folder_id):
    files = drive_service.files().list(
        q=f"'{folder_id}' in parents and name='feeding_log.csv'",
        fields='files(id, name)'
    ).execute().get('files', [])
    if not files:
        return None
    tmp = Path(tempfile.mkdtemp()) / 'feeding_log.csv'
    req = drive_service.files().get_media(fileId=files[0]['id'])
    with open(tmp, 'wb') as fh:
        dl = MediaIoBaseDownload(fh, req)
        done = False
        while not done:
            _, done = dl.next_chunk()
    return tmp


def send_telegram(token, chat_id, text):
    requests.post(
        f'https://api.telegram.org/bot{token}/sendMessage',
        json={'chat_id': chat_id, 'text': text},
        timeout=10
    )


def build_digest(rows):
    if not rows:
        return 'No feeding data this week.'
    dan = [float(r.get('dan_kibble', 0)) for r in rows]
    sanbo = [float(r.get('sanbo_kibble', 0)) for r in rows]
    hand = sum(int(r.get('hand_feeding', 0)) for r in rows)
    dan_avg = sum(dan) / len(dan)
    sanbo_avg = sum(sanbo) / len(sanbo)
    min_day = min(rows, key=lambda r: float(r.get('dan_kibble', 0)))
    dates = f"{rows[0]['date']} – {rows[-1]['date']}"
    return (
        f'── Week of {dates} ──\n'
        f'Dan avg kibble:   {dan_avg:.1f}/day\n'
        f'Sanbo avg kibble: {sanbo_avg:.1f}/day\n'
        f'Hand-feeding:     {hand}x this week\n'
        f'Lowest day:       {min_day["date"]} (Dan ate {min_day["dan_kibble"]})'
    )


if __name__ == '__main__':
    key = json.loads(os.environ['GDRIVE_SERVICE_ACCOUNT_KEY'])
    creds = service_account.Credentials.from_service_account_info(
        key, scopes=['https://www.googleapis.com/auth/drive.readonly']
    )
    drive = build('drive', 'v3', credentials=creds)
    csv_path = download_csv(drive, os.environ['GDRIVE_FOLDER_ID'])

    if not csv_path:
        msg = 'No feeding_log.csv found on Drive yet.'
    else:
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        last_7 = rows[-7:]
        msg = build_digest(last_7)

    send_telegram(os.environ['TelegramBotToken'], os.environ['TelegramChatId'], msg)
    print('Weekly digest sent.')
```

**Step 2: Add weekly job to workflow**

Append to `.github/workflows/morning-report.yml`:

```yaml
  weekly-digest:
    runs-on: ubuntu-latest
    if: github.event_name == 'schedule'

    steps:
      - name: Checkout repo
        uses: actions/checkout@v4

      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          cache: 'pip'

      - name: Install dependencies
        run: pip install google-auth google-api-python-client requests

      - name: Send weekly digest (Mondays only)
        if: ${{ fromJson('{"0":"Sun","1":"Mon","2":"Tue","3":"Wed","4":"Thu","5":"Fri","6":"Sat"}')[format('{0:dddd}', github.event.schedule)] == 'Mon' || github.event_name == 'workflow_dispatch' }}
        env:
          GDRIVE_SERVICE_ACCOUNT_KEY: ${{ secrets.GDRIVE_SERVICE_ACCOUNT_KEY }}
          GDRIVE_FOLDER_ID: ${{ secrets.GDRIVE_FOLDER_ID }}
          TelegramBotToken: ${{ secrets.TELEGRAM_BOT_TOKEN }}
          TelegramChatId: ${{ secrets.TELEGRAM_CHAT_ID }}
        run: python weekly_digest.py
```

> Note: GitHub Actions doesn't have a native "day of week" check in job-level `if`.
> Simpler alternative: add a second cron entry just for Monday:
> ```yaml
> - cron: '0 0 * * 1'  # 7:00am Thailand Monday
> ```
> and split into two separate workflow files. Use whichever feels cleaner.

**Step 3: Test weekly digest manually**

```bash
# Set env vars locally (Windows PowerShell)
$env:GDRIVE_SERVICE_ACCOUNT_KEY = (Get-Content gdrive_key.json -Raw)
$env:GDRIVE_FOLDER_ID = "your_folder_id"
$env:TelegramBotToken = "your_token"
$env:TelegramChatId = "your_chat_id"
python weekly_digest.py
```

Expected: Telegram message received with week summary.

**Step 4: Commit**

```bash
git add weekly_digest.py .github/workflows/morning-report.yml
git commit -m "feat: add weekly feeding digest job and weekly_digest.py"
```

---

## Verification Checklist

After all tasks complete:

- [ ] `grep -n "mkeihfCCP" config.py` returns nothing
- [ ] `grep -n "rtsp_transport=tcp" motion_recorder.py` returns 2 lines
- [ ] Send `/status` to Telegram bot → Pi replies within 4 seconds
- [ ] Send `/lastclip` → most recent clip arrives in Telegram
- [ ] GitHub Actions workflow runs via `workflow_dispatch` → morning report received in Telegram
- [ ] `feeding_log.csv` appears on Drive after first successful run
- [ ] Monday: weekly digest Telegram message received
