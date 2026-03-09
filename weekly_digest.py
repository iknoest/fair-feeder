"""
Fair Feeder Weekly Digest
Reads feeding_log.csv from Google Drive and sends a weekly Telegram summary.
Run via GitHub Actions every Monday morning.

Usage (local test):
    set GDRIVE_SERVICE_ACCOUNT_KEY=<json contents>
    set GDRIVE_OUTPUT_FOLDER_ID=<folder id>
    set TelegramBotToken=<token>
    set TelegramChatId=<chat id>
    python weekly_digest.py
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


def get_drive_service():
    key = json.loads(os.environ['GDRIVE_SERVICE_ACCOUNT_KEY'])
    creds = service_account.Credentials.from_service_account_info(
        key, scopes=['https://www.googleapis.com/auth/drive.readonly']
    )
    return build('drive', 'v3', credentials=creds)


def download_csv(drive, folder_id):
    files = drive.files().list(
        q=f"'{folder_id}' in parents and name='feeding_log.csv' and trashed=false",
        fields='files(id, name)'
    ).execute().get('files', [])
    if not files:
        return None
    tmp = Path(tempfile.mkdtemp()) / 'feeding_log.csv'
    req = drive.files().get_media(fileId=files[0]['id'])
    with open(tmp, 'wb') as fh:
        dl = MediaIoBaseDownload(fh, req)
        done = False
        while not done:
            _, done = dl.next_chunk()
    return tmp


def build_digest(rows):
    if not rows:
        return 'No feeding data this week.'
    dan = [float(r.get('dan_kibble', 0) or 0) for r in rows]
    sanbo = [float(r.get('sanbo_kibble', 0) or 0) for r in rows]
    hand = sum(int(r.get('hand_feeding', 0) or 0) for r in rows)
    dan_avg = sum(dan) / len(dan)
    sanbo_avg = sum(sanbo) / len(sanbo)
    min_day = min(rows, key=lambda r: float(r.get('dan_kibble', 0) or 0))
    dates = f"{rows[0]['date']} - {rows[-1]['date']}"
    return (
        f'── Week of {dates} ──\n'
        f'Dan avg kibble:   {dan_avg:.1f}/day\n'
        f'Sanbo avg kibble: {sanbo_avg:.1f}/day\n'
        f'Hand-feeding:     {hand}x this week\n'
        f'Lowest day:       {min_day["date"]} (Dan ate {min_day["dan_kibble"]})'
    )


def send_telegram(text):
    token = os.environ['TelegramBotToken']
    chat_id = os.environ['TelegramChatId']
    requests.post(
        f'https://api.telegram.org/bot{token}/sendMessage',
        json={'chat_id': chat_id, 'text': text},
        timeout=10
    )


if __name__ == '__main__':
    drive = get_drive_service()
    folder_id = os.environ['GDRIVE_OUTPUT_FOLDER_ID']
    csv_path = download_csv(drive, folder_id)

    if not csv_path:
        msg = '⚠️ No feeding_log.csv found on Drive yet.'
    else:
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        last_7 = rows[-7:]
        msg = build_digest(last_7)

    send_telegram(msg)
    print(f'Sent: {msg}')
