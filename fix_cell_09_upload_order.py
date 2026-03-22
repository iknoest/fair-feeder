import json, sys
sys.stdout.reconfigure(encoding="utf-8")

NB = "morning_report.ipynb"

NEW_CELL9_SOURCE = r"""# ── Append to feeding_log.csv on Drive + upload annotated video ─────
# NOTE: This cell must run AFTER Phase 3 (output-and-telegram).
# video_summaries and summary are populated by Phase 2/3.
import csv, os
from datetime import date
from pathlib import Path as _Path

_LOG_FILE = _Path('feeding_log.csv')
_FIELDS = ['date', 'dan_kibble', 'sanbo_kibble', 'hand_feeding', 'compensation', 'video_count']

_vcount = len(video_summaries) if 'video_summaries' in globals() else 0
# Use last summary if multiple events; or empty dict if no results yet
_s = video_results[-1]['summary'] if ('video_results' in globals() and video_results) else {}

_row = {
    'date': str(date.today()),
    'dan_kibble': _s.get('dan_kibble_eaten', 0),
    'sanbo_kibble': _s.get('sanbo_kibble_eaten', 0),
    'hand_feeding': sum(e['kibble_added'] for e in _s.get('hand_episodes', [])),
    'compensation': _s.get('sanbo_kibble_eaten', 0),
    'video_count': _vcount,
}

if RUNNING_IN_CI:
    from google.oauth2 import service_account as _sa
    from googleapiclient.discovery import build as _build
    from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
    import json as _json2

    _key2 = _json2.loads(os.environ['GDRIVE_SERVICE_ACCOUNT_KEY'])
    _creds2 = _sa.Credentials.from_service_account_info(
        _key2, scopes=['https://www.googleapis.com/auth/drive']
    )
    _drive2 = _build('drive', 'v3', credentials=_creds2)
    _out_id = os.environ['GDRIVE_OUTPUT_FOLDER_ID']

    # ── CSV: download existing, append row, re-upload ────────────
    _existing_csv = _drive2.files().list(
        q=f"'{_out_id}' in parents and name='feeding_log.csv' and trashed=false",
        fields='files(id)'
    ).execute().get('files', [])

    if _existing_csv:
        _req = _drive2.files().get_media(fileId=_existing_csv[0]['id'])
        with open(_LOG_FILE, 'wb') as _fh:
            _dl = MediaIoBaseDownload(_fh, _req)
            _done = False
            while not _done:
                _, _done = _dl.next_chunk()
        print(f'Downloaded existing feeding_log.csv ({_LOG_FILE.stat().st_size} bytes)')

    _write_header = not _LOG_FILE.exists()
    with open(_LOG_FILE, 'a', newline='') as _f:
        _writer = csv.DictWriter(_f, fieldnames=_FIELDS)
        if _write_header:
            _writer.writeheader()
        _writer.writerow(_row)
    print(f'Logged: {_row}')

    _media_csv = MediaFileUpload(str(_LOG_FILE), mimetype='text/csv')
    if _existing_csv:
        _drive2.files().update(fileId=_existing_csv[0]['id'], media_body=_media_csv).execute()
        print('Updated feeding_log.csv on Drive')
    else:
        try:
            _drive2.files().create(
                body={'name': 'feeding_log.csv', 'parents': [_out_id]},
                media_body=_media_csv
            ).execute()
            print('Created feeding_log.csv on Drive')
        except Exception as _csv_err:
            print(f'⚠️ Could not create feeding_log.csv: {_csv_err}')

    # ── Annotated video upload ────────────────────────────────────
    # video_paths was set by Cell 1; annotated video is in OUTPUT_DIR
    # NOTE: files().create() may fail with 403 storageQuotaExceeded if the SA's
    # personal quota is exhausted (Issue #21). The except clause catches this
    # gracefully — the video is still sent via Telegram in Phase 3.
    if 'video_paths' in globals() and video_paths:
        for _vp in video_paths:
            _vid_stem = _Path(str(_vp)).stem
            _out_vid = os.path.join(OUTPUT_DIR, f"{_vid_stem}_annotated.mp4")
            if os.path.exists(_out_vid):
                _vname = os.path.basename(_out_vid)
                _vmedia = MediaFileUpload(_out_vid, mimetype='video/mp4')
                try:
                    _drive2.files().create(
                        body={'name': _vname, 'parents': [_out_id]},
                        media_body=_vmedia
                    ).execute()
                    print(f'Uploaded {_vname} to Drive output folder')
                except Exception as _ev:
                    print(f'⚠️ Drive video upload failed for {_vname}: {_ev}')
            else:
                print(f'⚠️ Annotated video not found: {_out_vid}')
else:
    _write_header = not _LOG_FILE.exists()
    with open(_LOG_FILE, 'a', newline='') as _f:
        _writer = csv.DictWriter(_f, fieldnames=_FIELDS)
        if _write_header:
            _writer.writeheader()
        _writer.writerow(_row)
    print(f'Logged locally: {_row}')
"""

with open(NB, encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

# Rewrite cell 9 source
cells[9]["source"] = [line + "\n" for line in NEW_CELL9_SOURCE.splitlines()]
cells[9]["source"][-1] = cells[9]["source"][-1].rstrip("\n")

# Move cell 9 to after Phase 3:
# We need to find Phase 3's current position by looking for its content marker
phase3_idx = None
for i, c in enumerate(cells):
    src = "".join(c["source"])
    if "Phase 3" in src and ("Output" in src or "Telegram" in src or "output-and-telegram" in src):
        phase3_idx = i
        break

if phase3_idx is None:
    raise RuntimeError("Could not find Phase 3 cell")

print(f"Phase 3 found at cell index {phase3_idx}")

# Remove cell 9 from current position
csv_cell = cells.pop(9)
# After removal, Phase 3 is now at phase3_idx - 1 (since we removed before it)
# We want to insert AFTER Phase 3, so insert at phase3_idx (which is now the Phase 3 position after pop)
cells.insert(phase3_idx, csv_cell)

nb["cells"] = cells

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")

print("✅ Cell 9 rewritten: correct OUTPUT_DIR path, reads summary from video_results")
print("✅ CSV+upload cell moved to after Phase 3")
