# Project Map

Fair Feeder is a computer-vision cat feeding monitor for Dan and Sanbo. It uses a
Tapo C210 IR camera, YOLOv11, OCR of the burned-in Tapo timestamp, and Telegram
reports to answer whether Dan ate enough and whether Sanbo stole food.

## Product Scope

Goals:
- Attribute kibble eaten per cat from recorded feeding videos.
- Detect Dan_hand feeding episodes with timestamps.
- Send structured Telegram reports with snapshots and annotated video.
- Maintain model quality through Roboflow flagging and retraining.

In scope:
- YOLOv11 detection for `Dan`, `Sanbo`, `Dan_hand`, `Bowl`, and `Kibble`.
- OCR timestamp extraction from the Tapo OSD.
- Phase-based eating attribution by bowl-overlap time.
- Snapshot capture for Sanbo arrival, Dan_hand, and kibble-dispensed moments.
- Boxes-only annotated video output.
- Multi-camera support (Tapo C210 + Logitech C925e USB).

Out of scope:
- Real-time live alerting; the report pipeline batch-processes recorded clips.
- Web dashboard or feeding scheduler.

## Root Files With Hard Dependencies

These files stay at repository root because notebooks, workflows, service files, or
deployment paths depend on them:

- `AGENTS.md` - compact agent bootstrap and reference index.
- `CLAUDE.md` - compact Claude bootstrap and reference index.
- `README.md` - user-facing overview.
- `requirements.txt` - core Python dependencies.
- `data.yaml` - YOLO dataset config.
- `config.py` - camera and detection settings imported by `motion_recorder.py`.
- `motion_recorder.py` - Pi daemon; path is hardcoded in systemd/SCP flows.
- `morning_report.ipynb` - CI notebook; path is hardcoded in GitHub Actions.
- `flagging.py` - imported by `morning_report.ipynb`.
- `roboflow_upload.py` - imported by `morning_report.ipynb`.
- `schedule_log.py` - imported by `morning_report.ipynb`.

Rule: if a file is imported by `morning_report.ipynb` or `motion_recorder.py`, or
its path is hardcoded in a config, workflow, or service, keep it at root.

## Repository Areas

- `notebooks/` - interactive training and review notebooks.
- `scripts/` - one-off dataset, training, and debugging tools.
- `deploy/` - Pi deployment files such as `cat-monitor.service` and `usb-monitor.service`.
- `tests/` - unit and regression tests.
- `docs/` - product, runbook, model, and agent documentation (includes `USB_CAMERA_GUIDE.md`).
- `tasks/` - project tracking; `todo.md` and `lessons.md`.

## Core Dependencies

- `ultralytics` - YOLOv11 training and inference.
- `roboflow` - dataset download and flagged-frame upload.
- `easyocr` - Tapo timestamp OCR.
- `opencv-python` - video/image processing and RTSP.
- `onvif-zeep-async` - ONVIF camera events.
- `infisicalsdk` - pip package; import remains `from infisical_sdk`.

## External Services

- Roboflow - API key via Infisical.
- Infisical - `INFISICAL_ID`, `INFISICAL_SECRET`, `INFISICAL_PROJECT_ID`.
- Telegram - `TelegramBotToken`, `TelegramChatId`, plus `TELEGRAM_CHAT_ID` in CI.
- Google Drive - user mount in Colab; service account in CI.
- Tapo C210 - `TAPO_IP`, `TAPO_USER`, `TAPO_PASS` via Infisical or env vars.

