# KICKOFF BRIEF: fair-feeder hybrid pivot — 2026-07-04 (rev 2, FINAL)

Supersedes rev 1 (same date). Amended after client supplied the IR hardware
boundary: the custom YOLO was trained on IR night-vision frames (dataset
`IR-kibble-14`, see docs/MODELS.md) and cannot be fully replaced by a VLM for
kibble counting. Verdict remains PIVOT per AUDIT-2026-07-04.md; scope revised.
The moving-robot direction remains forbidden until the 2026-08-15 re-audit.

## Correction of Rev-1 Assumptions (audit errata)

1. **Rev 1 assumed one uniform lighting domain.** Wrong: the feeding window is
   pre-dawn; the Tapo C210 delivers grayscale IR frames, and the model was
   trained specifically on that domain. VLM per-piece counting on IR
   monochrome kibble clusters is not viable.
2. **Rev 1 treated the model layer as one broken monolith.** Wrong: the domain
   shift is per-camera. v14 is *in-domain* on Tapo/IR and presumed still
   healthy there; it is only out-of-domain on the Logitech ambient frames.
3. **"Scrap custom YOLO entirely" was overbroad.** The correct cut is between
   *using* the frozen v14 model (zero-maintenance, keep) and *retraining* it
   via the manual-labeling flywheel (the actual time sink, stop). A frozen
   model is a sensor; only the flywheel was the backlog generator.

## Problem (Y)

Every morning, know whether Dan and Sanbo each ate properly and whether kibble
needs topping up — without the owner spending personal time watching video or
labeling training data.

## Purpose class

SOLVE.

## Build Verdict

ADAPT — keep the proven Pi → CI → Telegram skeleton and the frozen v14
IR-YOLO; add a hosted VLM layer for the tasks the frozen model can no longer
do (Logitech-domain identity and coarse bowl status). No retraining, no
labeling, no new custom models.

## Hybrid Architecture (the routing contract)

Route by frame domain, not by wish. The notebook already branches on
`CAMERA_NAME`; the hybrid extends that existing seam.

| Boundary | Camera / domain | Engine | Output | Authority |
|---|---|---|---|---|
| Kibble segmentation & counts | TAPO — IR grayscale (trained domain) | Frozen v14 IR-YOLO | per-piece kibble counts, Dan_hand episodes, snapshots | **Authoritative** for all kibble math |
| Cat identity + coarse bowl status | LOGITECH — ambient color (untrained domain) | Zero-shot VLM (Gemini/Claude class) on ~10–30 keyframes, with Dan/Sanbo reference photos in prompt | which cat, ate/didn't, bowl empty/low/half/full | Authoritative for Logitech report |
| Identity cross-check (optional, cheap) | TAPO keyframes | Same VLM call | confirms/flags YOLO's Dan-vs-Sanbo ID | Advisory only; disagreement → flag in report, never auto-override kibble counts |

Rules:
- Domain detection should be defensive: route on `CAMERA_NAME`, but also
  detect IR mode per-frame (grayscale check: R≈G≈B) so a Tapo daylight clip or
  future camera doesn't silently hit the wrong engine.
- The VLM never produces per-piece kibble numbers anywhere; the IR-YOLO never
  runs on ambient Logitech frames. No engine works out of domain.
- Keyframe selection on the VLM path uses stock COCO yolov8n `cat`/`bowl`
  gating (domain-robust, per lesson 51) — no custom model involved.
- The Roboflow flywheel (flagging review, retrain cycles, V15 comparison) is
  mothballed. `flagging.py`/`roboflow_upload.py` stay in the repo, disabled in
  CI, until the re-audit.

## Risk Verdict

YELLOW — viable with named changes:

- **NEW, and the biggest one — Logitech in the dark.** The C925e has no IR
  illumination and an IR-cut filter; in winter the 06:18–06:30 Amsterdam
  window is fully dark, so the VLM (like any camera consumer) may receive
  near-black frames. Named change: guarantee minimal ambient light during the
  feeding window (e.g. a small lamp on a smart-plug schedule), and make the
  spike test worst-case low-light frames explicitly. If unlit winter frames
  are unusable, the fallback is swapping the Logitech for a second IR camera
  — which would reopen the kibble-model question for that bowl.
- **Identity zero-shot:** requires Dan/Sanbo reference photos in-prompt and
  visually distinguishable cats; tested first in the spike.
- **Assumed, must verify:** v14 kibble accuracy on Tapo/IR is still sound
  (in-domain, so presumed yes — Phase 1 confirms with recent clips before
  anything is built on it).
- **Privacy/cost:** ~10–40 frames/day to a hosted API, est. $1–5/month
  (guess; small-tier vision model). Client has accepted the cloud path by
  proposing Gemini.
- **Secrets/CI plumbing:** GREEN — existing Infisical/GitHub-secrets
  machinery covers one more API key.

## Estimates (ranges, assumptions named)

- Phase 1 spike: 1–2 sessions — rough (assumes ≥5 recorded feeding windows
  per camera are retrievable from Drive).
- Phase 2 integration + shadow week: 2–4 sessions — rough (assumes the
  existing `CAMERA_NAME` branch is the only integration seam needed).
- Recurring: $1–5/month API — guess. Owner labeling time: 0 hours/month —
  firm as a design target; it is the success metric.

## MVP

One real morning where the Telegram reports are produced by the hybrid — Tapo
kibble counts from frozen IR-YOLO, Logitech identity/status from the VLM —
and match the owner's own judgment of the same videos.

Definition of Done:
- Measurable: on ≥5 historical Logitech feeding windows (including the
  darkest available), VLM identity matches human ground truth ≥90% and coarse
  bowl state matches ≥4/5 — verified by the spike notebook's scoring output.
- Measurable: on ≥3 recent Tapo IR windows, v14 kibble counts agree with
  human count within ±5 — verified the same way (confirms the frozen model is
  still trustworthy in-domain).
- Experiential: one end-to-end morning Telegram message per camera that the
  owner acts on without opening the video — artifact: the messages themselves.

## Constraints

- Must not preclude a future movable camera: engines consume routed frames
  and are camera-agnostic behind the routing table. (Constraint, not work.)
- Keep the verdict-first Telegram format (lesson 41) and all Kibble Snapshot
  invariants (AGENTS.md).
- v14 weights are frozen: no retraining, no new dataset versions, no Roboflow
  review obligations. Any future retrain proposal goes to re-audit first.
- No secrets in code; one new API key via existing secrets machinery.

## Phases (≤3)

1. **Split-domain spike (concrete):** one notebook that (a) pulls ≥3 recent
   Tapo IR feeding windows and scores frozen-v14 kibble counts against human
   counts, and (b) pulls ≥5 Logitech windows (worst lighting included),
   selects keyframes via COCO yolov8n gating, sends them with reference
   photos to the VLM, and scores identity + coarse-state verdicts against
   human ground truth. Output: a pass/fail table against the DoD thresholds.
2. Integration + shadow week: add the VLM engine behind the existing
   `CAMERA_NAME == 'LOGITECH'` branch in `morning_report.ipynb`; run both old
   and new Logitech verdicts side by side in the report for ~1 week.
3. Cutover: VLM verdict drives the Logitech Telegram report; flywheel steps
   removed from CI; this brief's boot section finalized against reality.

## Open Questions

None — rev-1's three questions were answered by the client's revision:
identity layer is broken and goes to the VLM; exact counts stay on the IR
boundary with coarse state on ambient; the cloud path is accepted (Gemini
named). Residual *assumptions* (v14 still accurate on Tapo; Logitech window
can be lit) are verified inside Phase 1, not deferred to the client.

---

## Stateless Agent Boot (for `agy` / Antigravity CLI and any fresh single-shot agent)

This section is the boot loader. A fresh agent with no skills, no memory, and
no conversation history must be able to act correctly from this file alone.
Invoke `agy` with a prompt that begins: **"Read BRIEF.md in the repo root,
follow its Stateless Agent Boot section, then do: <task>."**

### 1. What this system is (read this, skip nothing)

Two cameras watch two cat bowls at breakfast. A Raspberry Pi 5 records motion
clips and uploads them to Google Drive (`motion_recorder.py`, runs as
`cat-monitor.service` on the Pi — NOT here). Every morning GitHub Actions
runs `morning_report.ipynb` via papermill, once per camera
(`CAMERA_NAME=TAPO`, then `LOGITECH`), analyzes the clips, and sends a
Telegram verdict. Analysis is hybrid: TAPO frames (infrared) go through a
frozen custom YOLO model (v14) for exact kibble counts; LOGITECH frames
(ambient color) go through a hosted VLM for cat identity and coarse bowl
state. The routing table above is the contract; never run an engine out of
its domain.

### 2. Ground yourself with commands, not with trust

Docs go stale; command output doesn't. Before changing anything, run:

- `git log --oneline | head -20` — what actually happened recently.
- `gh run list --workflow=morning-report.yml --limit 5` — is production
  (GitHub Actions) healthy right now.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest tests/ -q` —
  local test baseline BEFORE your change, again after.
- `python -c "import json; json.load(open('morning_report.ipynb', encoding='utf-8'))"`
  — notebook JSON is valid (run after ANY notebook edit).

### 3. File map (one line each — read only what your task touches)

- `morning_report.ipynb` — the daily CI analysis pipeline; edit as JSON via a
  Python script, never as text; strip `\r` from cell source.
- `motion_recorder.py` — Pi daemon; edits are not live until SCP'd to
  `/home/pi5/Feeder/fair-feeder/`, compiled there, and
  `cat-monitor.service` restarted.
- `config.py`, `flagging.py`, `roboflow_upload.py`, `schedule_log.py` — root
  modules imported by the notebook/runtime; must stay at repo root.
  Flywheel modules (`flagging`, `roboflow_upload`) are mothballed — do not
  extend them.
- `.github/workflows/morning-report.yml` — CI schedule + secrets `env:`
  block; every secret the notebook reads MUST be listed there explicitly.
- `docs/agent/*.md` — deep context per subsystem; `AGENTS.md` lists which one
  to read for which file. `tasks/lessons.md` — 67 numbered mistakes already
  made; check it before debugging anything that feels surprising.
- `docs/MODELS.md` — model version ledger; v14 is deployed and FROZEN.

### 4. Invariants a stateless agent must not violate (the fatal subset)

- Class names are exactly: Dan, Sanbo, Dan_hand, Bowl, Kibble; map via
  `model.names`, never by assumed index.
- Never hardcode secrets, chat IDs, or folder IDs — env/Infisical only.
- `feeding_log.csv` on Drive is modified via `update()` only, never
  `create()` (service accounts have zero Drive quota).
- YOLO inference: `rect=True`, single-int `imgsz`; annotated video is
  boxes-only, `show_label=False`.
- Tapo OCR repair order: `\|:` → `:1` before `\|` → `1`.
- Stitch adjacent clips only when the gap ≤ 10 seconds.
- Telegram messages lead with the actionable verdict on line 1; scheduler
  heartbeat goes to GitHub summaries + CSV, never Telegram.
- CI changes are not live until committed AND pushed to `main`; Pi changes
  are not live until deployed to the Pi (see file map).
- No retraining, no Roboflow labeling tasks, no new model versions — frozen
  by audit; escalate to the human instead.

### 5. Task contract for single-shot work

State what you will change and which surface it lives on (CI notebook / Pi
daemon / repo module) → make the change → run the section-2 verification
commands relevant to that surface → report actual command output, not claims.
If a step needs credentials or hardware you don't have (Pi SSH, Drive,
Telegram), say so and stop; do not simulate success.

### Multi-agent conventions

Claude Code boots from `CLAUDE.md`, Codex from `AGENTS.md`, `agy` from this
section — all three converge on the same `docs/agent/*` deep docs, so
subsystem knowledge is written once there and never duplicated per agent.
When any agent learns a new gotcha, it appends to `tasks/lessons.md` (the
shared, agent-agnostic memory), not to its own boot file.
