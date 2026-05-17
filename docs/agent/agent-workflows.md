# Agent Workflow Candidates

These are good candidates for `.claude/skills/` or slash commands because they
are repetitive and have clear trigger language. Keep the root files short and load
these workflows only when needed.

## Skill Candidates

### `fair-feeder-ci-debug`

Trigger on: GitHub Actions, CI, Drive, `feeding_log.csv`, scheduler delay,
papermill, morning report failure.

Reference: `docs/agent/ci-drive-runbook.md`.

Workflow: inspect run logs, enumerate likely failure modes, audit workflow env
vars, check Drive update/create behavior, verify timezone handling, run focused
notebook/test checks, then commit and push if safe.

### `fair-feeder-notebook-edit`

Trigger on: `morning_report.ipynb`, FeedingTracker, snapshots, Telegram report,
Kibble logic, Sanbo/Dan attribution.

Reference: `docs/agent/morning-report-pipeline.md` and
`docs/agent/coding-conventions.md`.

Workflow: edit notebook JSON programmatically, strip `\r`, preserve snapshot
rules, run focused regression tests and notebook JSON validation.

### `fair-feeder-pi-runtime`

Trigger on: `motion_recorder.py`, Pi, systemd, RTSP, `/status`, `/weight`,
recording, rclone.

Reference: `docs/agent/runtime-surfaces.md`.

Workflow: patch Pi-runtime file, SCP to Pi, compile on Pi, restart
`cat-monitor.service`, verify active status and logs.

### `fair-feeder-model-flywheel`

Trigger on: Roboflow, model, V14, V15, mAP, flagged frames, retraining, dataset.

Reference: `docs/agent/model-data-flywheel.md`.

Workflow: compare fixed holdout metrics, report mAP50/per-class AP50, avoid
augmentation choices that harm Kibble, and preserve Roboflow dedup rules.

### `fair-feeder-doc-update`

Trigger on: document a lesson, decision, fix, update AGENTS/CLAUDE, update todo.

Reference: `docs/agent/documentation-policy.md`.

Workflow: update the correct root index and detailed agent doc, then check
`tasks/lessons.md`, `tasks/todo.md`, and `README.md` for matching changes.

## Slash Command Candidates

- `/ff-ci-preflight` - run the CI/Drive checklist before a workflow push.
- `/ff-pi-deploy` - deploy Pi runtime changes and verify systemd.
- `/ff-notebook-test` - validate notebook JSON and run focused report tests.
- `/ff-review-feeding-log` - inspect schedule/start times and daily rows.
- `/ff-doc-lesson` - add a lesson across docs, lessons, todo, and README as needed.

