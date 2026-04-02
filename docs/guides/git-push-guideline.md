# What to Push to GitHub (and What Not To)

## Push (commit + push to main)

| Category | Examples | Why |
|----------|---------|-----|
| **Core production code** | `motion_recorder.py`, `config.py`, `flagging.py`, `roboflow_upload.py` | Runs on Pi, CI, or Colab — must be version-controlled |
| **Notebooks** | `morning_report.ipynb`, `batch_review.ipynb`, `smoketest.ipynb`, `fair_feeder_v14.ipynb` | Shared across CI and Colab; changes must propagate |
| **Unit tests for production modules** | `test_flagging.py`, `test_roboflow_upload.py` | Guard production code; run before merging changes |
| **CI workflows** | `.github/workflows/morning-report.yml` | Controls automated pipeline |
| **Training/dataset tools** | `train.py`, `download_dataset.py`, `verify_labels.py`, `polygon_to_bbox.py` | Reproducible model training |
| **Configuration** | `data.yaml`, `requirements.txt`, `sync_cleanup.sh` | Deployment-critical config |
| **Documentation** | `CLAUDE.md`, `tasks/todo.md`, `tasks/lessons.md`, `docs/**/*.md` | Project knowledge for future sessions |
| **Deployment guides** | `README_GIT_PULL.md`, `README_RPI_SERVICE.md` | Onboarding and Pi setup |

## Do NOT push

| Category | Examples | Why |
|----------|---------|-----|
| **One-shot notebook patch scripts** | `fix_cell_01_harden.py`, `fix_notebook_tasks345.py`, `patch_batch_review.py` | Applied once to modify `.ipynb` JSON, never needed again. Delete after use. |
| **Generator scripts for notebooks** | `create_batch_review.py` | Produces a notebook once; the notebook itself is what matters |
| **One-time cleanup utilities** | `clean_feeding_log.py`, `list_cells.py`, `cells.txt` | Run once to fix data; no ongoing value |
| **Session artifacts** | `artifacts/`, `2026-*-continuation*.txt`, `__pycache__/` | Debug outputs, conversation logs — local only |
| **Secrets or credentials** | `.env`, `credentials.json`, `service-account*.json` | Security risk; use Infisical or GitHub Secrets |
| **Large binary files** | Model weights (`.pt`), video files (`.mp4`) | Store on Google Drive; reference via `GDRIVE_MODEL_FILE_ID` |
| **IDE/editor config** | `.vscode/`, `.idea/` | Personal preference, not project config |

## Decision checklist

Before `git add`, ask:

1. **Will another session/person need this file?** No = don't push.
2. **Is it a one-time operation?** Yes = delete after running, don't commit.
3. **Does it contain secrets or large binaries?** Yes = never push.
4. **Is it a test for production code that still exists?** Yes = push it.
5. **Is it a test for a patch script that was deleted?** Yes = delete the test too.

## .gitignore recommendations

These patterns should be in `.gitignore`:

```
fix_*.py
patch_*.py
clean_*.py
list_*.py
cells.txt
artifacts/
__pycache__/
*.pt
*.mp4
.env
```
