# Pipeline Regressions Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 5 regressions in `morning_report.ipynb` so the daily CI report shows correct kibble counts, analyzes only one event per stitch group, sends the right annotated video, uploads it to Drive, and logs accurate CSV data.

**Architecture:** All changes are to `morning_report.ipynb`. Notebooks cannot be edited with standard file tools — every change is done by a Python update script that reads the notebook JSON, replaces the target cell's `source`, and writes it back. Tests are pure Python scripts that validate logic without running the notebook.

**Tech Stack:** Python 3.11, papermill, cv2, numpy, Google Drive API v3, ffmpeg concat

---

## Root Causes (confirmed by code inspection)

| Bug | Root cause |
|-----|-----------|
| Kibble returns 0 | `_find_clear_kibble_count()` searches frames with no cats; model only detects kibble when cats are present → always returns 0 |
| 3 videos processed instead of 1 | Phase 1 (Cell 12) and Phase 2 (Cell 13) re-scan `SOURCE_DIR`, overriding the `video_paths` set by the stitch cell (Cell 1) |
| Wrong/missing annotated video | Cell 9 constructs annotated path from `SOURCE_DIR`, but the file is in `OUTPUT_DIR` |
| CSV zeros | Cell 9 runs **before** Phase 1–3; `summary` and `video_summaries` don't exist yet |
| Gap-blind stitching | Cell 1 merges all clips regardless of time gap — events 3m40s apart get merged |

---

## Files

| File | Action | Purpose |
|------|--------|---------|
| `morning_report.ipynb` | Modify cells 1, 9, 10, 12, 13 | All 5 fixes live here |
| `tests/legacy_notebook/test_notebook_fixes.py` | Create | Unit tests for gap-grouping logic and kibble attribution |
| `fix_cell_01_stitch_gap.py` | Create | Update script for Task 1 |
| `fix_cell_12_13_rescan.py` | Create | Update script for Task 2 |
| `fix_cell_10_kibble.py` | Create | Update script for Task 3 |
| `fix_cell_09_upload_order.py` | Create | Update script for Tasks 4+5 (cell 9 rewrite + reorder) |

All update scripts are run once and then deleted (or kept in `artifacts/` for reference).

---

## Task 1: Fix stitch cell — only merge clips within 10-second gap

**Files:**
- Modify: `morning_report.ipynb` cell 1
- Create: `fix_cell_01_stitch_gap.py`
- Test: `tests/legacy_notebook/test_notebook_fixes.py` (new, `TestGroupByGap` class)

The stitch cell must parse clip start/end times from filenames, group consecutive clips by gap ≤ 10s, and produce one `video_paths` entry per event group (merged if >1 clip in group, raw if single). It must also populate `merged_sources = {merged_filename: [clip1, clip2, ...]}` for use in Phase 2.

Filename format: `motion_YYYYMMDD_HHMMSS_Xm_Ys.mp4` (minutes optional).

- [ ] **Step 1: Write failing tests for gap-grouping logic**

Create `tests/legacy_notebook/test_notebook_fixes.py`:

```python
from pathlib import Path
from datetime import datetime, timedelta
import re

# ── Paste the two functions under test here (will match notebook) ──

def _parse_clip_times(filename):
    m = re.match(r'motion_(\d{8})_(\d{6})(?:_(\d+)m)?_(\d+)s', Path(filename).stem)
    if not m:
        return None, None
    start = datetime.strptime(f"{m.group(1)}{m.group(2)}", "%Y%m%d%H%M%S")
    minutes = int(m.group(3)) if m.group(3) else 0
    seconds = int(m.group(4))
    return start, start + timedelta(minutes=minutes, seconds=seconds)

def _group_by_gap(paths, gap_sec=10):
    if not paths:
        return []
    timed = [(p, *_parse_clip_times(p.name)) for p in paths]
    groups, cur = [], [timed[0]]
    for i in range(1, len(timed)):
        prev_end = cur[-1][2]
        curr_start = timed[i][1]
        if prev_end and curr_start:
            gap = (curr_start - prev_end).total_seconds()
        else:
            gap = gap_sec + 1
        if gap <= gap_sec:
            cur.append(timed[i])
        else:
            groups.append([t[0] for t in cur])
            cur = [timed[i]]
    groups.append([t[0] for t in cur])
    return groups

# ── Tests ──

def test_parse_clip_times_with_minutes():
    p = Path("motion_20260322_062004_1m_56s.mp4")
    start, end = _parse_clip_times(p.name)
    assert start == datetime(2026, 3, 22, 6, 20, 4)
    assert end == datetime(2026, 3, 22, 6, 22, 0)

def test_parse_clip_times_seconds_only():
    p = Path("motion_20260322_062540_25s.mp4")
    start, end = _parse_clip_times(p.name)
    assert start == datetime(2026, 3, 22, 6, 25, 40)
    assert end == datetime(2026, 3, 22, 6, 26, 5)

def test_group_far_apart_clips_are_separate_events():
    # Gap = 06:25:40 - 06:22:00 = 220s > 10s → 2 groups
    clips = [
        Path("motion_20260322_062004_1m_56s.mp4"),
        Path("motion_20260322_062540_25s.mp4"),
    ]
    groups = _group_by_gap(clips)
    assert len(groups) == 2
    assert groups[0][0].name == "motion_20260322_062004_1m_56s.mp4"
    assert groups[1][0].name == "motion_20260322_062540_25s.mp4"

def test_group_close_clips_are_same_event():
    # Clip 1 ends 06:20:30, Clip 2 starts 06:20:35 → gap = 5s ≤ 10s → 1 group
    clips = [
        Path("motion_20260322_062000_30s.mp4"),
        Path("motion_20260322_062035_20s.mp4"),
    ]
    groups = _group_by_gap(clips)
    assert len(groups) == 1
    assert len(groups[0]) == 2

def test_group_exactly_10s_gap_is_same_event():
    clips = [
        Path("motion_20260322_062000_30s.mp4"),
        Path("motion_20260322_062040_20s.mp4"),  # gap = 10s exactly
    ]
    groups = _group_by_gap(clips)
    assert len(groups) == 1

def test_group_11s_gap_is_separate():
    clips = [
        Path("motion_20260322_062000_30s.mp4"),
        Path("motion_20260322_062041_20s.mp4"),  # gap = 11s
    ]
    groups = _group_by_gap(clips)
    assert len(groups) == 2

def test_empty_input():
    assert _group_by_gap([]) == []

def test_single_clip():
    clips = [Path("motion_20260322_062000_30s.mp4")]
    groups = _group_by_gap(clips)
    assert len(groups) == 1
    assert len(groups[0]) == 1

if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    if failed:
        raise SystemExit(1)
```

- [ ] **Step 2: Run tests — expect FAIL (functions not yet implemented in notebook)**

```
python tests/legacy_notebook/test_notebook_fixes.py
```
Expected: tests run but the functions are defined inline in the test file, so they should PASS already (this validates the function design before putting it in the notebook).

Actually all tests should PASS here since the functions are defined in the test file itself. The tests prove the logic is correct before we embed it in the notebook.

- [ ] **Step 3: Write the notebook update script**

Create `fix_cell_01_stitch_gap.py`:

```python
import json, re
from pathlib import Path

NB = "morning_report.ipynb"
NEW_SOURCE = r"""# ── Stitch feeding-window clips into one video (CI only) ──────────
import os as _os2, re as _re2, subprocess as _sp
from pathlib import Path as _Path2
from datetime import datetime as _dt2, timedelta as _td2

STITCH_GAP_SECONDS = 10

def _parse_clip_times(filename):
    m = _re2.match(r'motion_(\d{8})_(\d{6})(?:_(\d+)m)?_(\d+)s', _Path2(filename).stem)
    if not m:
        return None, None
    start = _dt2.strptime(f"{m.group(1)}{m.group(2)}", "%Y%m%d%H%M%S")
    minutes = int(m.group(3)) if m.group(3) else 0
    seconds = int(m.group(4))
    return start, start + _td2(minutes=minutes, seconds=seconds)

def _group_by_gap(paths, gap_sec=STITCH_GAP_SECONDS):
    if not paths:
        return []
    timed = [(p, *_parse_clip_times(p.name)) for p in paths]
    groups, cur = [], [timed[0]]
    for i in range(1, len(timed)):
        prev_end = cur[-1][2]
        curr_start = timed[i][1]
        if prev_end and curr_start:
            gap = (curr_start - prev_end).total_seconds()
        else:
            gap = gap_sec + 1
        if gap <= gap_sec:
            cur.append(timed[i])
        else:
            groups.append([t[0] for t in cur])
            cur = [timed[i]]
    groups.append([t[0] for t in cur])
    return groups

if RUNNING_IN_CI:
    merged_sources = {}  # merged_filename -> [source clip names]
    video_paths = []

    groups = _group_by_gap([_Path2(p) for p in downloaded_paths])
    print(f"ℹ️ {len(downloaded_paths)} clip(s) grouped into {len(groups)} feeding event(s) (gap threshold: {STITCH_GAP_SECONDS}s)")

    for _gi, _group in enumerate(groups):
        if len(_group) > 1:
            _list_file = _os2.path.join(SOURCE_DIR, f'_concat_{_gi}.txt')
            with open(_list_file, 'w') as _lf:
                _lf.write('\n'.join(f"file '{p}'" for p in _group))
            _mname = f"feeding_merged_{_gi}.mp4" if len(groups) > 1 else "feeding_merged.mp4"
            _merged = _os2.path.join(SOURCE_DIR, _mname)
            _r = _sp.run(
                ['ffmpeg', '-f', 'concat', '-safe', '0', '-i', _list_file,
                 '-c', 'copy', '-y', _merged],
                capture_output=True, text=True
            )
            if _r.returncode == 0:
                video_paths.append(_Path2(_merged))
                merged_sources[_mname] = [p.name for p in _group]
                print(f"  ✅ Event {_gi+1}: merged {len(_group)} clips → {_mname}")
            else:
                print(f"  ⚠️ Merge failed for event {_gi+1}:\n{_r.stderr}\n  Falling back to individual clips")
                video_paths.extend(_group)
        else:
            video_paths.append(_group[0])
            print(f"  ℹ️ Event {_gi+1}: single clip {_group[0].name}")
"""

with open(NB, encoding="utf-8") as f:
    nb = json.load(f)

# Cell 1 is the stitch cell
nb["cells"][1]["source"] = [line + "\n" for line in NEW_SOURCE.splitlines()]
nb["cells"][1]["source"][-1] = nb["cells"][1]["source"][-1].rstrip("\n")

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")

print("✅ Cell 1 updated: stitch gap check")
```

- [ ] **Step 4: Run the update script**

```
python fix_cell_01_stitch_gap.py
```
Expected: `✅ Cell 1 updated: stitch gap check`

- [ ] **Step 5: Verify cell 1 content in notebook**

```
python -c "
import json,sys; sys.stdout.reconfigure(encoding='utf-8')
nb=json.load(open('morning_report.ipynb',encoding='utf-8'))
print(''.join(nb['cells'][1]['source'])[:200])
"
```
Expected: first lines start with `# ── Stitch feeding-window clips` and contain `STITCH_GAP_SECONDS`.

- [ ] **Step 6: Commit**

```bash
git add morning_report.ipynb tests/legacy_notebook/test_notebook_fixes.py fix_cell_01_stitch_gap.py
git commit -m "fix(notebook): stitch only clips within 10s gap, separate events independently"
```

---

## Task 2: Phase 1 and Phase 2 must not re-scan SOURCE_DIR in CI

**Files:**
- Modify: `morning_report.ipynb` cells 12 (Phase 1) and 13 (Phase 2)
- Create: `fix_cell_12_13_rescan.py`

Cells 12 and 13 both begin by re-scanning SOURCE_DIR, which overwrites `video_paths` set by Cell 1. In CI, guard those re-scans so they only run in Colab mode. Also fix the `merged_names` injection in Cell 13 to use the `merged_sources` dict.

- [ ] **Step 1: Write the update script**

Create `fix_cell_12_13_rescan.py`:

```python
import json, sys
sys.stdout.reconfigure(encoding="utf-8")

NB = "morning_report.ipynb"

RESCAN_GUARD = (
    "if not RUNNING_IN_CI:\n"
    "    all_files = sorted(Path(SOURCE_DIR).iterdir())\n"
    "    video_paths = [f for f in all_files if classify_file(f) == \"video\"]\n"
)

with open(NB, encoding="utf-8") as f:
    nb = json.load(f)

# ── Cell 12 (Phase 1) ──────────────────────────────────────────
src12 = "".join(nb["cells"][12]["source"])
old12 = (
    "all_files = sorted(Path(SOURCE_DIR).iterdir())\n"
    "video_paths = [f for f in all_files if classify_file(f) == \"video\"]\n"
    "print(f\"✅ Found {len(video_paths)} video(s) in SOURCE_DIR\")"
)
new12 = (
    RESCAN_GUARD +
    "print(f\"✅ Found {len(video_paths)} video(s) to process\")"
)
assert old12 in src12, "Cell 12: expected re-scan block not found"
src12 = src12.replace(old12, new12, 1)
nb["cells"][12]["source"] = [line + "\n" for line in src12.splitlines()]
nb["cells"][12]["source"][-1] = nb["cells"][12]["source"][-1].rstrip("\n")
print("✅ Cell 12 updated")

# ── Cell 13 (Phase 2) ──────────────────────────────────────────
src13 = "".join(nb["cells"][13]["source"])
old13 = (
    "all_files = sorted(Path(SOURCE_DIR).iterdir())\n"
    "video_paths = [f for f in all_files if classify_file(f) == \"video\"]\n"
    "print(f\"✅ Found {len(video_paths)} video(s)\")"
)
new13 = (
    RESCAN_GUARD +
    "print(f\"✅ Found {len(video_paths)} video(s)\")"
)
assert old13 in src13, "Cell 13: expected re-scan block not found"
src13 = src13.replace(old13, new13, 1)

# Fix merged_names injection: use merged_sources dict instead of fragile string check
old_merged = (
    "    try:\n"
    "        if 'downloaded_paths' in globals() and 'merged' in str(video_paths[0]):"
    " summary['merged_names'] = [__p.split('/')[-1] for __p in downloaded_paths]\n"
    "    except: pass\n"
)
new_merged = (
    "    if RUNNING_IN_CI and 'merged_sources' in globals() and vid_name in merged_sources:\n"
    "        summary['merged_names'] = merged_sources[vid_name]\n"
)
assert old_merged in src13, "Cell 13: expected merged_names block not found"
src13 = src13.replace(old_merged, new_merged, 1)

nb["cells"][13]["source"] = [line + "\n" for line in src13.splitlines()]
nb["cells"][13]["source"][-1] = nb["cells"][13]["source"][-1].rstrip("\n")
print("✅ Cell 13 updated")

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")

print("✅ Cells 12 and 13 updated: no SOURCE_DIR rescan in CI")
```

- [ ] **Step 2: Run the update script**

```
python fix_cell_12_13_rescan.py
```
Expected: three `✅` lines, no AssertionError.

- [ ] **Step 3: Verify**

```
python -c "
import json,sys; sys.stdout.reconfigure(encoding='utf-8')
nb=json.load(open('morning_report.ipynb',encoding='utf-8'))
src12 = ''.join(nb['cells'][12]['source'])
src13 = ''.join(nb['cells'][13]['source'])
assert 'if not RUNNING_IN_CI:' in src12, 'Cell 12 guard missing'
assert 'if not RUNNING_IN_CI:' in src13, 'Cell 13 guard missing'
assert 'merged_sources' in src13, 'Cell 13 merged_sources missing'
assert 'downloaded_paths' not in src13, 'Cell 13 old merged_names still there'
print('✅ Verification passed')
"
```

- [ ] **Step 4: Commit**

```bash
git add morning_report.ipynb fix_cell_12_13_rescan.py
git commit -m "fix(notebook): preserve CI video_paths in Phase 1/2, fix merged_names lookup"
```

---

## Task 3: Fix FeedingTracker kibble attribution — use phase-entry frames

**Files:**
- Modify: `morning_report.ipynb` cell 10 (FeedingTracker class)
- Create: `fix_cell_10_kibble.py`
- Test: `tests/legacy_notebook/test_notebook_fixes.py` (add `TestKibbleAttribution` class)

**The problem:** `_find_clear_kibble_count` requires frames where no cats are at the bowl. In Pi IR footage, kibble is only detected when cats ARE at the bowl. Clear frames have `kibble_count=0` always. Attribution fails.

**The fix:** When `_find_clear_kibble_count` returns 0, fall back to sampling kibble from the first/last 1–2 seconds of the feeding phase itself (cat just arrived = minimal eating has occurred yet; last frames = remaining kibble before cat left).

Add two new methods to `FeedingTracker`:
- `_find_kibble_at_phase_entry(phase_start, phase_end, entry_sec=1.0)` — median of kibble counts from first `entry_sec` seconds of phase
- `_find_kibble_at_phase_exit(phase_start, phase_end, exit_sec=2.0)` — median of kibble counts from last `exit_sec` seconds of phase

Update `summarize()`:
1. In phase loop: if `kb_before == 0`, use `_find_kibble_at_phase_entry`
2. In phase loop: if `kb_after is None`, use `_find_kibble_at_phase_exit`
3. For `start_kibble`: if `first_clear == 0` and phases exist, use entry of first phase

- [ ] **Step 1: Add kibble attribution tests to `tests/legacy_notebook/test_notebook_fixes.py`**

Append to `tests/legacy_notebook/test_notebook_fixes.py`:

```python
# ── Minimal FeedingTracker stub for testing attribution logic ──

import numpy as np

class _MockTracker:
    """Minimal tracker stub — only implements what we're testing."""
    def __init__(self, fps, kibble_counts, dan_at_bowl, sanbo_at_bowl):
        self.fps = fps
        self.kibble_counts = kibble_counts
        self.dan_at_bowl = dan_at_bowl
        self.sanbo_at_bowl = sanbo_at_bowl

    def _find_kibble_at_phase_entry(self, phase_start, phase_end, entry_sec=1.0):
        entry_frames = max(1, int(self.fps * entry_sec))
        end_sample = min(phase_start + entry_frames, phase_end + 1)
        counts = self.kibble_counts[phase_start:end_sample]
        return int(np.median(counts)) if counts else None

    def _find_kibble_at_phase_exit(self, phase_start, phase_end, exit_sec=2.0):
        exit_frames = max(1, int(self.fps * exit_sec))
        start_sample = max(phase_start, phase_end - exit_frames + 1)
        counts = self.kibble_counts[start_sample:phase_end + 1]
        return int(np.median(counts)) if counts else None


def test_phase_entry_returns_first_frames_median():
    # kibble=0 before cat, then 8,7,9 when cat arrives, then drops to 2
    t = _MockTracker(fps=2.0,
        kibble_counts=[0,0,0,0,0, 8,7,9,6,5,4,3,2, 0,0],
        dan_at_bowl=[False]*5 + [True]*8 + [False]*2,
        sanbo_at_bowl=[False]*15)
    # Phase: frames 5-12. Entry: first 2 frames (fps=2, entry_sec=1.0 → 2 frames): [8,7]
    # int(np.median([8,7])) = int(7.5) = 7
    result = t._find_kibble_at_phase_entry(5, 12)
    assert result == 7, f"Expected 7, got {result}"


def test_phase_exit_returns_last_frames_median():
    t = _MockTracker(fps=2.0,
        kibble_counts=[0,0,0,0,0, 8,7,9,6,5,4,3,2, 0,0],
        dan_at_bowl=[False]*5 + [True]*8 + [False]*2,
        sanbo_at_bowl=[False]*15)
    # Phase: frames 5-12. Exit: last 4 frames (fps=2, exit_sec=2.0 → 4 frames): [5,4,3,2]
    result = t._find_kibble_at_phase_exit(5, 12)
    assert result == 3, f"Expected 3 (median of [3,2,5,4] last 4 frames), got {result}"


def test_phase_entry_single_frame_fallback():
    # fps=0.5 means int(0.5 * 1.0) = 0, max(1, 0) = 1 frame
    t = _MockTracker(fps=0.5,
        kibble_counts=[10, 8, 5, 2],
        dan_at_bowl=[True]*4,
        sanbo_at_bowl=[False]*4)
    result = t._find_kibble_at_phase_entry(0, 3)
    assert result == 10, f"Expected 10, got {result}"


def test_phase_entry_empty_phase():
    t = _MockTracker(fps=2.0,
        kibble_counts=[5, 5, 5],
        dan_at_bowl=[True]*3,
        sanbo_at_bowl=[False]*3)
    # phase_start > phase_end (degenerate)
    result = t._find_kibble_at_phase_entry(5, 3)  # empty slice
    assert result is None
```

- [ ] **Step 2: Run new tests**

```
python tests/legacy_notebook/test_notebook_fixes.py
```
Expected: all PASS (the functions are defined in the test stub — confirms the logic before writing to notebook).

**Note on `test_phase_entry_returns_first_frames_median`:** `median([8,7])` is `7.5`, `int(7.5)` is `7` in Python. Fix the assertion to `assert result == 7`.

- [ ] **Step 3: Write the update script**

Create `fix_cell_10_kibble.py`:

```python
import json, sys
sys.stdout.reconfigure(encoding="utf-8")

NB = "morning_report.ipynb"

# ── New methods to insert into FeedingTracker ──────────────────
NEW_METHODS = '''
    def _find_kibble_at_phase_entry(self, phase_start, phase_end, entry_sec=1.0):
        """Kibble count from first N seconds of phase (cat just arrived, minimal eating)."""
        entry_frames = max(1, int(self.fps * entry_sec))
        end_sample = min(phase_start + entry_frames, phase_end + 1)
        counts = self.kibble_counts[phase_start:end_sample]
        return int(np.median(counts)) if counts else None

    def _find_kibble_at_phase_exit(self, phase_start, phase_end, exit_sec=2.0):
        """Kibble count from last N seconds of phase (just before cat left)."""
        exit_frames = max(1, int(self.fps * exit_sec))
        start_sample = max(phase_start, phase_end - exit_frames + 1)
        counts = self.kibble_counts[start_sample:phase_end + 1]
        return int(np.median(counts)) if counts else None

'''

# ── Patch 1: insert new methods before _find_clear_kibble_count ─
INSERT_BEFORE = "    def _find_clear_kibble_count(self, from_frame, direction=\"before\"):"

# ── Patch 2: update phase-attribution kb_before fallback ────────
OLD_KB_BEFORE = (
    "            kb_before = self._find_clear_kibble_count(start_f, direction=\"before\")\n"
    "            kb_after = self._find_clear_kibble_count(end_f, direction=\"after\")\n"
    "\n"
    "            # Edge case: video starts during feeding, no clear frames before\n"
    "            if kb_before is None and start_f < int(self.fps * 2):\n"
    "                kb_before = self.kibble_counts[0]\n"
    "            # Edge case: video ends during feeding, no clear frames after\n"
    "            if kb_after is None and end_f > n_frames - int(self.fps * 2):\n"
    "                kb_after = self.kibble_counts[-1]\n"
)
NEW_KB_BEFORE = (
    "            kb_before = self._find_clear_kibble_count(start_f, direction=\"before\")\n"
    "            # Fallback: clear frames have 0 kibble (model only detects kibble with cats)\n"
    "            if not kb_before:\n"
    "                kb_before = self._find_kibble_at_phase_entry(start_f, end_f)\n"
    "            kb_after = self._find_clear_kibble_count(end_f, direction=\"after\")\n"
    "            # Fallback: no clear frames after phase (e.g. video ends with cat at bowl)\n"
    "            if kb_after is None:\n"
    "                kb_after = self._find_kibble_at_phase_exit(start_f, end_f)\n"
    "\n"
    "            # Edge case: video starts during feeding, no clear frames before\n"
    "            if kb_before is None and start_f < int(self.fps * 2):\n"
    "                kb_before = self.kibble_counts[0]\n"
    "            # Edge case: video ends during feeding, no clear frames after\n"
    "            if kb_after is None and end_f > n_frames - int(self.fps * 2):\n"
    "                kb_after = self.kibble_counts[-1]\n"
)

# ── Patch 3: update start_kibble fallback ─────────────────────
OLD_FIRST_CLEAR = (
    "        first_clear = self._find_clear_kibble_count(0, direction=\"after\")\n"
    "        last_clear = self._find_clear_kibble_count(n_frames - 1, direction=\"before\")\n"
)
NEW_FIRST_CLEAR = (
    "        first_clear = self._find_clear_kibble_count(0, direction=\"after\")\n"
    "        # Fallback: clear frames have 0 kibble — reuse phases (already computed above)\n"
    "        if not first_clear and phases:\n"
    "            first_clear = self._find_kibble_at_phase_entry(phases[0]['start'], phases[0]['end'])\n"
    "        last_clear = self._find_clear_kibble_count(n_frames - 1, direction=\"before\")\n"
)

with open(NB, encoding="utf-8") as f:
    nb = json.load(f)

src = "".join(nb["cells"][10]["source"])

# Apply patch 1: insert new methods
assert INSERT_BEFORE in src, "Patch 1: insertion point not found"
src = src.replace(INSERT_BEFORE, NEW_METHODS + INSERT_BEFORE, 1)

# Apply patch 2: kb_before/after fallbacks
assert OLD_KB_BEFORE in src, f"Patch 2: kb_before block not found"
src = src.replace(OLD_KB_BEFORE, NEW_KB_BEFORE, 1)

# Apply patch 3: start_kibble fallback
assert OLD_FIRST_CLEAR in src, "Patch 3: first_clear block not found"
src = src.replace(OLD_FIRST_CLEAR, NEW_FIRST_CLEAR, 1)

nb["cells"][10]["source"] = [line + "\n" for line in src.splitlines()]
nb["cells"][10]["source"][-1] = nb["cells"][10]["source"][-1].rstrip("\n")

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")

print("✅ Cell 10 updated: FeedingTracker kibble attribution fallback")
```

- [ ] **Step 4: Run the update script**

```
python fix_cell_10_kibble.py
```
Expected: `✅ Cell 10 updated: FeedingTracker kibble attribution fallback` — no AssertionErrors.

- [ ] **Step 5: Verify methods appear in notebook**

```
python -c "
import json,sys; sys.stdout.reconfigure(encoding='utf-8')
nb=json.load(open('morning_report.ipynb',encoding='utf-8'))
src = ''.join(nb['cells'][10]['source'])
assert '_find_kibble_at_phase_entry' in src
assert '_find_kibble_at_phase_exit' in src
assert 'Fallback: clear frames have 0 kibble' in src
assert 'Fallback: clear frames have 0 kibble' in src
print('✅ All patches verified')
"
```

- [ ] **Step 6: Commit**

```bash
git add morning_report.ipynb tests/legacy_notebook/test_notebook_fixes.py fix_cell_10_kibble.py
git commit -m "fix(tracker): fall back to phase-entry/exit kibble when clear frames return 0"
```

---

## Task 4 + 5: Fix Drive video upload path AND move CSV cell to after Phase 3

**Files:**
- Modify: `morning_report.ipynb` cell 9 (rewrite), then reorder so it's after cell 14
- Create: `fix_cell_09_upload_order.py`

Two separate bugs fixed together because they're both in Cell 9:
1. Annotated video path uses SOURCE_DIR — should use OUTPUT_DIR + stem
2. Cell 9 runs before Phase 1–3 — must run after Phase 3 (cell 14)

The fix: rewrite Cell 9 to correctly derive the annotated video path, then move it to position 15 in the notebook (after the current cell 14 = Phase 3).

- [ ] **Step 1: Write the update script**

Create `fix_cell_09_upload_order.py`:

```python
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

# Move cell 9 to after cell 14 (Phase 3):
# Current order: 0..8, [9_csv], 10..14, 15, 16
# Target order:  0..8, 10..14, [9_csv], 15, 16
csv_cell = cells.pop(9)          # remove from position 9
# Now Phase 3 is at position 13 (was 14, shifted left by 1)
cells.insert(14, csv_cell)       # insert after new position 13 (= old cell 14)

nb["cells"] = cells

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")

print("✅ Cell 9 rewritten: correct OUTPUT_DIR path, reads summary from video_results")
print("✅ CSV+upload cell moved to after Phase 3")
```

- [ ] **Step 2: Run the update script**

```
python fix_cell_09_upload_order.py
```
Expected: two `✅` lines.

- [ ] **Step 3: Verify cell order and content**

```
python -c "
import json,sys; sys.stdout.reconfigure(encoding='utf-8')
nb=json.load(open('morning_report.ipynb',encoding='utf-8'))
for i,c in enumerate(nb['cells']):
    src=''.join(c['source'])
    print(f'{i:02d}: {src.split(chr(10))[0][:80]}')
"
```
Expected: The CSV cell (containing `Append to feeding_log.csv`) appears AFTER the Phase 3 cell (containing `Phase 3: Output & Telegram`).

Also verify the annotated video path fix:
```
python -c "
import json,sys; sys.stdout.reconfigure(encoding='utf-8')
nb=json.load(open('morning_report.ipynb',encoding='utf-8'))
# Find the csv cell by content
for i,c in enumerate(nb['cells']):
    src=''.join(c['source'])
    if 'feeding_log.csv' in src and 'Phase' not in src[:50]:
        assert 'OUTPUT_DIR' in src, 'OUTPUT_DIR missing from video upload path'
        assert 'SOURCE_DIR' not in src or 'RUNNING_IN_CI' in src, 'SOURCE_DIR still used for video path'
        print(f'✅ CSV+upload cell at position {i}, uses OUTPUT_DIR correctly')
        break
"
```

- [ ] **Step 4: Commit**

```bash
git add morning_report.ipynb fix_cell_09_upload_order.py
git commit -m "fix(notebook): correct annotated video upload path and move CSV cell after Phase 3"
```

---

## Task 6: Verify end-to-end with CI artifact

After pushing, trigger a manual workflow run and examine the output notebook artifact.

- [ ] **Step 1: Push and trigger**

```bash
git push
gh workflow run "Fair Feeder Morning Report" --ref main
```

- [ ] **Step 2: Wait and download artifact**

```
gh run list --workflow morning-report.yml --limit 3
gh run view <run_id>
gh run download <run_id>
```

- [ ] **Step 3: Inspect the output notebook**

Open `smoketest_output.ipynb` (or whatever the artifact is named) and check:
- Cell 1 prints how many events were grouped (should show 2 events for the test clips)
- Phase 2 prints non-zero `dan_kibble_eaten` or `sanbo_kibble_eaten` in the summary
- Phase 3 shows it sent the merged video (`feeding_merged_N_annotated.mp4`) not the individual 25s clip
- CSV cell prints "Updated feeding_log.csv on Drive" (not "Logged: {all zeros}")

- [ ] **Step 4: Check Drive output folder**

Open `H:\My Drive\Fun Project\Cat monitor\Test_postmodel_output` and verify:
- An `_annotated.mp4` file exists with today's date
- `feeding_log.csv` has a new row with non-zero kibble values

- [ ] **Step 5: Check Telegram**

The morning report should show:
- `Start: ~N kibble` where N > 0
- Either `Dan ate X%` / `Sanbo ate Y%` bars, OR a valid "no activity" reason
- Only the merged annotated video (not the individual 25s clip separately)
- If 2 separate events: 2 separate report messages, one per event
