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

import sys
sys.stdout.reconfigure(encoding="utf-8")
print("✅ Cell 1 updated: stitch gap check")
