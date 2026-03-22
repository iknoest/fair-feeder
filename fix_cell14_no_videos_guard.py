import json, sys
sys.stdout.reconfigure(encoding="utf-8")

NB = "morning_report.ipynb"

# Patch: add a _skip_csv guard and change the two if/else branches to check it
OLD_VCOUNT = (
    "_vcount = len(video_summaries) if 'video_summaries' in globals() else 0\n"
    "# Use last summary if multiple events; or empty dict if no results yet\n"
    "_s = video_results[-1]['summary'] if ('video_results' in globals() and video_results) else {}\n"
)
NEW_VCOUNT = (
    "_vcount = len(video_summaries) if 'video_summaries' in globals() else 0\n"
    "_skip_csv = not ('video_results' in globals() and video_results)\n"
    "if _skip_csv:\n"
    "    print('ℹ️ No video_results — skipping CSV log (no videos processed today)')\n"
    "# Use last summary if multiple events; or empty dict if no results yet\n"
    "_s = video_results[-1]['summary'] if ('video_results' in globals() and video_results) else {}\n"
)

OLD_CI_GUARD = "if RUNNING_IN_CI:\n"
NEW_CI_GUARD = "if RUNNING_IN_CI and not _skip_csv:\n"

OLD_ELSE = "else:\n    _write_header = not _LOG_FILE.exists()"
NEW_ELSE = "elif not _skip_csv:\n    _write_header = not _LOG_FILE.exists()"

with open(NB, encoding="utf-8") as f:
    nb = json.load(f)

src = "".join(nb["cells"][14]["source"])

assert OLD_VCOUNT in src, "Patch 1: vcount block not found"
src = src.replace(OLD_VCOUNT, NEW_VCOUNT, 1)

# Replace only the first occurrence of "if RUNNING_IN_CI:\n" in this cell
assert OLD_CI_GUARD in src, "Patch 2: CI guard not found"
src = src.replace(OLD_CI_GUARD, NEW_CI_GUARD, 1)

assert OLD_ELSE in src, "Patch 3: else block not found"
src = src.replace(OLD_ELSE, NEW_ELSE, 1)

nb["cells"][14]["source"] = [line + "\n" for line in src.splitlines()]
nb["cells"][14]["source"][-1] = nb["cells"][14]["source"][-1].rstrip("\n")

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")

print("✅ Cell 14 patched: skips CSV log when no videos processed")
