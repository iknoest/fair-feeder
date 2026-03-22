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
assert old12 in src12, f"Cell 12: expected re-scan block not found\nGot: {src12[:500]}"
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
assert old13 in src13, f"Cell 13: expected re-scan block not found\nGot: {src13[:500]}"
src13 = src13.replace(old13, new13, 1)

# Fix merged_names injection: use merged_sources dict instead of fragile string check
old_merged = (
    "    try:\n"
    "        if 'downloaded_paths' in globals() and 'merged' in str(video_paths[0]): summary['merged_names'] = [__p.split('/')[-1] for __p in downloaded_paths]\n"
    "    except: pass\n"
)
new_merged = (
    "    if RUNNING_IN_CI and 'merged_sources' in globals() and vid_name in merged_sources:\n"
    "        summary['merged_names'] = merged_sources[vid_name]\n"
)
assert old_merged in src13, f"Cell 13: expected merged_names block not found\nGot: {src13[src13.find('merged_names')-200:src13.find('merged_names')+200] if 'merged_names' in src13 else 'merged_names not in cell'}"
src13 = src13.replace(old_merged, new_merged, 1)

nb["cells"][13]["source"] = [line + "\n" for line in src13.splitlines()]
nb["cells"][13]["source"][-1] = nb["cells"][13]["source"][-1].rstrip("\n")
print("✅ Cell 13 updated")

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")

print("✅ Cells 12 and 13 updated: no SOURCE_DIR rescan in CI")
