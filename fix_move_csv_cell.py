import json, sys
sys.stdout.reconfigure(encoding="utf-8")

NB = "morning_report.ipynb"

with open(NB, encoding="utf-8") as f:
    nb = json.load(f)

cells = nb["cells"]

# Find Phase 3 cell index (must contain "Phase 3" header AND output/telegram content)
# but NOT the CSV cell (which only mentions Phase 3 in a comment)
phase3_idx = None
for i, c in enumerate(cells):
    src = "".join(c["source"])
    # Phase 3 cell starts with the dashes comment and has "Phase 3: Output & Telegram"
    if "Phase 3: Output" in src and "Telegram" in src:
        phase3_idx = i
        break

if phase3_idx is None:
    raise RuntimeError("Could not find Phase 3 cell")

print(f"Phase 3 found at cell index {phase3_idx}")

# Find the CSV cell (the one to move)
csv_idx = None
for i, c in enumerate(cells):
    src = "".join(c["source"])
    if "Append to feeding_log.csv on Drive + upload annotated video" in src:
        csv_idx = i
        break

if csv_idx is None:
    raise RuntimeError("Could not find CSV+upload cell")

print(f"CSV+upload cell found at index {csv_idx}")

if csv_idx > phase3_idx:
    print("CSV cell is already after Phase 3 — no move needed")
else:
    # Remove CSV cell from current position
    csv_cell = cells.pop(csv_idx)
    # After removal, Phase 3 index shifts down by 1 since csv_idx < phase3_idx
    new_phase3_idx = phase3_idx - 1
    # Insert after Phase 3
    insert_at = new_phase3_idx + 1
    cells.insert(insert_at, csv_cell)
    print(f"Moved CSV cell from index {csv_idx} to index {insert_at} (after Phase 3 at {new_phase3_idx})")

nb["cells"] = cells

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")

print("Done.")
