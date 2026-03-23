import json, sys
sys.stdout.reconfigure(encoding="utf-8")

NB = "morning_report.ipynb"

with open(NB, encoding="utf-8") as f:
    nb = json.load(f)

# Find the retry/discord-notification cell (Cell 15 by position, but search by content)
for i, c in enumerate(nb["cells"]):
    src = "".join(c["source"])
    if "Re-send" in src and "video_summaries" in src:
        # Prepend the CI guard
        OLD = src[:src.index("\n") + 1]  # first line
        if "RUNNING_IN_CI" in src:
            print(f"Cell {i}: already guarded, skipping")
            break
        new_src = "if RUNNING_IN_CI:\n    print('ℹ️ CI mode: skipping retry cell (Phase 3 already sent)')\nelse:\n" + "\n".join("    " + line for line in src.splitlines()) + "\n"
        nb["cells"][i]["source"] = [line + "\n" for line in new_src.splitlines()]
        nb["cells"][i]["source"][-1] = nb["cells"][i]["source"][-1].rstrip("\n")
        print(f"✅ Cell {i} guarded: retry cell skips in CI")
        break
else:
    print("⚠️ Retry cell not found")

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write("\n")
