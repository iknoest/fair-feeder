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
assert INSERT_BEFORE in src, f"Patch 1: insertion point not found. Looking for: {INSERT_BEFORE!r}"
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
