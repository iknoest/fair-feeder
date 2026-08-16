# Coding Conventions

Use these rules for code and notebook edits.

## Python Style

- Python code should not add type annotations unless the edited file already uses
  them.
- Use existing helpers such as `draw_boxes`, `bbox_iou`, and `parse_results`.
- Keep notebook cells self-contained where possible.
- Class colors flow through `CLASS_COLORS_RGB` and `get_color_bgr()`.
- Detection thresholds belong in the Config cell; do not scatter magic numbers.
- Add comments only when they explain non-obvious logic.

## YOLO and Detection

- Preserve 16:9 aspect ratio with `rect=True` in all YOLO calls.
- Pass `imgsz` as a single int, never a tuple.
- Use `model.names` for class index mapping; YOLO sorts alphabetically.
- Do not assume class index 0 maps to the first class in `data.yaml`.
- Keep annotated video boxes-only with `show_label=False`.
- Use actual class names: Dan, Sanbo, Dan_hand, Bowl, Kibble.

## Notebook Editing

- Edit `.ipynb` files programmatically through JSON.
- Always strip Windows carriage returns with `.replace('\r', '')` before writing
  cell source.
- Guard Colab-only imports and `drive.mount()` with `RUNNING_IN_CI`.
- Use `tqdm.auto`, not `tqdm.notebook`.
- Test regex changes against partial OCR reads such as `09:51:5` and
  `2026-01-25`.
- Tapo OCR replacement order matters: replace `\|:` with `:1` before replacing
  `\|` with `1`.
- For cameras without on-screen timestamps (OSD), calculate frame time
  mathematically: `start_time_from_filename + (frame_idx / fps)`.

## Secrets

- Never hardcode Chat IDs, folder IDs, API keys, or credentials.
- Load secrets through Infisical or environment variables.
- Before committing, scan diffs for secrets and environment-specific IDs.

## Verification

- Match verification to blast radius.
- For CI-facing or notebook runtime changes, run focused tests and notebook JSON
  checks at minimum.
- For Pi runtime changes, compile on Pi and verify the restarted service.
- For CI-facing or Pi-runtime fixes, prepare commit and push to `main` when safe.
- Stop and ask if a push would include destructive history changes, credentials,
  or unrelated user work.

