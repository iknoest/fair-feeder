import json
import re

with open('morning_report.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_cells = []
for cell in nb['cells']:
    src = "".join(cell.get('source', []))
    
    # Strip unnecessary testing/validation cells
    if "roboflow" in src.lower() and "download dataset" in src.lower(): continue
    if "Fix Labels: Polygon" in src: continue
    if "metrics = model.val" in src: continue
    if "## Model Validation" in src: continue
    if "Benchmark inference speed" in src: continue
    
    # Update timezone and feeding window logic
    if "FEEDING_WINDOW_START" in src and "_in_feeding_window" in src:
        new_src = src.replace(
            "def _in_feeding_window(filename, start, end):",
            "import pytz\nfrom datetime import datetime\n\ndef _in_feeding_window(filename, start, end):"
        )
        # Update regex matching and min bounds to include today's date
        old_logic = """    m = _re.match(r'motion_\\d{8}_(\\d{2})(\\d{2})\\d{2}', filename)
    if not m:
        return False
    file_min  = int(m.group(1)) * 60 + int(m.group(2))
    start_min = start[0] * 60 + start[1]
    end_min   = end[0]   * 60 + end[1]
    return start_min <= file_min <= end_min"""

        new_logic = """    m = _re.match(r'motion_(\\d{8})_(\\d{2})(\\d{2})\\d{2}', filename)
    if not m:
        return False
        
    cet_tz = pytz.timezone('Europe/Amsterdam')
    today_str = datetime.now(cet_tz).strftime('%Y%m%d')
    file_date = m.group(1)
    
    # Only process today's videos
    if file_date != today_str:
        return False

    file_min  = int(m.group(2)) * 60 + int(m.group(3))
    start_min = start[0] * 60 + start[1]
    end_min   = end[0]   * 60 + end[1]
    return start_min <= file_min <= end_min"""

        new_src = new_src.replace(old_logic, new_logic)
        
        # Also fix the debug print to show today's date
        new_src = new_src.replace('f"({FEEDING_WINDOW_START[0]', 'f"(Date: {datetime.now(pytz.timezone(\'Europe/Amsterdam\')).strftime(\'%Y%m%d\')} | {FEEDING_WINDOW_START[0]')
        
        # Split back into lines array for notebook format
        lines = [line + '\\n' for line in new_src.split('\\n')]
        if lines:
            lines[-1] = lines[-1][:-1] # remove last newline
            cell['source'] = lines
            
    new_cells.append(cell)

nb['cells'] = new_cells

with open('morning_report.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook processed successfully!")
