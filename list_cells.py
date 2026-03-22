import json, sys
data = json.load(open(sys.argv[1], encoding='utf-8'))
for i, cell in enumerate(data['cells']):
    src = ''.join(cell.get('source', []))[:120].replace('\n', '  ')
    print(f"Cell {i:02d} [{cell['cell_type']}]: {src}")
