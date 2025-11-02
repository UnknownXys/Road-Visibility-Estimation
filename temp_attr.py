import re
from pathlib import Path
config_path = Path('road_visibility/config.py')
text = config_path.read_text(encoding='utf-8')
attrs = re.findall(r"^\s*(\w+):", text, re.MULTILINE)
for attr in attrs:
    count = 0
    for path in Path('.').rglob('*.py'):
        if path == config_path:
            continue
        data = path.read_text(encoding='utf-8', errors='ignore')
        count += data.count(f'.{attr}')
    print(f"{attr}: {count}")
