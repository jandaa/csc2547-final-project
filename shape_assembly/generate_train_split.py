import json
import jsbeautifier
from pathlib import Path

opts = jsbeautifier.default_options()
opts.indent_size = 2

train_split = [str(i) for i in range(900)]

train_split_file = Path("test_split.json")
with train_split_file.open(mode="w") as f:
    json.dump(train_split, f, indent=None, separators=(',\n', ': '))