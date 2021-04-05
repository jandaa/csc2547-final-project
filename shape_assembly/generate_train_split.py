import json
from pathlib import Path

train_split = ["val_" + str(i) for i in range(900)]

train_split_file = Path("test_split.json")
with train_split_file.open(mode="w") as f:
    json.dump(train_split, f, indent=None, separators=(',\n', ': '))