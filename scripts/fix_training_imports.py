#!/usr/bin/env python3
"""Fix training script paths and imports for standalone repo layout."""

from pathlib import Path
import re

REPO = Path(__file__).resolve().parents[1]
TRAINING = REPO / "training"

HEADER = '''
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "utils"))
from utils.constants import DATASET_YAML, GT_JSON, PRETRAINED_YOLO11, RUNS_ROOT, REPO_ROOT
'''.strip()

IDEA_MAP = {
    "train_idea1.py": "idea1_lska",
    "train_idea2.py": "idea2_ema",
    "train_idea3.py": "idea3_msda",
    "train_idea4.py": "idea4_dyconv",
    "train_idea5.py": "idea5_csd",
}

for path in TRAINING.glob("train_idea*.py"):
    text = path.read_text(encoding="utf-8")
    idea = IDEA_MAP.get(path.name, "idea4_dyconv")
    idea_root = f'IDEA_ROOT = REPO_ROOT / "ideas" / "{idea}"'
    text = re.sub(r"IDEA_ROOT = Path\(__file__\).*?\n", f"{idea_root}\n", text)
    text = re.sub(r"REPO = IDEA_ROOT\.parents\[.*?\]\n", "", text)
    if "REPO_ROOT = Path(__file__)" not in text:
        text = text.replace(
            "from pathlib import Path\n",
            "from pathlib import Path\n\nREPO_ROOT = Path(__file__).resolve().parents[1]\n",
            1,
        )
    if "from utils.constants import" not in text:
        insert_after = "import sys\n"
        text = text.replace(
            insert_after,
            insert_after
            + "REPO_ROOT = Path(__file__).resolve().parents[1]\n"
            + "sys.path.insert(0, str(REPO_ROOT))\n"
            + "sys.path.insert(0, str(REPO_ROOT / \"utils\"))\n"
            + "from utils.constants import DATASET_YAML, GT_JSON, PRETRAINED_YOLO11, RUNS_ROOT\n",
            1,
        )
    text = re.sub(
        r'DATA_YAML = .*?\n',
        "DATA_YAML = DATASET_YAML\n",
        text,
        count=1,
    )
    text = re.sub(
        r'GT_JSON = .*?\n',
        "GT_JSON = GT_JSON\n",
        text,
        count=1,
    )
    text = re.sub(
        r'WEIGHTS = .*?\n',
        "WEIGHTS = PRETRAINED_YOLO11\n",
        text,
        count=1,
    )
    text = re.sub(
        r'RUNS_ROOT = .*?\n',
        "RUNS_ROOT = RUNS_ROOT\n",
        text,
        count=1,
    )
    text = text.replace('sys.path.insert(0, str(REPO / "ideas"))', "")
    text = text.replace("sys.path.append(str(REPO))", "sys.path.insert(0, str(IDEA_ROOT))")
    text = text.replace(
        "from idea_train_callbacks import",
        "from utils.idea_train_callbacks import",
    )
    path.write_text(text, encoding="utf-8")
    print(f"fixed {path.name}")

print("done")
