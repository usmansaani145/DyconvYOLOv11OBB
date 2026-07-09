import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_REPO / "utils") not in sys.path:
    sys.path.insert(0, str(_REPO / "utils"))

from utils.obb_coco_eval import *  # noqa: F401,F403
