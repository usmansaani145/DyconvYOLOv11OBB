"""Safe imports from SmallTrack without triggering heavy backbone __init__.py."""

from __future__ import annotations

import os
import sys
import types
from typing import Type

SMALLTRACK_ROOT = "./SmallTrack"


def _ensure_smalltrack_path() -> None:
  if SMALLTRACK_ROOT not in sys.path:
    sys.path.insert(0, SMALLTRACK_ROOT)


def _stub_backbone_package() -> None:
  """Pre-register backbone package to skip resnet_atrous_DWT imports."""
  name = "siamban.models.backbone"
  if name in sys.modules:
    return
  pkg = types.ModuleType(name)
  pkg.__path__ = [os.path.join(SMALLTRACK_ROOT, "siamban", "models", "backbone")]
  sys.modules[name] = pkg


def _stub_optional_viz_imports() -> None:
  """wad_module.py imports viz libraries at module level (unused in forward)."""
  import types

  stubs = {
      "imageio": types.ModuleType("imageio"),
      "cv2": types.ModuleType("cv2"),
      "PIL": types.ModuleType("PIL"),
      "PIL.Image": types.ModuleType("PIL.Image"),
      "matplotlib": types.ModuleType("matplotlib"),
      "matplotlib.pyplot": types.ModuleType("matplotlib.pyplot"),
  }
  for name, mod in stubs.items():
    if name not in sys.modules:
      sys.modules[name] = mod


def import_wad_module_class():
  """Return the original SmallTrack wad_module class."""
  _ensure_smalltrack_path()
  _stub_backbone_package()
  _stub_optional_viz_imports()
  from siamban.models.backbone.DWT.wad_module import wad_module  # noqa: WPS433

  return wad_module


def import_gal_class() -> Type:
  """Return the original SmallTrack GAL class."""
  _ensure_smalltrack_path()
  from siamban.models.gal.gal import GAL  # noqa: WPS433

  return GAL
