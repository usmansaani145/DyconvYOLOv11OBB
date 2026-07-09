"""Build WaveGEM-OBB: YOLOv11-OBB + WPL backbone + GEM head."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import torch
from ultralytics import YOLO

from models.inject_gem import inject_gem
from models.inject_wpl import inject_wpl

REPO = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = "pretrained/yolo11m-obb.pt"
P2_YAML = str(REPO / "configs" / "yolov11_p2.yaml")


def _torch_load_checkpoint(weights_path: str):
  """Load Ultralytics .pt checkpoint (requires weights_only=False on PyTorch 2.6+)."""
  return torch.load(weights_path, map_location="cpu", weights_only=False)


def _transfer_pretrained(model: YOLO, weights_path: str) -> int:
  state = _torch_load_checkpoint(weights_path)
  if isinstance(state, dict) and "model" in state:
    state = state["model"].state_dict()
  model_state = model.model.state_dict()
  matched = {
      k: v
      for k, v in state.items()
      if k in model_state and model_state[k].shape == v.shape
  }
  model_state.update(matched)
  model.model.load_state_dict(model_state, strict=False)
  return len(matched)


def _load_finetuned_yolo(weights_path: str) -> YOLO:
  """Load trained weights as-is; checkpoint already includes WPL/GEM/P2 modules."""
  model = YOLO(weights_path)
  print(f"Loaded fine-tuned checkpoint: {weights_path}")
  return model


def _has_wpl(model: YOLO) -> bool:
  from modules.wpl_adapted import WPLYOLOAdapter

  for m in model.model.model:
    if isinstance(m, WPLYOLOAdapter):
      return True
  return False


def build_wavegem_obb(
    weights: str = DEFAULT_WEIGHTS,
    use_wpl: bool = True,
    use_gem: bool = True,
    p2_yaml: Optional[str] = None,
    gem_scales: Optional[List[int]] = None,
    wpl_max_replace: int = 2,
    gem_gamma_init: float = 0.0,
):
  """
  Build WaveGEM-OBB model.

  Args:
      weights:   pretrained or fine-tuned checkpoint (.pt)
      use_wpl:   inject WPL into backbone
      use_gem:   inject GEM into detection head
      p2_yaml:   if provided, use P2 YAML config (4-scale head)
      gem_scales: cv3 scale indices for GEM; auto [1,2] with P2 else [0,1]

  Returns:
      Ultralytics YOLO model with WPL and/or GEM injected
  """
  weights_path = Path(weights)
  is_finetuned = weights_path.exists() and str(weights) != DEFAULT_WEIGHTS

  if is_finetuned:
    model = _load_finetuned_yolo(str(weights_path))
  else:
    if p2_yaml:
      model = YOLO(p2_yaml)
      n = _transfer_pretrained(model, DEFAULT_WEIGHTS)
      print(f"Transferred {n} layers from {DEFAULT_WEIGHTS}")
    else:
      model = YOLO(DEFAULT_WEIGHTS)

    if use_wpl and not _has_wpl(model):
      model = inject_wpl(model, max_replace=wpl_max_replace)

    if use_gem:
      if gem_scales is None:
        gem_scales = [1, 2] if p2_yaml else [0, 1]
      model = inject_gem(model, apply_scales=gem_scales, gamma_init=gem_gamma_init)

  if torch.cuda.is_available():
    model.model.cuda()

    dummy = torch.randn(2, 3, 1024, 1024).cuda()
    with torch.no_grad():
      out = model.model(dummy)

    outputs = out if isinstance(out, (list, tuple)) else [out]
    for i, o in enumerate(outputs):
      if isinstance(o, torch.Tensor):
        assert not torch.isnan(o).any(), f"NaN in output tensor {i}"
    print("Forward pass:  PASSED (no NaN)")
  else:
    print("CUDA not available — skipping GPU forward sanity check")

  params = sum(p.numel() for p in model.model.parameters()) / 1e6
  baseline_params = 25.9
  print("\n=== WaveGEM-OBB build complete ===")
  print(f"WPL injected:  {use_wpl}")
  print(f"GEM injected:  {use_gem}")
  print(f"P2 head:       {p2_yaml is not None}")
  print(
      f"Parameters:    {params:.1f}M (+{params - baseline_params:.1f}M vs baseline)"
  )

  return model
