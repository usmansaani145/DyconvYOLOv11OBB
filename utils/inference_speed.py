#!/usr/bin/env python3
"""
Table 4 — Inference speed for KBS paper (12 models, seed-42 best checkpoints).

Run in two Docker phases on the SAME GPU (see run_inference_speed_docker.sh):
  --group mmrotate     Oriented R-CNN + S2ANet  (bit:5000/mmrotate:0.3.4-cu117)
  --group ultralytics  10 Ultralytics models    (bit:5000/ship_aug:cu118)
  --print-table        merge partial JSONs and print table (no GPU)
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

REPO = Path(".")
OUT = Path(".")
RUNS_DIR = OUT / "runs"
RESULTS_DIR = OUT / "results"

WARMUP = 50
N_RUNS = 250
TIMED = 200
IMG = 1024
SEED = 42

PARTIAL_MM = RESULTS_DIR / "inference_speed_partial_mmrotate.json"
PARTIAL_ULTRA = RESULTS_DIR / "inference_speed_partial_ultralytics.json"
OUT_JSON = RESULTS_DIR / "inference_speed.json"
OUT_CSV = RESULTS_DIR / "inference_speed.csv"
OUT_TXT = RESULTS_DIR / "inference_speed_table.txt"

PROTOCOL = {
    "hardware": "NVIDIA A40 47.6GB",
    "batch_size": 1,
    "image_size": "1024x1024",
    "precision": "float16 (latency); float32 (FLOPs via thop/fvcore)",
    "warmup_passes": WARMUP,
    "timed_passes": TIMED,
    "synchronization": "torch.cuda.synchronize()",
    "cudnn_benchmark": False,
    "footnote": (
        "Inference speed measured on a single NVIDIA A40 GPU (47.6 GB VRAM) with "
        "batch size 1 and input resolution 1024×1024. Latency reported as mean ± "
        "standard deviation over 200 forward passes following 50-pass GPU warmup. "
        "FLOPs computed using thop (Ultralytics) or fvcore (MMRotate) on float32 "
        "input (MACs × 2 = FLOPs). All models evaluated sequentially on the same "
        "GPU within a single PBS job to ensure comparability."
    ),
}


class ModelSpec(object):
  def __init__(self, key, display, group, model_type, weight=None, config=None,
               is_baseline=False, is_proposed=False):
    self.key = key
    self.display = display
    self.group = group
    self.model_type = model_type
    self.weight = weight
    self.config = config
    self.is_baseline = is_baseline
    self.is_proposed = is_proposed


def _spec(name, display, group, model_type, weight=None, config=None,
          is_baseline=False, is_proposed=False):
  return ModelSpec(name, display, group, model_type, weight, config,
                   is_baseline, is_proposed)


def _specs() -> List[ModelSpec]:
  r = RUNS_DIR
  return [
      _spec("orcnn", "Oriented R-CNN", "mmrotate", "2-stage",
            config=REPO / "baselines/orcnn/config_orcnn_v2.py"),
      _spec("s2anet", "S2ANet", "mmrotate", "1-stage",
            config=REPO / "baselines/s2anet/config_s2anet_v2.py"),
      _spec("yolov8_obb", "YOLOv8-OBB", "ultralytics", "YOLO",
            weight=r / "yolov8_obb/yolov8_obb_seed42/weights/best.pt"),
      _spec("yolov11_obb", "YOLOv11-OBB †", "ultralytics", "YOLO",
            weight=r / "yolov11_obb/yolov11_obb_seed42/weights/best.pt",
            is_baseline=True),
      _spec("yolov26_obb", "YOLOv26-OBB", "ultralytics", "YOLO",
            weight=r / "yolov26_obb_medium/yolov26_obb_medium_seed42/train/weights/best.pt"),
      _spec("ablation_A_p2", "YOLOv11 + P2 head", "ultralytics", "ablation",
            weight=r / "ablation_A_p2/ablation_A_p2_seed42/train/weights/best.pt"),
      _spec("ablation_C_gem_gammafix", "YOLOv11 + GEM", "ultralytics", "ablation",
            weight=r / "ablation_C_gem_gammafix/ablation_C_gem_gammafix_seed42/train/weights/best.pt"),
      _spec("idea1_lska", "YOLOv11 + LSKA", "ultralytics", "idea",
            weight=r / "idea1_lska/idea1_lska_seed42/train/weights/best.pt"),
      _spec("idea2_ema", "YOLOv11 + EMA", "ultralytics", "idea",
            weight=r / "idea2_ema/idea2_ema_seed42/train/weights/best.pt"),
      _spec("idea3_msda", "YOLOv11 + MSDA", "ultralytics", "idea",
            weight=r / "idea3_msda/idea3_msda_seed42/train/weights/best.pt"),
      _spec("idea4_dyconv", "YOLOv11 + DyConv ★", "ultralytics", "PROPOSED",
            weight=r / "idea4_dyconv/idea4_dyconv_seed42/train/weights/best.pt",
            is_proposed=True),
      _spec("idea5_csd_lkca_msdp", "YOLOv11 + CSD", "ultralytics", "idea",
            weight=r / "idea5_csd_lkca_msdp/idea5_csd_lkca_msdp_seed42/train/weights/best.pt"),
  ]


def check_weights(specs: Optional[List[ModelSpec]] = None) -> List[str]:
  specs = specs or _specs()
  missing: List[str] = []
  for s in specs:
    if s.group == "ultralytics":
      if s.weight is None or not s.weight.is_file():
        alt = None
        if s.weight:
          alt = s.weight.parent / "last.pt"
        if alt and alt.is_file():
          print(f"  NOTE {s.display}: best.pt missing, will use last.pt")
        else:
          missing.append(f"{s.display}: {s.weight}")
    elif s.group == "mmrotate":
      try:
        _find_mmrotate_ckpt(s.key)
      except FileNotFoundError as exc:
        missing.append(str(exc))
  return missing


def _find_mmrotate_ckpt(model: str, seed: int = SEED) -> Path:
  work_dir = RUNS_DIR / model / f"{model}_seed{seed}"
  for name in ("epoch_100.pth", "latest.pth", "best_mAP.pth", "best_bbox_mAP.pth"):
    p = work_dir / name
    if p.is_file():
      return p.resolve() if p.is_symlink() else p
  epochs = sorted(work_dir.glob("epoch_*.pth"), key=lambda p: int(p.stem.split("_")[1]))
  if epochs:
    return epochs[-1]
  raise FileNotFoundError(f"No MMRotate checkpoint in {work_dir}")


def _setup_cuda() -> None:
  import torch
  if not torch.cuda.is_available():
    raise RuntimeError("CUDA required")
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = False
  torch.cuda.empty_cache()


def _count_params_m(module) -> float:
  return sum(p.numel() for p in module.parameters() if p.requires_grad) / 1e6


def _gflops_thop(module, device, use_forward_dummy=False) -> Tuple[Optional[float], float]:
  import torch
  import torch.nn as nn

  params_m = _count_params_m(module)
  x32 = torch.randn(1, 3, IMG, IMG, device=device, dtype=torch.float32)

  prof_module = module
  if use_forward_dummy and hasattr(module, "forward_dummy"):
    class _DummyWrap(nn.Module):
      def __init__(self, m):
        super().__init__()
        self.m = m

      def forward(self, x):
        return self.m.forward_dummy(x)

    prof_module = _DummyWrap(module)

  try:
    from thop import profile

    macs, params = profile(prof_module, inputs=(x32,), verbose=False)
    return float(macs) * 2.0 / 1e9, float(params) / 1e6
  except Exception as exc:
    print(f"    thop failed: {exc}")
  if not use_forward_dummy:
    try:
      from fvcore.nn import FlopCountAnalysis

      flops = FlopCountAnalysis(prof_module, x32).total()
      return float(flops) / 1e9, params_m
    except Exception as exc:
      print(f"    fvcore failed: {exc}")
  return None, params_m


def _time_forward(forward_fn: Callable[[], Any], use_half: bool = True) -> Dict[str, float]:
  import numpy as np
  import torch

  times: List[float] = []
  with torch.no_grad():
    for i in range(N_RUNS):
      torch.cuda.synchronize()
      t0 = time.perf_counter()
      forward_fn()
      torch.cuda.synchronize()
      t1 = time.perf_counter()
      if i >= WARMUP:
        times.append((t1 - t0) * 1000.0)
  mean_ms = float(np.mean(times))
  std_ms = float(np.std(times))
  return {
      "latency_mean_ms": mean_ms,
      "latency_std_ms": std_ms,
      "fps": 1000.0 / mean_ms,
  }


def _purge_module_cache(prefixes):
  import sys
  for key in list(sys.modules):
    for prefix in prefixes:
      if key == prefix or key.startswith(prefix + "."):
        del sys.modules[key]
        break


def _preload_idea_modules(idea_root: Path, submodules: Tuple[str, ...]) -> None:
  """Load idea modules/*.py without triggering REPO/modules/__init__.py."""
  import importlib.util
  import sys
  import types

  modules_dir = idea_root / "modules"
  if not modules_dir.is_dir():
    return
  pkg = types.ModuleType("modules")
  pkg.__path__ = [str(modules_dir)]
  sys.modules["modules"] = pkg
  for sub in submodules:
    path = modules_dir / f"{sub}.py"
    if not path.is_file():
      continue
    full = f"modules.{sub}"
    spec = importlib.util.spec_from_file_location(full, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[full] = mod
    spec.loader.exec_module(mod)


def _register_ultralytics_modules(spec_key: str) -> None:
  """Match training PYTHONPATH: idea root (or repo) first, then import inject module."""
  import importlib
  import sys

  _purge_module_cache(["models", "modules"])

  path_map = {
      "ablation_C_gem_gammafix": [REPO, REPO / "SmallTrack"],
      "idea1_lska": [REPO / "ideas/idea1_lska"],
      "idea2_ema": [REPO / "ideas/idea2_ema"],
      "idea3_msda": [REPO / "ideas/idea3_msda"],
      "idea4_dyconv": [REPO / "ideas/idea4_dyconv"],
      "idea5_csd_lkca_msdp": [REPO / "ideas/idea5_csd_lkca_msdp"],
  }
  import_map = {
      "ablation_C_gem_gammafix": "models.inject_gem",
      "idea1_lska": "models.inject_lska",
      "idea2_ema": "models.inject_ema",
      "idea3_msda": "models.inject_msda_neck",
      "idea4_dyconv": "models.inject_dyconv",
      "idea5_csd_lkca_msdp": "models.inject_csd",
  }
  if spec_key not in path_map:
    return

  repo_paths = [str(REPO), str(REPO / "SmallTrack")]
  managed = set(repo_paths)
  for paths in path_map.values():
    for p in paths:
      managed.add(str(p))
  for p in list(sys.path):
    if p in managed:
      sys.path.remove(p)

  idea_paths = [str(x) for x in path_map[spec_key]]
  if spec_key == "ablation_C_gem_gammafix":
    for p in reversed(repo_paths + idea_paths):
      sys.path.insert(0, p)
  else:
    for p in reversed(idea_paths):
      sys.path.insert(0, p)

  if spec_key == "idea3_msda":
    _preload_idea_modules(REPO / "ideas/idea3_msda", ("msda", "cspstage"))

  importlib.import_module(import_map[spec_key])


def _patch_dynamic_conv(net) -> None:
  """Fix DyConv checkpoints pickled before out_ch was stored on instances."""
  for m in net.modules():
    if type(m).__name__ != "DynamicConv":
      continue
    if hasattr(m, "out_ch"):
      continue
    w = getattr(m, "weight", None)
    if w is None or w.dim() != 5:
      continue
    m.K = int(w.shape[0])
    m.out_ch = int(w.shape[1])
    m.in_ch = int(w.shape[2])
    m.kernel_size = int(w.shape[-1])
    if not hasattr(m, "stride"):
      m.stride = 1


def measure_ultralytics(spec: ModelSpec) -> Dict[str, Any]:
  import torch
  from ultralytics import YOLO

  weight = spec.weight
  if weight is None or not weight.is_file():
    last = weight.parent / "last.pt" if weight else None
    if last and last.is_file():
      weight = last
    else:
      raise FileNotFoundError(f"Missing weights for {spec.display}: {spec.weight}")

  print(f"\n=== {spec.display} ===")
  print(f"  Weights: {weight}")

  _register_ultralytics_modules(spec.key)
  yolo = YOLO(str(weight))
  net = yolo.model.cuda().eval()
  _patch_dynamic_conv(net)
  params_m = _count_params_m(net)

  # Latency before thop — thop hooks break subsequent forward passes (e.g. GEM).
  try:
    net_h = net.half()
    x = torch.randn(1, 3, IMG, IMG, device="cuda", dtype=torch.float16)

    def _fwd():
      net_h(x)

    timing = _time_forward(_fwd, use_half=True)
    precision = "float16"
  except Exception as exc:
    print(f"    float16 forward failed ({exc}), using float32")
    net_f = net.float()
    x = torch.randn(1, 3, IMG, IMG, device="cuda", dtype=torch.float32)

    def _fwd_f():
      net_f(x)

    timing = _time_forward(_fwd_f, use_half=False)
    precision = "float32"

  gflops = None
  try:
    prof_net = net.float().cuda().eval()
    gflops, params_m_thop = _gflops_thop(prof_net, torch.device("cuda"))
    if params_m_thop > 0:
      params_m = params_m_thop
  except Exception as exc:
    print(f"    GFLOPs profiling skipped: {exc}")

  result = {
      "display": spec.display,
      "type": spec.model_type,
      "params_M": params_m,
      "gflops": gflops,
      "gflops_note": None if gflops is not None else "N/A (custom ops)",
      "precision_latency": precision,
      "weights": str(weight),
      **timing,
  }
  print(
      f"  Params={params_m:.1f}M  GFLOPs={gflops if gflops else 'N/A'}  "
      f"Latency={timing['latency_mean_ms']:.1f}±{timing['latency_std_ms']:.1f}ms  "
      f"FPS={timing['fps']:.1f}"
  )
  return result


def _load_mmrotate_detector(config_path, ckpt_path, device="cuda:0"):
  """Build MMRotate detector (compatible with mmrotate 0.3.x Docker image)."""
  import importlib
  import mmrotate  # noqa: F401 — register OrientedRCNN, S2ANet, etc.
  import torch
  from mmcv import Config
  from mmcv.runner import load_checkpoint

  cfg = Config.fromfile(str(config_path))
  if cfg.get("custom_imports"):
    from mmcv.utils import import_modules_from_strings
    import_modules_from_strings(**cfg["custom_imports"])

  cfg.model.pretrained = None
  cfg.model.train_cfg = None

  model = None
  last_err = None
  for mod_name in ("mmdet.models", "mmrotate.models"):
    try:
      mod = importlib.import_module(mod_name)
      build_fn = getattr(mod, "build_detector")
      model = build_fn(cfg.model, test_cfg=cfg.get("test_cfg"))
      print(f"  Built detector via {mod_name}.build_detector")
      break
    except Exception as exc:
      last_err = exc
      print(f"  {mod_name}.build_detector failed: {exc}")

  if model is None:
    raise RuntimeError("Could not build MMRotate detector: %s" % last_err)

  load_checkpoint(model, str(ckpt_path), map_location="cpu")
  model.cfg = cfg
  model.to(device)
  model.eval()
  return model, cfg


def _make_mmrotate_forward(model, img_tensor):
  """Return callable + description for timed forward passes."""
  if hasattr(model, "forward_dummy"):
    def _fwd():
      model.forward_dummy(img_tensor)
    return _fwd, "forward_dummy (backbone+neck+heads, no NMS)"

  import numpy as np

  img_meta = [{
      "ori_shape": (IMG, IMG, 3),
      "img_shape": (IMG, IMG, 3),
      "pad_shape": (IMG, IMG, 3),
      "scale_factor": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
      "flip": False,
      "img_norm_cfg": {"mean": np.zeros(3), "std": np.ones(3), "to_rgb": True},
  }]
  data = {"img": [img_tensor], "img_metas": [img_meta]}

  def _fwd():
    try:
      model(return_loss=False, rescale=True, **data)
    except Exception:
      model.test_step(data)

  return _fwd, "full test forward (includes postproc)"


def measure_mmrotate(spec: ModelSpec) -> Dict[str, Any]:
  import torch

  sys.path.insert(0, "/opt/mmrotate")
  sys.path.insert(0, str(REPO))

  ckpt = _find_mmrotate_ckpt(spec.key)
  config = spec.config
  if config is None or not config.is_file():
    raise FileNotFoundError(f"Missing config for {spec.key}: {config}")

  print(f"\n=== {spec.display} ===")
  print(f"  Config:  {config}")
  print(f"  Weights: {ckpt}")

  detector, _cfg = _load_mmrotate_detector(config, ckpt, device="cuda:0")

  params_m = _count_params_m(detector)
  gflops, params_m_g = _gflops_thop(detector, torch.device("cuda"), use_forward_dummy=True)
  if params_m_g > 0:
    params_m = params_m_g

  img = torch.randn(1, 3, IMG, IMG, device="cuda", dtype=torch.float32)
  forward_fn, forward_note = _make_mmrotate_forward(detector, img)
  timing = _time_forward(forward_fn, use_half=False)
  precision = "float32"

  result = {
      "display": spec.display,
      "type": spec.model_type,
      "params_M": params_m,
      "gflops": gflops,
      "gflops_note": None if gflops is not None else "N/A (custom ops)",
      "precision_latency": precision,
      "weights": str(ckpt),
      "note": forward_note,
      **timing,
  }
  print(
      f"  Params={params_m:.1f}M  GFLOPs={gflops if gflops else 'N/A'}  "
      f"Latency={timing['latency_mean_ms']:.1f}±{timing['latency_std_ms']:.1f}ms  "
      f"FPS={timing['fps']:.1f}  ({forward_note})"
  )
  return result


def run_group(group: str) -> Dict[str, Any]:
  import torch

  _setup_cuda()
  print(f"GPU: {torch.cuda.get_device_name(0)}")
  specs = [s for s in _specs() if s.group == group]
  missing = check_weights(specs)
  if missing:
    raise FileNotFoundError("Missing weights:\n  " + "\n  ".join(missing))

  partial_path = PARTIAL_MM if group == "mmrotate" else PARTIAL_ULTRA
  results: Dict[str, Any] = {}
  if partial_path.is_file():
    data = json.loads(_read_text(partial_path))
    results = dict(data.get("models", {}))
    if results:
      print(f"Resuming {group}: {len(results)} model(s) already in {partial_path}")

  for spec in specs:
    if spec.key in results and "error" not in results.get(spec.key, {}):
      print(f"\n=== {spec.display} === SKIPPED (already measured)")
      continue
    torch.cuda.empty_cache()
    try:
      if group == "ultralytics":
        results[spec.key] = measure_ultralytics(spec)
      else:
        results[spec.key] = measure_mmrotate(spec)
    except Exception as exc:
      print(f"ERROR measuring {spec.display}: {exc}", file=sys.stderr)
      results[spec.key] = {
          "display": spec.display,
          "type": spec.model_type,
          "error": str(exc),
      }
    save_partial(partial_path, group, results)
    torch.cuda.empty_cache()
  return results


def _read_text(path: Path) -> str:
  with open(str(path), "r", encoding="utf-8") as f:
    return f.read()


def _write_text(path: Path, text: str) -> None:
  with open(str(path), "w", encoding="utf-8") as f:
    f.write(text)


def save_partial(path: Path, group: str, models: Dict[str, Any]) -> None:
  RESULTS_DIR.mkdir(parents=True, exist_ok=True)
  payload = {"protocol": PROTOCOL, "group": group, "models": models}
  _write_text(path, json.dumps(payload, indent=2))
  print(f"Saved partial: {path}")


def load_all_results() -> Dict[str, Any]:
  models: Dict[str, Any] = {}
  for path in (PARTIAL_MM, PARTIAL_ULTRA):
    if path.is_file():
      data = json.loads(_read_text(path))
      models.update(data.get("models", {}))
  return models


def _fmt_gflops(val: Optional[float], note: Optional[str]) -> str:
  if val is None or (isinstance(val, float) and val < 0):
    return note or "N/A"
  return f"{val:.1f}"


def print_table(models: Dict[str, Any]) -> str:
  order = [s.key for s in _specs()]
  lines = [
      "=========================================================================",
      "  Inference Speed Comparison — batch=1, imgsz=1024×1024, NVIDIA A40",
      "  Protocol: 50-pass warmup, 200-pass timed, torch.cuda.synchronize()",
      "  FLOPs via thop/fvcore (MACs×2). Latency = mean ± std over 200 runs.",
      "=========================================================================",
      f"{'Model':<24}{'Params(M)':>10}{'GFLOPs':>8}{'Latency(ms)':>14}{'FPS':>8}  {'Type':<10}",
      "-------------------------------------------------------------------------",
  ]
  for key in order:
    if key not in models:
      lines.append(f"{key:<24}  MISSING")
      continue
    m = models[key]
    if "error" in m:
      lines.append(f"{m.get('display', key):<24}  ERROR: {m['error']}")
      continue
    lat = f"{m['latency_mean_ms']:.1f} ± {m['latency_std_ms']:.1f}"
    g = _fmt_gflops(m.get("gflops"), m.get("gflops_note"))
    lines.append(
        f"{m['display']:<24}{m['params_M']:>10.1f}{g:>8}{lat:>14}{m['fps']:>8.1f}  {m['type']:<10}"
    )
  lines += [
      "-------------------------------------------------------------------------",
      "† Primary baseline.  ★ Proposed method.",
      "=========================================================================",
  ]

  base_key = "yolov11_obb"
  prop_key = "idea4_dyconv"
  if base_key in models and prop_key in models and "error" not in models[base_key] and "error" not in models[prop_key]:
    b, p = models[base_key], models[prop_key]
    dp = p["params_M"] - b["params_M"]
    pp = 100.0 * dp / b["params_M"] if b["params_M"] else 0.0
    lines.append("Delta (proposed vs baseline YOLOv11-OBB):")
    lines.append(f"  Params:  {dp:+.1f}M  ({pp:+.1f}% change)")
    if b.get("gflops") is not None and p.get("gflops") is not None and b["gflops"] > 0:
      dg = p["gflops"] - b["gflops"]
      gp = 100.0 * dg / b["gflops"]
      lines.append(f"  GFLOPs:  {dg:+.1f}  ({gp:+.1f}% change)")
    dl = p["latency_mean_ms"] - b["latency_mean_ms"]
    lp = 100.0 * dl / b["latency_mean_ms"] if b["latency_mean_ms"] else 0.0
    df = p["fps"] - b["fps"]
    fp = 100.0 * df / b["fps"] if b["fps"] else 0.0
    lines.append(f"  Latency: {dl:+.1f}ms ({lp:+.1f}% change)")
    lines.append(f"  FPS:     {df:+.1f}  ({fp:+.1f}% change)")
    lines.append("=========================================================================")

  lines.append("")
  lines.append(PROTOCOL["footnote"])
  text = "\n".join(lines)
  print(text)
  return text


def save_final(models: Dict[str, Any], table_text: str) -> None:
  RESULTS_DIR.mkdir(parents=True, exist_ok=True)
  payload = {"protocol": PROTOCOL, "models": models}
  _write_text(OUT_JSON, json.dumps(payload, indent=2))
  _write_text(OUT_TXT, table_text + "\n")

  order = [s.key for s in _specs()]
  with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow([
        "key", "display", "type", "params_M", "gflops", "gflops_note",
        "latency_mean_ms", "latency_std_ms", "fps", "precision_latency", "weights",
    ])
    for key in order:
      if key not in models:
        continue
      m = models[key]
      w.writerow([
          key, m.get("display"), m.get("type"), m.get("params_M"),
          m.get("gflops"), m.get("gflops_note"),
          m.get("latency_mean_ms"), m.get("latency_std_ms"), m.get("fps"),
          m.get("precision_latency"), m.get("weights"),
      ])
  print(f"Saved: {OUT_JSON}")
  print(f"Saved: {OUT_CSV}")
  print(f"Saved: {OUT_TXT}")


def merge_and_print() -> int:
  models = load_all_results()
  if not models:
    print("No partial results found.", file=sys.stderr)
    return 1
  table = print_table(models)
  save_final(models, table)
  return 0


def main() -> int:
  parser = argparse.ArgumentParser(description="Inference speed benchmark")
  parser.add_argument(
      "--group",
      choices=("mmrotate", "ultralytics", "check"),
      help="Which model group to measure (GPU required except check)",
  )
  parser.add_argument(
      "--print-table",
      action="store_true",
      help="Merge partial JSONs and print final table (no GPU)",
  )
  args = parser.parse_args()

  if args.print_table:
    return merge_and_print()

  if args.group == "check":
    missing = check_weights()
    if missing:
      print("MISSING:")
      for m in missing:
        print(f"  {m}")
      return 1
    print("All 12 weight paths OK.")
    return 0

  if args.group is None:
    parser.print_help()
    return 1

  try:
    models = run_group(args.group)
  except Exception as exc:
    print(f"FATAL: {exc}", file=sys.stderr)
    return 1

  if args.group == "mmrotate":
    save_partial(PARTIAL_MM, args.group, models)
  else:
    save_partial(PARTIAL_ULTRA, args.group, models)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
