# DyConv-YOLOv11: Oriented Ship Detection Benchmark

**Paper:** DyConv-YOLOv11: A Unified Benchmark and Systematic Module Evaluation for Oriented Ship Detection in Aerial Imagery  
**Journal:** Knowledge-Based Systems (Q1, Elsevier) — submitted   
**Authors:** Rana Muhammad Usman, Faizan Faiz, Ali Javaid, Muhammad Sajid, Feng Zhao (*corresponding*)

---

## Overview

This repository contains the complete code for reproducing all experiments in the paper. The key contributions are:

1. **Unified benchmark** — Five OBB detectors trained under identical conditions on DOTA 2.0 (ships-only)
2. **Systematic module evaluation** — Seven plug-in modules plus three combinations (>50 training runs)
3. **Optimizer confounding finding** — AdamW explains +0.008 AP; DyConv adds −0.001 AP vs AdamW control
4. **DyConv-YOLOv11** — Matched accuracy at −13.7% params, −21.0% GFLOPs vs YOLOv11-OBB

---

## Key Results

| Model | AP | AP50 | AP75 | APs | APm | APl |
|---|---|---|---|---|---|---|
| Oriented R-CNN | 0.127±.002 | 0.261 | 0.124 | 0.123 | 0.175 | 0.144 |
| S2ANet | 0.121±.001 | 0.262 | 0.112 | 0.113 | 0.179 | 0.115 |
| YOLOv8-OBB | 0.401±.002 | 0.581 | 0.487 | 0.354 | 0.630 | 0.340 |
| YOLOv11-OBB † | 0.401±.004 | 0.581 | 0.480 | 0.354 | 0.637 | 0.339 |
| YOLOv26-OBB | 0.411±.001 | 0.582 | 0.502 | 0.365 | 0.626 | 0.381 |
| **DyConv-YOLOv11 ★** | **0.408±.001** | **0.589** | **0.497** | **0.356** | **0.639** | 0.330 |

† Primary baseline   ★ Proposed method   All results: mean±std over 3 seeds (42, 123, 456)

Published numbers are archived in `results/all_models_stats.json`.

---

## Repository Structure

```
shipobb_project/
├── configs/              # paths.example.yaml — optional overrides
├── dataset/              # Downloaded from Hugging Face (gitignored; see below)
│   ├── train/            # 6624 SAHI-tiled training images
│   ├── val/              # 1006 validation images
│   ├── dota_test/        # DOTA 2.0 test images (cross-dataset eval)
│   └── hrsc2016/         # HRSC2016 images + Annotations (zero-shot eval)
├── scripts/
│   ├── download_dataset.sh   # Fetch DOTA-ShipBench from Hugging Face
│   └── dataset_pipeline/     # Optional rebuild scripts from raw DOTA
├── shared/               # Compatibility shim for legacy imports
├── pretrained/           # Download Ultralytics OBB weights here
├── runs/                 # Created during training (gitignored)
├── ideas/                # Module implementations (idea1–idea5)
│   ├── idea1_lska/       # Large-kernel spatial attention
│   ├── idea2_ema/        # Exponential moving average
│   ├── idea3_msda/       # Multi-scale deformable attention (neck)
│   ├── idea4_dyconv/     # DynamicConv mid-backbone (proposed)
│   └── idea5_csd/        # CSD + LKCA + MSDP
├── training/             # All train_*.py scripts (baselines, ideas, ablations, combos)
├── evaluation/           # COCO-OBB eval helpers
├── results/              # Published CSV/JSON metrics (no images or weights)
├── utils/                # experiment_runner, constants, collect_results, …
├── scripts/              # check_env.sh, download_dataset.sh, dataset_pipeline/
├── requirements.txt
├── environment.yml
└── README.md
```

---

## Prerequisites

### Hardware

- NVIDIA GPU with ≥11 GB VRAM (A40 / RTX 3090 recommended)
- CUDA 11.8, Python 3.10

### Dataset — [DOTA-ShipBench on Hugging Face](https://huggingface.co/datasets/usmansaani145/DOTA-ShipBench) (~18 GB)

The filtered DOTA 2.0 ships-only benchmark is hosted on Hugging Face as **[DOTA-ShipBench](https://huggingface.co/datasets/usmansaani145/DOTA-ShipBench)**. It is **not** included in this Git repository — download it once before training or evaluation:

```bash
bash scripts/download_dataset.sh
# or: python utils/download_dataset.py
# custom location: SHIP_OBB_DATA_ROOT=/path/to/data bash scripts/download_dataset.sh
```

This writes the full layout to `dataset/` (or `$SHIP_OBB_DATA_ROOT`):

| Path | Contents | Count |
|---|---|---|
| `dataset/train/` | SAHI-tiled training images + YOLO/MMRotate labels | 6,624 images |
| `dataset/val/` | Validation tiles + labels | 1,006 images |
| `dataset/dota_test/` | DOTA 2.0 test images (cross-dataset eval) | 191 images |
| `dataset/hrsc2016/` | HRSC2016 images + Annotations (zero-shot eval) | 1,680 images |
| `dataset/gt_coco_filtered.json` | Unified COCO-OBB val GT | — |
| `dataset/dataset.yaml` | Ultralytics data config (path-relative) | — |

Users must comply with the [DOTA](https://captain-whu.github.io/DOTA/) and [HRSC2016](http://www.escience.cn/people/gongjianying.html) licenses when using the data.

### Pretrained weights (download separately)

```bash
# Ultralytics OBB checkpoints (place in a local pretrained/ folder)
yolo11m-obb.pt
yolov8m-obb.pt
yolo26m-obb.pt
```

---

## Installation

### Option A — Conda (recommended)

```bash
git clone https://github.com/YOUR_ORG/shipobb.git
cd shipobb
conda env create -f environment.yml
conda activate shipobb
bash scripts/download_dataset.sh   # fetch DOTA-ShipBench from Hugging Face
bash scripts/check_env.sh
```

### Option B — pip

```bash
pip install -r requirements.txt
# For CUDA 11.8 wheels, install PyTorch from https://pytorch.org first if needed.
```

### MMRotate baselines (Oriented R-CNN, S2ANet)

```bash
pip install openmim
mim install mmcv-full==1.7.1
mim install mmrotate==0.3.4
```

---

## Quick Start (fully self-contained)

```bash
git clone https://github.com/YOUR_ORG/shipobb.git
cd shipobb
source scripts/env.sh          # sets all repo-relative paths
conda env create -f environment.yml && conda activate shipobb
# OR: pip install -r requirements.txt

bash scripts/download_dataset.sh   # ~18 GB from Hugging Face
bash scripts/check_env.sh

# Download pretrained weights → pretrained/
# yolo11m-obb.pt, yolov8m-obb.pt, yolo26n-obb.pt

# Train primary baseline (seed 42)
python training/train_yolov11.py 42

# Train proposed method
python training/train_idea4.py 42
```

**No cluster paths are required.** Download the dataset from [Hugging Face](https://huggingface.co/datasets/usmansaani145/DOTA-ShipBench) into `dataset/`; outputs go under `runs/` and `results/`.

---

## Configure Paths (optional)

By default everything resolves relative to the repo root via `utils/constants.py`:

| Path | Default |
|---|---|
| Training data | `dataset/` (download via `scripts/download_dataset.sh`) |
| Runs / checkpoints | `runs/` |
| Metrics | `results/` |
| DOTA test | `dataset/dota_test/` |
| HRSC2016 | `dataset/hrsc2016/` |
| Pretrained weights | `pretrained/` |

Override only if needed:

```bash
source scripts/env.sh
# or individually:
export SHIP_OBB_DATA_ROOT=/custom/path/dataset
export PRETRAINED_YOLO11=/custom/yolo11m-obb.pt
```

---

## Reproduction Pipeline

### Step 1 — Download dataset

Fetch **[DOTA-ShipBench](https://huggingface.co/datasets/usmansaani145/DOTA-ShipBench)** from Hugging Face:

```bash
bash scripts/download_dataset.sh
bash scripts/check_env.sh    # verify file counts
```

To rebuild from raw DOTA instead of using the hosted tiles (optional):

```bash
python scripts/dataset_pipeline/filter_dataset_v2.py
python scripts/dataset_pipeline/build_dataset_v2.py
python scripts/dataset_pipeline/build_gt_coco.py
```

**Training hyperparameters (all YOLO models):** 100 epochs, batch 8, imgsz 1024, SGD lr0=0.01, seeds {42, 123, 456}.

### Step 2 — Train baselines

| Model | Script | Output dir |
|---|---|---|
| YOLOv8-OBB | `training/train_yolov8.py <seed>` | `runs/yolov8_obb/` |
| YOLOv11-OBB † | `training/train_yolov11.py <seed>` | `runs/yolov11_obb/` |
| YOLOv26-OBB | `training/train_yolov26.py <seed>` | `runs/yolov26_obb/` |
| YOLOv11 + AdamW | `training/train_yolov11_adamw.py <seed>` | `runs/yolov11_adamw/` |
| S2ANet | MMRotate config (see `utils/mmrotate_*`) | `runs/s2anet/` |
| Oriented R-CNN | MMRotate config | `runs/orcnn/` |

Example:

```bash
for SEED in 42 123 456; do
  python training/train_yolov11.py $SEED
done
```

### Step 3 — Train module ideas (systematic evaluation)

| Idea | Module | Script |
|---|---|---|
| Idea 1 | LSKA | `training/train_idea1.py <seed>` |
| Idea 2 | EMA | `training/train_idea2.py <seed>` |
| Idea 3 | MSDA | `training/train_idea3.py <seed>` |
| **Idea 4** | **DyConv ★** | `training/train_idea4.py <seed>` |
| Idea 5 | CSD/LKCA/MSDP | `training/train_idea5.py <seed>` |

**DyConv injection (inference on trained weights):**

```python
from ultralytics import YOLO
from models.inject_dyconv import patch_dynamic_conv  # or ideas/idea4_dyconv/inject_dyconv.py

model = YOLO("runs/idea4_dyconv/.../weights/best.pt")
patch_dynamic_conv(model.model)
```

> Do **not** call `inject_dyconv()` on already-trained DyConv checkpoints — use `patch_dynamic_conv` only.

### Step 4 — Train combinations

```bash
python training/train_combo1.py <seed>   # DyConv + MSDA
python training/train_combo2.py <seed>   # DyConv + EMA
python training/train_combo3.py <seed>   # DyConv + MSDA + EMA
```

### Step 5 — Evaluate (COCO-OBB AP)

```bash
python evaluation/obb_coco_eval.py   # unified COCO eval wrapper
python utils/collect_results.py      # aggregate seeds → results/*.json
```

Compare your output to the published archive:

```bash
python -c "import json; print(json.load(open('results/all_models_stats.json'))['yolov11_obb']['AP'])"
```

---

## Expected Checkpoints

After full reproduction, verify these files exist under `$SHIP_OBB_RUNS_ROOT`:

```
yolov11_obb/yolov11_obb_seed42/weights/best.pt
idea4_dyconv/idea4_dyconv_seed42/train/weights/best.pt
s2anet/s2anet_seed42/...
```

And aggregated metrics:

```
results/all_models_stats.json
results/ablation_table.csv
results/inference_speed.csv
results/hrsc2016_zeroshot_combined.json   # if HRSC eval was run
```

---

## Ablation Studies

| Ablation | Script |
|---|---|
| A — P2 head | `training/train_ablation_A_p2.py` |
| B — WPL | `training/train_ablation_B_wpl.py` |
| C — GEM | `training/train_ablation_C_gem.py` |
| D — WPL+GEM | `training/train_ablation_D_wpl_gem.py` |
| E — Full stack | `training/train_ablation_E_full.py` |

Results: `results/ablation_table.csv`

---

## Authors

Rana Muhammad Usman^a^, Faizan Faiz^b^, Ali Javaid^c,d^, Muhammad Sajid^e^, Feng Zhao^a,\*^

^a^ Department of Automation, School of Information Science and Technology, University of Science and Technology of China, Hefei 230002, PR China  
^b^ National Center of Artificial Intelligence (NCAI), National University of Science and Technology (NUST), Islamabad 44000, Pakistan  
^c^ School of Mechanical and Manufacturing Engineering (SMME), National University of Sciences and Technology (NUST), Islamabad 44000, Pakistan  
^d^ NUTECH School of Engineering & Technology (NUSET), National University of Technology, Islamabad 44000, Pakistan  
^e^ School of Mechanical and Materials Engineering, University College Dublin (UCD), Dublin D04 V1W8, Ireland

\* Corresponding author. E-mail: [fzhao956@ustc.edu.cn](mailto:fzhao956@ustc.edu.cn)

---

## Acknowledgements

We acknowledge the support of the GPU cluster built by the MCC Lab of the Information Science and Technology Institution at the University of Science and Technology of China (USTC), as well as the fellowship program funded by the Alliance of National and International Science Organizations (ANSO), which supported this work at USTC.

---

## Citation

```bibtex
@article{dyconv_yolov11_2026,
  title   = {DyConv-{YOLOv11}: A Unified Benchmark and Systematic Module Evaluation for Oriented Ship Detection in Aerial Imagery},
  author  = {Usman, Rana Muhammad and Faiz, Faizan and Javaid, Ali and Sajid, Muhammad and Zhao, Feng},
  journal = {Knowledge-Based Systems},
  year    = {2026},
  note    = {submitted}
}
```

---

## License

Code: MIT (see `LICENSE`).  
The [DOTA-ShipBench](https://huggingface.co/datasets/usmansaani145/DOTA-ShipBench) dataset (DOTA 2.0 and HRSC2016 subsets) is subject to the respective dataset licenses.

---

## Troubleshooting

| Issue | Fix |
|---|---|
| Dataset not found | Run `bash scripts/download_dataset.sh` ([DOTA-ShipBench](https://huggingface.co/datasets/usmansaani145/DOTA-ShipBench)) |
| `inject_dyconv` import wrong module | Put `ideas/idea4_dyconv` **first** in `PYTHONPATH` |
| DyConv inference returns 0 detections | Use `patch_dynamic_conv(model.model)` on trained weights, not fresh `inject_dyconv()` |
| MMRotate OOM on 11 GB GPU | Set `samples_per_gpu=1`, image scale 768 |

For environment validation: `bash scripts/check_env.sh`
