# Dataset pipeline (optional rebuild from raw DOTA)

The pre-built **DOTA-ShipBench** tiles are hosted on Hugging Face:

https://huggingface.co/datasets/usmansaani145/DOTA-ShipBench

Download with:

```bash
bash scripts/download_dataset.sh
```

Use the scripts in this folder only if you want to rebuild the filtered dataset from raw DOTA instead of downloading the hosted version:

```bash
python scripts/dataset_pipeline/filter_dataset_v2.py
python scripts/dataset_pipeline/build_dataset_v2.py
python scripts/dataset_pipeline/build_gt_coco.py
```

Output is written to `dataset/` at the repository root.
