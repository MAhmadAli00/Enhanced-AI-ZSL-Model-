# Enhanced AI-Driven Zero-Shot Learning for Android Malware Detection

Source code for the paper **"Enhanced AI-Driven Zero-Shot Learning for Android Malware Detection with Hybrid Clustering"** submitted to *The Computer Journal*.

## Requirements

- Python 3.9 or 3.10 (required for TensorFlow 2.20 compatibility)
- Install all dependencies with pinned versions:

```bash
pip install -r requirements.txt
```

## Dataset

This project uses the **CICMalDroid2020** dataset (system-call / binder frequency features).

1. Download from the Canadian Institute for Cybersecurity:
   https://www.unb.ca/cic/datasets/maldroid-2020.html
2. Place the file `feature_vectors_syscallsbinders_frequency_5_Cat.csv` in the project root (same directory as `main.py`).

The CSV must contain a `Class` column with integer labels 1–5:
- 1: Banking malware
- 2: Benign apps
- 3: SMS malware
- 4: Riskware (held out as *unseen* by default)
- 5: Adware (held out as *unseen* by default)

## Reproducing Paper Results

### Full pipeline (EDA + training + single-seed ZSL evaluation)

```bash
python main.py --epochs 10 --unseen_classes 4 5 --eda
```

What this does:
1. **EDA** — saves plots to `plots/` (class distribution, feature importance, t-SNE, PCA-3D)
2. **Split** — class-disjoint ZSL split: seen = {1,2,3}, unseen = {4,5}
3. **Train** — deep triplet network with semi-hard loss (BatchNorm + L2 reg)
4. **Evaluate** — 5-shot ZSL on unseen classes; prints Accuracy / Precision / Recall / F1

Expected output: ~69.8 % zero-shot accuracy (seed 42 support-set draw).

### Multi-seed variance evaluation (Table in §5)

Trains once, then evaluates across 5 independent support-set draws (seeds: 42, 123, 2024, 7, 1337):

```bash
python src/run_multi_seed.py --epochs 10 --unseen_classes 4 5
```

Or via the main entry point:

```bash
python main.py --epochs 10 --unseen_classes 4 5 --multi_seed
```

Outputs:
- Console table: per-seed metrics + mean ± std + 95 % CI
- `results/multi_seed_zsl.csv` — per-seed rows + aggregate
- `results/multi_seed_zsl.json` — full stats for manuscript table

### Supervised baselines (Table comparing RF / XGBoost upper bounds)

```bash
python src/ablations.py --unseen_classes 4 5
```

### Computational cost benchmark (training time + inference latency)

```bash
python src/benchmark.py
```

## Project Structure

```
Ahmad_Project/
├── main.py                        Entry point
├── requirements.txt               Pinned dependencies
├── feature_vectors_...Cat.csv     Dataset (not tracked by git)
└── src/
    ├── models.py                  Deep Triplet Network + HybridClusterModel
    ├── data_loader.py             Data loading, ZSL split, SMOTE
    ├── train.py                   Triplet semi-hard loss training loop
    ├── evaluate.py                Few-shot ZSL evaluation (seed-aware)
    ├── run_multi_seed.py          Multi-seed variance driver
    ├── benchmark.py               Computational cost analysis
    ├── ablations.py               Supervised baselines (RF, XGBoost)
    ├── losses.py                  Triplet semi-hard loss implementation
    ├── utils.py                   Metrics + confusion matrix
    └── visualization.py           EDA and training history plots
```

## License

MIT — see [LICENSE](LICENSE).
