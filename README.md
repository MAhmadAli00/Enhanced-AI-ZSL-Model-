# Enhanced AI-Driven Zero-Shot Learning for Android Malware Detection

This project implements a Zero-Shot Learning (ZSL) framework to detect **unseen** Android malware families (Zero-Day attacks) using a Deep Triplet Network with Hybrid Clustering.

## 1. Setup

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## 2. Running the Main Pipeline

To run the full pipeline (EDA + Training + ZSL Evaluation) with the settings used for the paper:

```bash
python main.py --epochs 10 --unseen_classes 4 5 --eda
```

### What this does:
1.  **EDA**: Generates plots in `plots/` (Class Distribution, feature importance, t-SNE).
2.  **Split**: Splits data into **Seen** (Classes 1,2,3) and **Unseen** (Classes 4,5).
3.  **Train**: Trains the Deep Embedding Model (with BatchNorm & L2 Reg) on Seen classes.
4.  **Evaluate**: Tests 5-shot ZSL performance on Unseen classes.
5.  **Output**: Saves results to console and diagrams (Confusion Matrix) to disk.

## 3. Running Baselines (for Paper Comparison)

To get the "Supervised" upper-bound numbers (e.g., the ~95% XGBoost result):

```bash
python src/ablations.py --unseen_classes 4 5
```

## 4. Running Benchmarks (Computational Cost)

To get the training time and inference latency (ms/sample) stats:

```bash
python src/benchmark.py
```

## Project Structure
*   `src/models.py`: Deep Triplet Network architecture.
*   `src/train.py`: Training loop with Triplet Semi-Hard Loss.
*   `src/evaluate.py`: Few-Shot ZSL evaluation protocol.
*   `plots/`: Generated figures.
