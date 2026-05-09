"""
Multi-seed ZSL evaluation driver.

Trains the embedding model ONCE, then evaluates across N independent support-set draws
using different random seeds. Produces mean ± std (and 95% CI) for all metrics.

Usage:
    python src/run_multi_seed.py --data_path <csv> --unseen_classes 4 5 --epochs 10
"""

import os
import sys
import json
import argparse
import numpy as np
import scipy.stats as stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

os.environ['PYTHONHASHSEED'] = '42'
import tensorflow as tf
tf.keras.utils.set_random_seed(42)
try:
    tf.config.experimental.enable_op_determinism()
except AttributeError:
    pass

from src.data_loader import load_data, get_zsl_split, prepare_training_data
from src.train import train_embedding_model
from src.evaluate import evaluate_on_unseen

SEEDS = [42, 123, 2024, 7, 1337]


def run(data_path, unseen_classes, epochs, n_support, method, n_clusters):
    # Single training run
    X, y = load_data(data_path)
    X_seen, y_seen, X_unseen, y_unseen, _ = get_zsl_split(X, y, unseen_classes)
    X_train, y_train, X_val, y_val = prepare_training_data(X_seen, y_seen)

    input_shape = (X_train.shape[1],)
    embedding_model, history = train_embedding_model(
        X_train, y_train, X_val, y_val, input_shape, epochs=epochs
    )

    embedding_model.save("malware_embedding_model.h5")
    print("\nModel saved to malware_embedding_model.h5")

    # 5-seed evaluation loop
    os.makedirs("plots", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    rows = []
    print("\n{:<8} {:>10} {:>10} {:>10} {:>10}".format(
        "Seed", "Accuracy", "Precision", "Recall", "F1"))
    print("-" * 52)

    for seed in SEEDS:
        metrics = evaluate_on_unseen(
            embedding_model, X_unseen, y_unseen,
            n_support=n_support, n_clusters=n_clusters,
            method=method, seed=seed
        )
        rows.append({"seed": seed, **metrics})
        print("{:<8} {:>10.4f} {:>10.4f} {:>10.4f} {:>10.4f}".format(
            seed, metrics["accuracy"], metrics["precision"],
            metrics["recall"], metrics["f1"]))

    # Aggregate
    keys = ["accuracy", "precision", "recall", "f1"]
    agg = {}
    n = len(SEEDS)
    for k in keys:
        vals = np.array([r[k] for r in rows])
        mean = float(vals.mean())
        std = float(vals.std(ddof=1))
        sem = std / np.sqrt(n)
        ci95 = float(stats.t.ppf(0.975, df=n - 1) * sem)
        agg[k] = {"mean": mean, "std": std, "ci95_half": ci95,
                  "ci95_low": mean - ci95, "ci95_high": mean + ci95}

    print("\n── Aggregate (mean ± std, 95 % CI) ──────────────────────────────────")
    for k in keys:
        a = agg[k]
        print(f"  {k:<12} {a['mean']:.4f} ± {a['std']:.4f}  "
              f"  95% CI [{a['ci95_low']:.4f}, {a['ci95_high']:.4f}]")

    # Save results
    import csv
    csv_path = "results/multi_seed_zsl.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed"] + keys)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({
            "seed": "mean",
            **{k: f"{agg[k]['mean']:.4f}" for k in keys}
        })
        writer.writerow({
            "seed": "std",
            **{k: f"{agg[k]['std']:.4f}" for k in keys}
        })

    json_path = "results/multi_seed_zsl.json"
    with open(json_path, "w") as f:
        json.dump({"per_seed": rows, "aggregate": agg, "seeds": SEEDS}, f, indent=2)

    print(f"\nResults saved to {csv_path} and {json_path}")
    return agg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-seed ZSL variance evaluation")
    parser.add_argument("--data_path", type=str,
                        default="feature_vectors_syscallsbinders_frequency_5_Cat.csv")
    parser.add_argument("--unseen_classes", type=int, nargs="+", default=[4, 5])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--n_support", type=int, default=5)
    parser.add_argument("--method", type=str, default="kmeans",
                        choices=["kmeans", "gmm", "agglomerative"])
    parser.add_argument("--n_clusters", type=int, default=1)
    args = parser.parse_args()

    run(args.data_path, args.unseen_classes, args.epochs,
        args.n_support, args.method, args.n_clusters)
