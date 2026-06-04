import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def relation_rows(array: np.ndarray, top_m: int) -> np.ndarray:
    if array.ndim != 4:
        raise ValueError(f"Expected [B, T, C, C], got shape {array.shape}")

    rows = array.reshape(-1, array.shape[-1])
    rows = np.nan_to_num(rows, nan=0.0, posinf=0.0, neginf=0.0)

    if top_m > 0 and top_m < rows.shape[1]:
        sparse = np.zeros_like(rows)
        keep = np.argpartition(rows, -top_m, axis=1)[:, -top_m:]
        row_ids = np.arange(rows.shape[0])[:, None]
        sparse[row_ids, keep] = rows[row_ids, keep]
        rows = sparse

    total = rows.sum(axis=1, keepdims=True)
    rows = np.divide(rows, total, out=np.zeros_like(rows), where=total > 0)
    return rows


def nearest_error(samples: np.ndarray, centers: np.ndarray) -> float:
    dist = np.sum((samples[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    return float(np.mean(np.min(dist, axis=1)))


def random_center_error(samples: np.ndarray, k: int, repeats: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    errors = []
    for _ in range(repeats):
        idx = rng.choice(samples.shape[0], size=k, replace=False)
        errors.append(nearest_error(samples, samples[idx]))
    return float(np.mean(errors))


def plot_codewords(centers: np.ndarray, output_path: Path, title: str) -> None:
    columns = [f"v{i}" for i in range(centers.shape[1])]
    plt.figure(figsize=(max(7, centers.shape[1] * 0.45), max(3.5, centers.shape[0] * 0.35)))
    plt.imshow(centers, aspect="auto", cmap="viridis")
    plt.colorbar(label="relation weight")
    plt.yticks(np.arange(centers.shape[0]), [f"B{i}" for i in range(centers.shape[0])])
    plt.xticks(np.arange(len(columns)), columns, rotation=60, ha="right", fontsize=8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--array-key", default="history")
    parser.add_argument("--output-dir", default="outputs_exported_cluster")
    parser.add_argument("--top-m", type=int, default=3)
    parser.add_argument("--k", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(input_path)
    if args.array_key not in data.files:
        raise KeyError(f"{args.array_key} not found. Available arrays: {data.files}")

    array = data[args.array_key]
    rows = relation_rows(array, args.top_m)

    split = int(rows.shape[0] * 0.7)
    train_x = rows[:split]
    test_x = rows[split:]

    results = []
    for k in args.k:
        if k >= train_x.shape[0]:
            continue
        model = KMeans(n_clusters=k, random_state=args.seed, n_init=20)
        train_labels = model.fit_predict(train_x)
        centers = model.cluster_centers_

        train_err = nearest_error(train_x, centers)
        test_err = nearest_error(test_x, centers)
        rand_err = random_center_error(train_x, k, repeats=30, seed=args.seed + k)
        sil = float(silhouette_score(train_x, train_labels)) if k > 1 else 0.0
        usage = np.bincount(train_labels, minlength=k) / len(train_labels)

        plot_codewords(
            centers,
            output_dir / f"{args.array_key}_codewords_k{k}.png",
            f"{args.array_key} relation codeword candidates (K={k})",
        )
        pd.DataFrame(centers).to_csv(output_dir / f"{args.array_key}_codewords_k{k}.csv", index=False)

        results.append(
            {
                "input": str(input_path),
                "array_key": args.array_key,
                "array_shape": list(array.shape),
                "relation_rows": int(rows.shape[0]),
                "variables": int(rows.shape[1]),
                "top_m": int(args.top_m),
                "k": int(k),
                "train_error": train_err,
                "test_error": test_err,
                "test_to_train_error_ratio": float(test_err / train_err) if train_err > 0 else None,
                "random_center_error": rand_err,
                "kmeans_vs_random_improvement": float((rand_err - train_err) / rand_err) if rand_err > 0 else None,
                "silhouette": sil,
                "min_usage": float(np.min(usage)),
                "max_usage": float(np.max(usage)),
            }
        )

    summary = pd.DataFrame(results)
    summary.to_csv(output_dir / f"{args.array_key}_summary.csv", index=False)
    (output_dir / f"{args.array_key}_summary.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
