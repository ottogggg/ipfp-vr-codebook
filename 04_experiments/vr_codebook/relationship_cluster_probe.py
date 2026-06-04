import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler


DATASETS = {
    "ETTh1": {
        "path": "data/ETTh1.csv",
        "cycle": 24,
        "top_m": 3,
    },
    "weather": {
        "path": "data/weather.csv",
        "cycle": 144,
        "top_m": 5,
    },
}


def load_numeric_series(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(path)
    if "date" not in df.columns and all(_looks_numeric(col) for col in df.columns):
        df = pd.read_csv(path, header=None)
        df.columns = [f"v{i}" for i in range(df.shape[1])]

    numeric = df.drop(columns=["date"], errors="ignore")
    numeric = numeric.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    numeric = numeric.fillna(method="ffill").fillna(method="bfill")
    values = numeric.to_numpy(dtype=np.float64)
    values = StandardScaler().fit_transform(values)
    return numeric, values


def _looks_numeric(value) -> bool:
    try:
        float(value)
        return True
    except Exception:
        return False


def build_relation_rows(values: np.ndarray, cycle: int, top_m: int) -> tuple[np.ndarray, pd.DataFrame]:
    period_count = values.shape[0] // cycle
    usable = values[: period_count * cycle]
    rows = []
    meta = []

    for period_idx in range(period_count):
        segment = usable[period_idx * cycle : (period_idx + 1) * cycle]
        corr = np.corrcoef(segment, rowvar=False)
        corr = np.nan_to_num(np.abs(corr), nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(corr, 0.0)

        for var_idx in range(corr.shape[0]):
            row = corr[var_idx].copy()
            if top_m > 0 and top_m < row.size:
                keep = np.argpartition(row, -top_m)[-top_m:]
                mask = np.zeros_like(row, dtype=bool)
                mask[keep] = True
                row[~mask] = 0.0
            total = row.sum()
            if total > 0:
                row = row / total
            rows.append(row)
            meta.append(
                {
                    "period": period_idx,
                    "period_mod_7": period_idx % 7,
                    "variable": var_idx,
                }
            )

    return np.asarray(rows), pd.DataFrame(meta)


def nearest_error(samples: np.ndarray, centers: np.ndarray) -> float:
    diff = samples[:, None, :] - centers[None, :, :]
    dist = np.sum(diff * diff, axis=2)
    return float(np.mean(np.min(dist, axis=1)))


def random_center_error(samples: np.ndarray, k: int, repeats: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    errors = []
    for _ in range(repeats):
        idx = rng.choice(samples.shape[0], size=k, replace=False)
        errors.append(nearest_error(samples, samples[idx]))
    return float(np.mean(errors))


def plot_codewords(centers: np.ndarray, columns: list[str], title: str, path: Path) -> None:
    fig_width = max(7, min(18, len(columns) * 0.45))
    fig_height = max(3.5, centers.shape[0] * 0.35)
    plt.figure(figsize=(fig_width, fig_height))
    plt.imshow(centers, aspect="auto", cmap="viridis")
    plt.colorbar(label="relation weight")
    plt.yticks(np.arange(centers.shape[0]), [f"B{i}" for i in range(centers.shape[0])])
    plt.xticks(np.arange(len(columns)), columns, rotation=60, ha="right", fontsize=8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_position_usage(labels: np.ndarray, positions: np.ndarray, k: int, title: str, path: Path) -> None:
    usage = np.zeros((7, k), dtype=np.float64)
    for label, pos in zip(labels, positions):
        usage[int(pos), int(label)] += 1
    row_sum = usage.sum(axis=1, keepdims=True)
    usage = np.divide(usage, row_sum, out=np.zeros_like(usage), where=row_sum > 0)

    plt.figure(figsize=(max(7, k * 0.55), 4))
    plt.imshow(usage, aspect="auto", cmap="magma")
    plt.colorbar(label="usage share")
    plt.yticks(np.arange(7), [f"pos{p}" for p in range(7)])
    plt.xticks(np.arange(k), [f"B{i}" for i in range(k)], rotation=45, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def run_dataset(name: str, base_dir: Path, output_dir: Path, k_values: list[int], seed: int) -> list[dict]:
    cfg = DATASETS[name]
    data_path = base_dir / cfg["path"]
    columns, values = load_numeric_series(data_path)
    samples, meta = build_relation_rows(values, cfg["cycle"], cfg["top_m"])

    period_count = int(meta["period"].max()) + 1
    train_periods = int(period_count * 0.7)
    train_mask = meta["period"].to_numpy() < train_periods
    train_x = samples[train_mask]
    test_x = samples[~train_mask]
    train_meta = meta[train_mask].reset_index(drop=True)
    test_meta = meta[~train_mask].reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for k in k_values:
        if k >= train_x.shape[0]:
            continue
        model = KMeans(n_clusters=k, random_state=seed, n_init=20)
        train_labels = model.fit_predict(train_x)
        test_labels = model.predict(test_x)
        centers = model.cluster_centers_

        train_err = nearest_error(train_x, centers)
        test_err = nearest_error(test_x, centers)
        rand_err = random_center_error(train_x, k, repeats=30, seed=seed + k)
        sil = float(silhouette_score(train_x, train_labels)) if k > 1 else 0.0
        pos_nmi = float(normalized_mutual_info_score(train_meta["period_mod_7"], train_labels))
        var_nmi = float(normalized_mutual_info_score(train_meta["variable"], train_labels))
        usage = np.bincount(train_labels, minlength=k) / len(train_labels)

        dataset_out = output_dir / name
        dataset_out.mkdir(parents=True, exist_ok=True)
        plot_codewords(
            centers,
            list(columns.columns),
            f"{name} relation codeword candidates (K={k})",
            dataset_out / f"codewords_k{k}.png",
        )
        plot_position_usage(
            train_labels,
            train_meta["period_mod_7"].to_numpy(),
            k,
            f"{name} codeword usage by period position (K={k})",
            dataset_out / f"position_usage_k{k}.png",
        )

        result = {
            "dataset": name,
            "rows": int(values.shape[0]),
            "variables": int(values.shape[1]),
            "cycle": int(cfg["cycle"]),
            "period_count": period_count,
            "top_m": int(cfg["top_m"]),
            "k": int(k),
            "train_relation_rows": int(train_x.shape[0]),
            "test_relation_rows": int(test_x.shape[0]),
            "train_error": train_err,
            "test_error": test_err,
            "test_to_train_error_ratio": float(test_err / train_err) if train_err > 0 else None,
            "random_center_error": rand_err,
            "kmeans_vs_random_improvement": float((rand_err - train_err) / rand_err) if rand_err > 0 else None,
            "silhouette": sil,
            "period_position_nmi": pos_nmi,
            "variable_nmi": var_nmi,
            "min_usage": float(np.min(usage)),
            "max_usage": float(np.max(usage)),
        }
        results.append(result)

        pd.DataFrame(centers, columns=columns.columns).to_csv(dataset_out / f"codewords_k{k}.csv", index=False)
        pd.DataFrame({"label": train_labels, **train_meta.to_dict(orient="list")}).to_csv(
            dataset_out / f"train_assignments_k{k}.csv", index=False
        )
        pd.DataFrame({"label": test_labels, **test_meta.to_dict(orient="list")}).to_csv(
            dataset_out / f"test_assignments_k{k}.csv", index=False
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["ETTh1", "weather"], choices=sorted(DATASETS))
    parser.add_argument("--k", nargs="+", type=int, default=[4, 8, 16])
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    all_results = []
    for dataset in args.datasets:
        all_results.extend(run_dataset(dataset, base_dir, output_dir, args.k, args.seed))

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_results).to_csv(output_dir / "summary.csv", index=False)
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(pd.DataFrame(all_results).to_string(index=False))


if __name__ == "__main__":
    main()
