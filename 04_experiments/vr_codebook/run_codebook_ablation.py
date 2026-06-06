import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import torch


METRIC_PATTERN = re.compile(
    r"mse:(?P<mse>[0-9.]+),\s*mae:(?P<mae>[0-9.]+),\s*rse:(?P<rse>[0-9.]+)"
)


def experiment_configs():
    return [
        {"name": "baseline", "use_codebook": False, "k": None, "beta_init": None},
        {"name": "k4_b005", "use_codebook": True, "k": 4, "beta_init": 0.05},
        {"name": "k8_b005", "use_codebook": True, "k": 8, "beta_init": 0.05},
        {"name": "k16_b005", "use_codebook": True, "k": 16, "beta_init": 0.05},
        {"name": "k8_b001", "use_codebook": True, "k": 8, "beta_init": 0.01},
        {"name": "k8_b010", "use_codebook": True, "k": 8, "beta_init": 0.10},
    ]


def checkpoint_dir(code_dir: Path, name: str, pred_len: int) -> Path:
    setting = (
        f"ETTh1_720_{pred_len}_RIMF_ETTh1_ftM_sl720_pl{pred_len}_"
        f"linear_vr_ablation_{name}_0_seed2025"
    )
    return code_dir / "checkpoints" / setting


def learned_beta(checkpoint: Path):
    if not checkpoint.exists():
        return None
    state = torch.load(checkpoint, map_location="cpu")
    for key, value in state.items():
        if key.endswith("relation_codebook.beta_logit"):
            return float(torch.sigmoid(value))
    return None


def build_command(args, config):
    command = [
        sys.executable,
        "-u",
        str(args.code_dir / "run_longExp.py"),
        "--is_training",
        "0" if args.test_only else "1",
        "--root_path",
        str(args.data_dir),
        "--data_path",
        "ETTh1.csv",
        "--model_id",
        f"ETTh1_720_{args.pred_len}",
        "--model",
        "RIMF",
        "--data",
        "ETTh1",
        "--features",
        "M",
        "--seq_len",
        "720",
        "--pred_len",
        str(args.pred_len),
        "--period_len",
        "24",
        "--enc_in",
        "7",
        "--dec_in",
        "7",
        "--c_out",
        "7",
        "--model_type",
        "linear",
        "--train_epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--itr",
        "1",
        "--batch_size",
        str(args.batch_size),
        "--learning_rate",
        str(args.learning_rate),
        "--num_workers",
        "0",
        "--des",
        f"vr_ablation_{config['name']}",
    ]
    if config["use_codebook"]:
        command.extend(
            [
                "--use_relation_codebook",
                "--relation_codebook_size",
                str(config["k"]),
                "--relation_codebook_beta_init",
                str(config["beta_init"]),
                "--relation_codebook_temperature",
                str(args.temperature),
            ]
        )
    return command


def run_experiment(args, config):
    command = build_command(args, config)
    print(f"\n===== {config['name']} =====", flush=True)
    process = subprocess.run(
        command,
        cwd=args.code_dir,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    print(process.stdout, flush=True)
    if process.returncode != 0:
        print(process.stderr, file=sys.stderr, flush=True)
        raise RuntimeError(f"Experiment {config['name']} failed with code {process.returncode}")

    matches = list(METRIC_PATTERN.finditer(process.stdout))
    if not matches:
        raise RuntimeError(f"Could not parse metrics for {config['name']}")
    metrics = matches[-1].groupdict()

    ckpt = checkpoint_dir(args.code_dir, config["name"], args.pred_len) / "checkpoint.pth"
    return {
        **config,
        "pred_len": args.pred_len,
        "epochs": args.epochs,
        "mse": float(metrics["mse"]),
        "mae": float(metrics["mae"]),
        "rse": float(metrics["rse"]),
        "learned_beta": learned_beta(ckpt),
        "checkpoint": str(ckpt),
    }


def save_results(results, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "summary.json"
    csv_path = output_dir / "summary.csv"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\n===== summary =====")
    for result in sorted(results, key=lambda item: item["mse"]):
        print(
            f"{result['name']:12s} mse={result['mse']:.6f} "
            f"mae={result['mae']:.6f} beta={result['learned_beta']}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--code-dir",
        type=Path,
        default=Path(r"F:\paper\timeSeries\mscf\03_code"),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(r"F:\paper\timeSeries\mscf\03_code\data\ETT"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"F:\paper\timeSeries\mscf\04_experiments\vr_codebook\outputs_ablation"),
    )
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--test-only", action="store_true", default=False)
    args = parser.parse_args()

    args.code_dir = args.code_dir.resolve()
    args.data_dir = args.data_dir.resolve()
    args.output_dir = args.output_dir.resolve()

    results = []
    for config in experiment_configs():
        results.append(run_experiment(args, config))
        save_results(results, args.output_dir)


if __name__ == "__main__":
    main()
