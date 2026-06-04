import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def build_args(cli_args):
    return SimpleNamespace(
        is_training=0,
        model_id=f"{cli_args.data_name}_{cli_args.seq_len}_{cli_args.pred_len}",
        model="RIMF",
        data=cli_args.data_name,
        root_path=cli_args.root_path,
        data_path=cli_args.data_path,
        features="M",
        target="OT",
        freq=cli_args.freq,
        checkpoints="./checkpoints/",
        seq_len=cli_args.seq_len,
        label_len=cli_args.label_len,
        pred_len=cli_args.pred_len,
        fc_dropout=0.05,
        head_dropout=0.0,
        patch_len=16,
        stride=8,
        padding_patch="end",
        revin=1,
        affine=0,
        subtract_last=0,
        decomposition=0,
        kernel_size=25,
        individual=0,
        embed_type=0,
        enc_in=cli_args.enc_in,
        dec_in=cli_args.enc_in,
        c_out=cli_args.enc_in,
        d_model=cli_args.d_model,
        n_heads=8,
        e_layers=2,
        d_layers=cli_args.d_layers,
        d_ff=2048,
        moving_avg=25,
        factor=1,
        distil=True,
        dropout=cli_args.dropout,
        embed="learned",
        activation="gelu",
        output_attention=False,
        do_predict=False,
        num_workers=cli_args.num_workers,
        itr=1,
        train_epochs=1,
        batch_size=cli_args.batch_size,
        patience=1,
        learning_rate=0.02,
        des="relation_export",
        loss="mse",
        lradj="type3",
        pct_start=0.3,
        use_amp=False,
        use_gpu=torch.cuda.is_available(),
        gpu=0,
        use_multi_gpu=0,
        devices="0",
        test_flop=False,
        period_len=cli_args.period_len,
        model_type=cli_args.model_type,
        corr_list=None,
        period_list_str="[24]",
        period_list=[cli_args.period_len],
        num_samples=cli_args.num_samples,
        samples_ratio=0.4,
        n_block=2,
        patch=24,
        mdm_k=3,
        mdm_c=2,
        ff_dim=512,
        num_experts=4,
        top_k=2,
        layernorm=1,
        norm=1,
        ms_num_scales=4,
        ms_pool_stride=2,
        ms_fusion="learnable",
        exp_name="MTSF",
        channel_independence=False,
        inverse=False,
        class_strategy="projection",
        target_root_path="./data/electricity/",
        target_data_path="electricity.csv",
        efficient_training=False,
        use_norm=True,
        partial_start_index=0,
    )


def to_numpy(tensor):
    return tensor.detach().float().cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-dir", default=r"F:\paper\timeSeries\mscf\03_code")
    parser.add_argument("--root-path", default=r"F:\paper\timeSeries\mscf\03_code\data\ETT")
    parser.add_argument("--data-path", default="ETTh1.csv")
    parser.add_argument("--data-name", default="ETTh1")
    parser.add_argument("--freq", default="h")
    parser.add_argument("--output-dir", default="outputs_rimf")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--max-batches", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=720)
    parser.add_argument("--label-len", type=int, default=48)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--period-len", type=int, default=24)
    parser.add_argument("--enc-in", type=int, default=7)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--model-type", default="linear", choices=["linear", "mlp"])
    parser.add_argument("--num-samples", type=int, default=7)
    args = parser.parse_args()

    code_dir = Path(args.code_dir).resolve()
    sys.path.insert(0, str(code_dir))

    from data_provider.data_factory import data_provider
    from models import RIMF
    from utils.tools import masked_softmax

    cfg = build_args(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = RIMF.Model(cfg).to(device)
    model.eval()

    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state)

    captures = {"history": [], "future_raw_layers": {}}

    def history_hook(_module, _inputs, output):
        captures["history"].append(to_numpy(output))

    def future_hook(layer_idx):
        def _hook(_module, _inputs, output):
            captures["future_raw_layers"].setdefault(layer_idx, []).append(to_numpy(output))

        return _hook

    handles = [model.Inter_Variable.register_forward_hook(history_hook)]
    for idx, layer in enumerate(model.sca.sca_layers):
        handles.append(layer.Relationship_learning.register_forward_hook(future_hook(idx)))

    _, loader = data_provider(cfg, args.split)
    with torch.no_grad():
        for batch_idx, (batch_x, _batch_y, _batch_x_mark, _batch_y_mark) in enumerate(loader):
            if batch_idx >= args.max_batches:
                break
            batch_x = batch_x.float().to(device)
            _ = model(batch_x)

    for handle in handles:
        handle.remove()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    arrays = {}
    if captures["history"]:
        arrays["history"] = np.concatenate(captures["history"], axis=0)

    for idx, chunks in captures["future_raw_layers"].items():
        raw = np.concatenate(chunks, axis=0)
        arrays[f"future_raw_layer{idx}"] = raw
        arrays[f"future_norm_layer{idx}"] = to_numpy(masked_softmax(torch.from_numpy(raw).to(device)))

    npz_path = output_dir / f"{args.data_name}_rimf_relations_{args.split}.npz"
    np.savez_compressed(npz_path, **arrays)

    meta = {
        "data_name": args.data_name,
        "data_path": str(Path(args.root_path) / args.data_path),
        "split": args.split,
        "max_batches": args.max_batches,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "period_len": args.period_len,
        "enc_in": args.enc_in,
        "checkpoint": args.checkpoint,
        "arrays": {key: list(value.shape) for key, value in arrays.items()},
        "note": "Future arrays are meaningful for forecasting only when a trained checkpoint is loaded.",
    }
    meta_path = output_dir / f"{args.data_name}_rimf_relations_{args.split}.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
