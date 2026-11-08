"""多种子重复实验：固定划分与协议，只改变初始化与批次顺序。

配置（与主实验一致）：CNN-AE、window 30、stride 1、batch 32、60 轮、patience 5、
逐文件 StandardScaler、前 400 点训练、q99 x 4/3、k=1、残差 MSE 均值。
验证集划分固定为 split_seed=42 的排列，三个种子共用同一划分。

输出：results/repeats/seed_{42,43,44}/{config.json,training_history.csv,
test_predictions.csv,evaluation_metrics.csv,per_file_metrics.csv,checkpoints/*.pth}
以及 results/repeats/{repeat_summary.csv,per_file_seed_matrix.csv}
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from skab_protocol import (  # noqa: E402
    FEATURES,
    THRESHOLD_SCALE,
    binary_metrics,
    consecutive_rule,
    discover_files,
    make_windows,
    read_dataset,
    reconstruction_error,
    resolve_data_dir,
    split_train_test,
    threshold_from_train,
    train_autoencoder,
    window_end_labels,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = resolve_data_dir(ROOT)

VALID_SCHEMES = ("random", "holdout", "random_matched")
VAL_SCHEME = "random"
OUT = ROOT / "results" / "repeats"

WINDOW, STRIDE = 30, 1
BATCH, EPOCHS, PATIENCE, LR = 32, 60, 5, 1e-3
QUANTILE, SCALE, K = 0.99, THRESHOLD_SCALE, 1
SPLIT_SEED = 42
SEEDS = (42, 43, 44)


def run_seed(seed: int) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = OUT / f"seed_{seed}"
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    files = discover_files(DATA_DIR)
    per_file, preds, history = [], [], []
    t0 = time.time()

    for i, path in enumerate(files, 1):
        rel = path.relative_to(DATA_DIR).as_posix()
        df = read_dataset(path)
        train_df, test_df = split_train_test(df)

        scaler = StandardScaler().fit(train_df[FEATURES].values)
        w_tr = make_windows(scaler.transform(train_df[FEATURES].values), WINDOW, STRIDE)
        w_te = make_windows(scaler.transform(test_df[FEATURES].values), WINDOW, STRIDE)
        y_te = window_end_labels(test_df["anomaly"].values.astype(int), WINDOW, STRIDE)
        y_tr = window_end_labels(train_df["anomaly"].values.astype(int), WINDOW, STRIDE)

        hist: list[dict] = []
        model = train_autoencoder(
            w_tr, WINDOW, epochs=EPOCHS, batch_size=BATCH, lr=LR, seed=seed,
            device=device, val_split=0.1, patience=PATIENCE,
            split_seed=SPLIT_SEED, history=hist, val_scheme=VAL_SCHEME,
        )
        for h in hist:
            history.append({"seed": seed, "file": rel, **h})

        ckpt = out_dir / "checkpoints" / (rel.replace("/", "__").replace(".csv", "") + ".pth")
        torch.save(model.state_dict(), ckpt)

        e_tr = reconstruction_error(model, w_tr, mode="mse")
        e_te = reconstruction_error(model, w_te, mode="mse")
        thr = threshold_from_train(e_tr, QUANTILE)
        pred = consecutive_rule((e_te > thr).astype(int), K)
        m = binary_metrics(y_te, pred)
        m.update(file=rel, threshold=thr, val_loss=float(hist[-1]["val_loss"]) if hist else np.nan,
                 n_train=int(len(y_tr)), n_test=int(len(y_te)))
        per_file.append(m)

        preds.append(pd.DataFrame({
            "file": rel, "window_end": np.arange(len(y_te)) + WINDOW - 1,
            "y_true": y_te, "residual": e_te, "predicted": pred, "threshold": thr,
        }))
        print(f"[seed {seed}] [{i:2d}/{len(files)}] {rel}  F1={m['f1']:.4f} thr={thr:.4f}")

    pf = pd.DataFrame(per_file)
    pd.concat(preds, ignore_index=True).to_csv(out_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False, encoding="utf-8-sig")
    pf.drop(columns=["tp", "fp", "fn", "tn"]).to_csv(
        out_dir / "evaluation_metrics.csv", index=False, encoding="utf-8-sig")
    pf.to_csv(out_dir / "per_file_metrics.csv", index=False, encoding="utf-8-sig")

    tp, fp, fn, tn = (int(pf[c].sum()) for c in ("tp", "fp", "fn", "tn"))
    pooled = dict(
        precision=tp / (tp + fp), recall=tp / (tp + fn),
        f1=2 * tp / (2 * tp + fp + fn), far=fp / (fp + tn), mar=fn / (fn + tp),
        tp=tp, fp=fp, fn=fn, tn=tn,
    )
    macro = dict(macro_f1=float(pf.f1.mean()), macro_far=float(pf.far.mean()),
                 macro_recall=float(pf.recall.mean()))

    cfg = dict(
        seed=seed, split_seed=SPLIT_SEED, model="CNNAutoencoder",
        window=WINDOW, stride=STRIDE, batch_size=BATCH, epochs=EPOCHS,
        patience=PATIENCE, lr=LR, val_split=0.1, scaler="per-file StandardScaler",
        val_scheme=VAL_SCHEME,
        residual="mse", quantile=QUANTILE, threshold_scale=SCALE, consecutive=K,
        train_size=400, n_files=len(files), pooled=pooled, macro=macro,
        elapsed_sec=round(time.time() - t0, 1),
    )
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[seed {seed}] pooled F1={pooled['f1']:.4f} FAR={pooled['far']:.4f} "
          f"MAR={pooled['mar']:.4f} macro-F1={macro['macro_f1']:.4f}  {cfg['elapsed_sec']}s")
    return cfg


def summarize(cfgs: list[dict]) -> None:
    rows = []
    for c in cfgs:
        rows.append(dict(seed=c["seed"],
                         **{k: c["pooled"][k] for k in ("precision", "recall", "f1", "far", "mar")},
                         macro_f1=c["macro"]["macro_f1"], macro_far=c["macro"]["macro_far"]))
    df = pd.DataFrame(rows)
    stat = []
    for col in ("precision", "recall", "f1", "far", "mar", "macro_f1", "macro_far"):
        v = df[col]
        stat.append(dict(metric=col, mean=v.mean(), std=v.std(ddof=1),
                         min=v.min(), max=v.max(), n=len(v)))
    st = pd.DataFrame(stat)
    st.to_csv(OUT / "repeat_summary.csv", index=False, encoding="utf-8-sig")
    df.to_csv(OUT / "per_seed_metrics.csv", index=False, encoding="utf-8-sig")

    frames = []
    for c in cfgs:
        pf = pd.read_csv(OUT / f"seed_{c['seed']}" / "per_file_metrics.csv")
        frames.append(pf.set_index("file")[["f1", "far", "mar"]].add_suffix(f"_{c['seed']}"))
    mat = pd.concat(frames, axis=1)
    f1cols = [f"f1_{c['seed']}" for c in cfgs]
    mat["f1_mean"] = mat[f1cols].mean(axis=1)
    mat["f1_std"] = mat[f1cols].std(axis=1, ddof=1)
    mat.sort_values("f1_std", ascending=False).to_csv(
        OUT / "per_file_seed_matrix.csv", encoding="utf-8-sig")

    print("\n=== pooled 指标 ===")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n=== 均值 ± 标准差（n=3） ===")
    print(st.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n=== 对种子最敏感的 5 个文件（按 F1 标准差） ===")
    print(mat.sort_values("f1_std", ascending=False).head(5)[f1cols + ["f1_std"]]
          .to_string(float_format=lambda v: f"{v:.4f}"))


def main() -> None:
    global VAL_SCHEME, OUT
    import argparse

    ap = argparse.ArgumentParser(description="多种子重复实验（可切换验证集划分方案）")
    ap.add_argument("--val-scheme", choices=VALID_SCHEMES, default="random")
    ap.add_argument("--out", default=None, help="输出目录，默认 results/repeats")
    args = ap.parse_args()
    VAL_SCHEME = args.val_scheme
    OUT = Path(args.out) if args.out else ROOT / "results" / (
        "repeats" if VAL_SCHEME == "random" else "repeats_holdout")

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"val_scheme={VAL_SCHEME}  out={OUT}")
    cfgs = []
    for seed in SEEDS:
        print(f"seed {seed}")
        cfgs.append(run_seed(seed))
    summarize(cfgs)

if __name__ == "__main__":
    main()
