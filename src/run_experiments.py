"""
SKAB 逐文件协议实验平台。

阶段 1：对每个窗口长度训练一次逐文件 Conv-AE，把训练段/测试段的残差落盘。
        （残差与阈值、后处理规则无关，所以只需训练一次，后面所有消融都复用。）
阶段 2：在缓存的残差上扫描 阈值分位数 × 连续窗口数 × 残差口径，产出指标矩阵。

用法：
    .venv\\Scripts\\python.exe src\\run_experiments.py --stage all
    .venv\\Scripts\\python.exe src\\run_experiments.py --stage sweep   # 只重扫，不重训

产物：
    results/residuals/residuals_w{window}.csv
    results/ablation/sweep.csv
    results/ablation/best_per_file.csv
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from skab_protocol import (  # noqa: E402
    FEATURES,
    binary_metrics,
    consecutive_rule,
    discover_files,
    make_windows,
    pool_metrics,
    read_dataset,
    reconstruction_error,
    split_train_test,
    threshold_from_train,
    train_autoencoder,
    window_end_labels,
    resolve_data_dir,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = resolve_data_dir(ROOT)
RES_DIR = ROOT / "results" / "residuals"
ABL_DIR = ROOT / "results" / "ablation"

WINDOWS = [30, 60]
QUANTILES = [0.99, 0.995, 0.999]
CONSECUTIVE = [1, 5, 20, 60]
RESIDUAL_MODES = ["mse", "mae_sum"]

EPOCHS = {30: 60, 60: 100}
BATCH = 32
LR = 1e-3
SEED = 42

COLS = ["file", "split", "window_end", "y_true", "residual_mse", "residual_mae_sum"]


def stage_residuals(force: bool = False) -> None:
    RES_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files = discover_files(DATA_DIR)

    for window in WINDOWS:
        out_path = RES_DIR / f"residuals_w{window}.csv"
        if out_path.exists() and not force:
            print(f"[skip] {out_path.name} 已存在")
            continue

        print(f"\n=== 训练窗口长度 {window}（{len(files)} 个文件）===")
        t0 = time.time()
        frames = []
        for i, path in enumerate(files, 1):
            rel = path.relative_to(DATA_DIR).as_posix()
            df = read_dataset(path)
            train_df, test_df = split_train_test(df)

            scaler = StandardScaler().fit(train_df[FEATURES].values)
            x_train = scaler.transform(train_df[FEATURES].values)
            x_test = scaler.transform(test_df[FEATURES].values)

            win_train = make_windows(x_train, window)
            win_test = make_windows(x_test, window)

            model = train_autoencoder(
                win_train,
                window,
                epochs=EPOCHS[window],
                batch_size=BATCH,
                lr=LR,
                seed=SEED,
                device=device,
            )

            for split, win, labels_src in (
                ("train", win_train, train_df["anomaly"].values),
                ("test", win_test, test_df["anomaly"].values),
            ):
                err_mse = reconstruction_error(model, win, mode="mse")
                err_mae = reconstruction_error(model, win, mode="mae_sum")
                y = window_end_labels(labels_src.astype(int), window, 1)
                ends = np.arange(len(y)) + window - 1
                frames.append(
                    pd.DataFrame(
                        {
                            "file": rel,
                            "split": split,
                            "window_end": ends,
                            "y_true": y,
                            "residual_mse": err_mse,
                            "residual_mae_sum": err_mae,
                        },
                        columns=COLS,
                    )
                )
            print(f"  [{i:2d}/{len(files)}] {rel}")

        pd.concat(frames, ignore_index=True).to_csv(
            out_path, index=False, encoding="utf-8-sig"
        )
        print(f"→ {out_path}（{time.time() - t0:.1f}s）")


def stage_sweep() -> None:
    ABL_DIR.mkdir(parents=True, exist_ok=True)
    rows, best_per_file = [], []

    for window in WINDOWS:
        res_path = RES_DIR / f"residuals_w{window}.csv"
        if not res_path.exists():
            print(f"[warn] 缺少 {res_path.name}，跳过 window={window}")
            continue
        df = pd.read_csv(res_path)

        for mode, q, k in itertools.product(RESIDUAL_MODES, QUANTILES, CONSECUTIVE):
            col = f"residual_{mode}"
            per_file = []
            for rel, g in df.groupby("file", sort=True):
                train = g[g["split"] == "train"]
                test = g[g["split"] == "test"]
                thr = threshold_from_train(train[col].values, q)
                pred = consecutive_rule((test[col].values > thr).astype(int), k)
                m = binary_metrics(test["y_true"].values, pred)
                m["file"] = rel
                m["threshold"] = thr
                per_file.append(m)

            agg = pool_metrics(per_file)
            row = dict(
                window=window, residual=mode, quantile=q, consecutive=k,
                **{kk: agg[kk] for kk in ("precision", "recall", "f1", "far", "mar")},
                macro_f1=float(np.mean([d["f1"] for d in per_file])),
                macro_far=float(np.mean([d["far"] for d in per_file])),
            )
            rows.append(row)

            if (window, mode, q, k) == (30, "mse", 0.99, 1) or not best_per_file:
                best_per_file = per_file

    sweep = pd.DataFrame(rows)
    sweep.to_csv(ABL_DIR / "sweep.csv", index=False, encoding="utf-8-sig")

    print("\n=== 扫描结果（按 macro-F1 排序，前 12）===")
    print(
        sweep.sort_values("macro_f1", ascending=False)
        .head(12)
        .to_string(index=False, float_format=lambda v: f"{v:.4f}")
    )

    # 同时保存 v2 默认配置（w30/mse/q0.99/k1）的逐文件明细
    pd.DataFrame(best_per_file)[
        ["file", "precision", "recall", "f1", "far", "mar", "threshold", "tp", "fp", "fn", "tn"]
    ].to_csv(ABL_DIR / "best_per_file.csv", index=False, encoding="utf-8-sig")

    summary = {
        "n_configs": len(sweep),
        "windows": WINDOWS,
        "quantiles": QUANTILES,
        "consecutive": CONSECUTIVE,
        "residual_modes": RESIDUAL_MODES,
        "best_macro_f1": sweep.loc[sweep["macro_f1"].idxmax()].to_dict(),
        "best_f1": sweep.loc[sweep["f1"].idxmax()].to_dict(),
    }
    (ABL_DIR / "sweep_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n→ {ABL_DIR / 'sweep.csv'}")


# --------------------------------------------------------------------------
# 阶段 3：留出验证段定阈值（治「训练段残差定阈值导致阈值过紧」）
# --------------------------------------------------------------------------
FIT_END = 320    # 前 320 行训练
VAL_END = 400    # 320~400 行做阈值标定，模型从未见过
HOLDOUT_EPOCHS = 60


def stage_holdout() -> None:
    ABL_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files = discover_files(DATA_DIR)
    window = 30

    per_config: dict[tuple, list] = {}
    rows = []
    t0 = time.time()

    for i, path in enumerate(files, 1):
        rel = path.relative_to(DATA_DIR).as_posix()
        df = read_dataset(path)
        fit_df = df.iloc[:FIT_END]
        val_df = df.iloc[FIT_END:VAL_END]
        test_df = df.iloc[VAL_END:]

        scaler = StandardScaler().fit(fit_df[FEATURES].values)
        x_fit = scaler.transform(fit_df[FEATURES].values)
        x_val = scaler.transform(val_df[FEATURES].values)
        x_test = scaler.transform(test_df[FEATURES].values)

        win_fit = make_windows(x_fit, window)
        win_val = make_windows(x_val, window)
        win_test = make_windows(x_test, window)
        y_val = window_end_labels(val_df["anomaly"].values.astype(int), window, 1)
        y_test = window_end_labels(test_df["anomaly"].values.astype(int), window, 1)

        model = train_autoencoder(
            win_fit, window, epochs=HOLDOUT_EPOCHS, batch_size=BATCH, lr=LR,
            seed=SEED, device=device, val_split=0.0, patience=5,
        )
        err_fit = reconstruction_error(model, win_fit, mode="mse")
        err_val = reconstruction_error(model, win_val, mode="mse")
        err_test = reconstruction_error(model, win_test, mode="mse")
        rows.extend(
            dict(file=rel, split=sp, y_true=list(y), residual_mse=list(e))
            for sp, y, e in (("fit", np.zeros(len(err_fit)), err_fit),
                             ("val", y_val, err_val),
                             ("test", y_test, err_test))
        )

        for q in (0.90, 0.95, 0.99):
            for k in (1, 5):
                thr = float(np.quantile(err_val, q) * 4 / 3)
                pred = consecutive_rule((err_test > thr).astype(int), k)
                m = binary_metrics(y_test, pred)
                m["file"] = rel
                per_config.setdefault((q, k), []).append(m)

        print(f"  [{i:2d}/{len(files)}] {rel}")

    pd.DataFrame(rows).to_csv(
        RES_DIR / "residuals_holdout.csv", index=False, encoding="utf-8-sig"
    )

    out = []
    for (q, k), per in per_config.items():
        agg = pool_metrics(per)
        out.append(dict(window=window, threshold_from="val", quantile=q, consecutive=k,
                        **{kk: agg[kk] for kk in ("precision", "recall", "f1", "far", "mar")},
                        macro_f1=float(np.mean([d["f1"] for d in per]))))
    t = pd.DataFrame(out).sort_values("macro_f1", ascending=False)
    t.to_csv(ABL_DIR / "holdout_threshold.csv", index=False, encoding="utf-8-sig")
    print("=== 留出验证段定阈值 ===")
    print(t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"用时 {time.time() - t0:.1f}s")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["all", "residuals", "sweep", "holdout"], default="all")
    p.add_argument("--force", action="store_true", help="忽略已有残差缓存")
    args = p.parse_args()

    if args.stage in ("all", "residuals"):
        stage_residuals(force=args.force)
    if args.stage in ("all", "sweep"):
        stage_sweep()
    if args.stage in ("all", "holdout"):
        stage_holdout()


if __name__ == "__main__":
    main()
