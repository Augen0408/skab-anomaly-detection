"""
按 SKAB 逐文件协议跑 Conv-AE 基线（protocol v2）。

用法（项目根目录下）：
    .venv\\Scripts\\python.exe src\\run_protocol_v2.py --out results/protocol_v2

产物：
    <out>/per_file_metrics.csv   逐文件指标
    <out>/predictions.csv        逐窗口残差与预测（含文件名、窗口末索引）
    <out>/summary.json           汇总指标与运行配置
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from skab_paths import resolve_data_dir  # noqa: E402
from skab_protocol import (  # noqa: E402
    FEATURES,
    LABEL_COLUMNS,
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
)

ROOT = Path(__file__).resolve().parents[1]


def resolve_data() -> Path:
    """定位 SKAB CSV 根目录（兼容不同解压层级），缺失时给出下载与配置提示。"""
    try:
        return resolve_data_dir(ROOT)
    except FileNotFoundError:
        raise SystemExit(
            "未找到 SKAB 数据目录。请从 Kaggle 下载 SKAB v0.9 "
            "（https://www.kaggle.com/datasets/katser/skoltech-anomaly-benchmark-skab），"
            "解压到 data/raw/SKAB-master/data/{valve1,valve2,other}/ 与 anomaly-free.csv；"
            "其他解压层级放在 data/raw/ 下也可自动识别（见 README「数据」一节）。"
        ) from None


PRESETS = {
    # 本项目 v2：窗口 30、q=0.99、MSE、单窗口即报警
    "v2": dict(window=30, epochs=30, batch_size=64, quantile=0.99,
               consecutive=1, residual="mse"),
    # 官方 Conv_AE.ipynb：窗口 60、q=0.999、MAE-sum、连续 60 窗口
    "official": dict(window=60, epochs=100, batch_size=32, quantile=0.999,
                     consecutive=60, residual="mae_sum"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="results/protocol_v2")
    p.add_argument("--preset", choices=["v2", "official"], default="v2")
    p.add_argument("--window", type=int, default=None)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--quantile", type=float, default=None)
    p.add_argument("--consecutive", type=int, default=None)
    p.add_argument("--residual", choices=["mse", "mae_sum"], default=None)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-size", type=int, default=400)
    p.add_argument("--only", type=str, default=None, help="只跑匹配的文件名子串，调试用")
    args = p.parse_args()
    for k, v in PRESETS[args.preset].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    return args


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备：{device}" + (f"（{torch.cuda.get_device_name(0)}）" if device.type == "cuda" else ""))
    print(f"协议[{args.preset}]：逐文件切分 @ {args.train_size} 行 | 窗口 {args.window} | "
          f"残差 {args.residual} | 阈值 q={args.quantile}×4/3 | 连续 {args.consecutive} 窗口")

    data_dir = resolve_data()
    files = discover_files(data_dir)
    if args.only:
        files = [f for f in files if args.only in f.as_posix()]
    print(f"文件数：{len(files)}\n")

    rows, prediction_frames, per_file = [], [], []
    t0 = time.time()

    for i, path in enumerate(files, 1):
        rel = path.relative_to(data_dir).as_posix()
        df = read_dataset(path)
        train_df, test_df = split_train_test(df, args.train_size)

        # 记录训练段里是否混入异常（other/2.csv 已知存在）
        train_anomaly = int(train_df["anomaly"].sum())

        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler().fit(train_df[FEATURES].values)
        x_train = scaler.transform(train_df[FEATURES].values)
        x_test = scaler.transform(test_df[FEATURES].values)

        win_train = make_windows(x_train, args.window, args.stride)
        win_test = make_windows(x_test, args.window, args.stride)
        y_test = window_end_labels(
            test_df["anomaly"].values.astype(int), args.window, args.stride
        )
        ends = np.arange(len(y_test)) * args.stride + args.window - 1

        model = train_autoencoder(
            win_train,
            args.window,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            device=device,
            val_split=args.val_split,
            patience=args.patience,
        )
        err_train = reconstruction_error(model, win_train, mode=args.residual)
        err_test = reconstruction_error(model, win_test, mode=args.residual)
        thr = threshold_from_train(err_train, args.quantile)

        pred = consecutive_rule((err_test > thr).astype(int), args.consecutive)
        m = binary_metrics(y_test, pred)
        per_file.append(m)

        rows.append(
            dict(
                file=rel,
                n_train=len(win_train),
                n_test=len(win_test),
                anomaly_ratio=float(y_test.mean()),
                train_anomaly=train_anomaly,
                threshold=thr,
                **m,
            )
        )
        prediction_frames.append(
            pd.DataFrame(
                dict(
                    file=rel,
                    window_end=ends,
                    y_true=y_test,
                    residual=err_test,
                    predicted=pred,
                )
            )
        )

        print(
            f"[{i:2d}/{len(files)}] {rel:18s} thr={thr:8.4f} "
            f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
            f"FAR={m['far']:.3f}"
            + (f"  ⚠训练段含 {train_anomaly} 个异常点" if train_anomaly else "")
        )

    per_file_df = pd.DataFrame(rows).sort_values("file").reset_index(drop=True)
    per_file_df.to_csv(out_dir / "per_file_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        out_dir / "predictions.csv", index=False, encoding="utf-8-sig"
    )

    summary = pool_metrics(per_file)
    summary["macro_f1"] = float(per_file_df["f1"].mean())
    summary["macro_far"] = float(per_file_df["far"].mean())
    summary["macro_recall"] = float(per_file_df["recall"].mean())
    summary["n_files"] = len(per_file_df)
    summary["config"] = vars(args)
    summary["elapsed_sec"] = round(time.time() - t0, 1)

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 72)
    print(f"汇总（{len(per_file_df)} 个文件，窗口加权）")
    print(f"  Precision {summary['precision']:.4f}  Recall {summary['recall']:.4f}  "
          f"F1 {summary['f1']:.4f}")
    print(f"  FAR {summary['far']:.4f}  MAR {summary['mar']:.4f}  "
          f"（macro-F1 {summary['macro_f1']:.4f}）")
    print(f"  混淆矩阵 TN={summary['tn']} FP={summary['fp']} FN={summary['fn']} TP={summary['tp']}")
    print(f"  用时 {summary['elapsed_sec']}s → {out_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
