"""汇总基线指标、并做自适应阈值（滚动分位）对照实验。

输出：
  results/baselines/baseline_metrics.csv   —— 由 per_file_*.csv 重新聚合（修正之前被覆盖的问题）
  results/ablation/adaptive_threshold.csv  —— 静态阈值 vs 滚动分位阈值
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
sys.path.insert(0, str(Path(__file__).resolve().parent))

from skab_protocol import binary_metrics, pool_metrics  # noqa: E402


def rebuild_baseline_table() -> pd.DataFrame:
    rows = []
    for path in sorted((RES / "baselines").glob("per_file_*.csv")):
        model = path.stem.replace("per_file_", "")
        df = pd.read_csv(path)
        pooled = pool_metrics(df.to_dict("records"))
        pooled["macro_f1"] = float(df["f1"].mean())
        pooled["model"] = model
        rows.append(pooled)
    out = pd.DataFrame(rows)[["model", "precision", "recall", "f1", "far", "mar", "macro_f1"]]
    out = out.sort_values("f1", ascending=False)
    out.to_csv(RES / "baselines" / "baseline_metrics.csv", index=False, encoding="utf-8-sig")
    return out


def adaptive_threshold() -> pd.DataFrame:
    res = pd.read_csv(RES / "residuals" / "residuals_w30.csv")
    rows = []
    for width in (100, 200, 400):
        for q in (0.99, 0.995):
            agg = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
            macro = []
            for fname, sub in res.groupby("file"):
                sub = sub.sort_values("window_end")
                test = sub[sub.split == "test"]
                if test.empty:
                    continue
                train = sub[sub.split == "train"].residual_mse.values
                thr_static = float(np.quantile(train, q)) * 4.0 / 3.0
                r = test.residual_mse.values
                series = pd.Series(r)
                thr = (series.rolling(width, min_periods=20).quantile(q).shift(1) * 4.0 / 3.0).to_numpy()
                thr = np.where(np.isnan(thr), thr_static, thr)
                pred = (r > thr).astype(int)
                m = binary_metrics(test.y_true.values, pred)
                for key in agg:
                    agg[key] += int(m[key])
                macro.append(m["f1"])
            tp, fp, fn, tn = agg["tp"], agg["fp"], agg["fn"], agg["tn"]
            pooled = {
                "precision": tp / (tp + fp) if (tp + fp) else 0.0,
                "recall": tp / (tp + fn) if (tp + fn) else 0.0,
                "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0,
                "far": fp / (fp + tn) if (fp + tn) else 0.0,
                "mar": fn / (fn + tp) if (fn + tp) else 0.0,
            }
            rows.append(
                {
                    "mode": "rolling",
                    "width": width,
                    "quantile": q,
                    **{k: pooled[k] for k in ("precision", "recall", "f1", "far", "mar")},
                    "macro_f1": float(np.mean(macro)),
                }
            )
    out = pd.DataFrame(rows).sort_values("f1", ascending=False)
    out.to_csv(RES / "ablation" / "adaptive_threshold.csv", index=False, encoding="utf-8-sig")
    return out


if __name__ == "__main__":
    base = rebuild_baseline_table()
    print("=== 基线（由逐文件结果重新聚合）===")
    print(base.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    ada = adaptive_threshold()
    print("\n=== 自适应阈值 ===")
    print(ada.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
