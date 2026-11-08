"""为报告收集可复核的派生统计量（全部由 results/ 与原始数据现算）。

产物（results/controlled/）：
  pca_evr.json            8 个主成分在各文件上的累计解释方差比例
  fp_segments.json        现行默认配置的误报窗口/连续段结构
  other2_exclusion.json   去掉 other/2.csv 后的点级指标
  fold_files.json         文件级二折的具体划分
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from skab_paths import resolve_data_dir  # noqa: E402
from skab_protocol import (  # noqa: E402
    FEATURES,
    THRESHOLD_SCALE,
    discover_files,
    make_windows,
    read_dataset,
    split_train_test,
    window_end_labels,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "controlled"
DATA_DIR = resolve_data_dir(ROOT)
WINDOW = 30


def default_predictions() -> pd.DataFrame:
    """现行默认（w30 / mse / q0.99 × 4/3 / k1）在残差缓存上的逐窗口判定。"""
    df = pd.read_csv(ROOT / "results" / "residuals" / "residuals_w30.csv")
    frames = []
    for f, g in df.groupby("file"):
        tr, te = g[g.split == "train"], g[g.split == "test"].sort_values("window_end")
        thr = float(np.quantile(tr.residual_mse.values, 0.99) * THRESHOLD_SCALE)
        frames.append(pd.DataFrame({"file": f, "y_true": te.y_true.values,
                                    "pred": (te.residual_mse.values > thr).astype(int)}))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # 1) PCA 累计解释方差（与原协议一致：逐文件标准化后滑窗，逐文件 fit）
    cum = []
    for path in discover_files(DATA_DIR):
        df = read_dataset(path)
        tr, _ = split_train_test(df, 400)
        sc = StandardScaler().fit(tr[FEATURES].values)
        w = make_windows(sc.transform(tr[FEATURES].values).astype(np.float32), WINDOW, 1)
        pca = PCA(n_components=8, random_state=42).fit(w.reshape(len(w), -1))
        cum.append(float(pca.explained_variance_ratio_.sum()))
    (OUT / "pca_evr.json").write_text(json.dumps(
        {"n_components": 8, "n_features": WINDOW * len(FEATURES),
         "mean": round(float(np.mean(cum)), 4), "min": round(float(np.min(cum)), 4),
         "max": round(float(np.max(cum)), 4)}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("PCA EVR:", round(float(np.mean(cum)), 4))

    # 2) 现行默认配置的误报段结构
    p = default_predictions()
    tp = int(((p.pred == 1) & (p.y_true == 1)).sum())
    fp = int(((p.pred == 1) & (p.y_true == 0)).sum())
    fn = int(((p.pred == 0) & (p.y_true == 1)).sum())
    tn = int(((p.pred == 0) & (p.y_true == 0)).sum())
    segs = []
    for _, g in p.groupby("file"):
        run = 0
        for pred, y in zip(g.pred.values, g.y_true.values):
            if pred == 1 and y == 0:
                run += 1
            else:
                if run:
                    segs.append(run)
                run = 0
        if run:
            segs.append(run)
    segs = np.array(segs)
    top10 = float(np.sort(segs)[::-1][:10].sum() / segs.sum())
    (OUT / "fp_segments.json").write_text(json.dumps(
        {"fp": fp, "tn": tn, "far": round(fp / (fp + tn), 4), "normal_windows": int(fp + tn),
         "segments": int(len(segs)), "mean_len": round(float(segs.mean()), 1),
         "median_len": round(float(np.median(segs)), 1), "max_len": int(segs.max()),
         "top10_share": round(top10, 3), "windows": int(len(p))}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print("FP", fp, "segments", len(segs))

    # 3) 去掉 other/2.csv 后的点级指标
    q = p[p.file != "other/2.csv"]
    tp2 = int(((q.pred == 1) & (q.y_true == 1)).sum()); fp2 = int(((q.pred == 1) & (q.y_true == 0)).sum())
    fn2 = int(((q.pred == 0) & (q.y_true == 1)).sum()); tn2 = int(((q.pred == 0) & (q.y_true == 0)).sum())
    (OUT / "other2_exclusion.json").write_text(json.dumps(
        {"n_files": 33, "precision": round(tp2 / (tp2 + fp2), 4), "recall": round(tp2 / (tp2 + fn2), 4),
         "f1": round(tp2 / (tp2 + 0.5 * (fp2 + fn2)), 4), "far": round(fp2 / (fp2 + tn2), 4),
         "mar": round(fn2 / (fn2 + tp2), 4)}, indent=2, ensure_ascii=False), encoding="utf-8")
    print("excl other/2 F1", round(tp2 / (tp2 + 0.5 * (fp2 + fn2)), 4))

    # 4) 二折划分
    files = sorted(set(pd.read_csv(ROOT / "results" / "residuals" / "residuals_w30.csv").file))
    (OUT / "fold_files.json").write_text(json.dumps(
        {"fold1_eval": files[0::2], "fold2_eval": files[1::2]}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print("folds:", len(files[0::2]), len(files[1::2]))


if __name__ == "__main__":
    main()
