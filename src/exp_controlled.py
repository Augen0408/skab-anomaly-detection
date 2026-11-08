"""审稿补充实验：受控对比、无偏配置选择、敏感性分析与官方单通道复刻。

stage：
  c1            阈值规则受控（同一模型，只换阈值分位/系数）
  selection     文件级二折：一半文件选配置，另一半评估（消除测试集选优）
  other2        other/2.csv 训练段污染敏感性
  global        全局标准化 + 逐文件训练（隔离标准化贡献）
  official1ch   复刻官方 filters=1 单通道输出，验证 FAR/MAR 机制

所有结果写入 results/controlled/。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from skab_paths import resolve_data_dir
from skab_protocol import (
    FEATURES,
    THRESHOLD_SCALE,
    binary_metrics,
    consecutive_rule,
    discover_files,
    make_windows,
    read_dataset,
    reconstruction_error,
    split_train_test,
    to_channels_first,
    train_autoencoder,
    window_end_labels,
)

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = RES / "controlled"
DATA_DIR = resolve_data_dir(ROOT)
WINDOW = 30
SEED = 42


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pooled(records: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    """把各文件混淆矩阵相加后算点级指标（与官方 chp_score 汇总口径一致）。"""
    tp = fp = fn = tn = 0
    for y, p in records:
        tp += int(((y == 1) & (p == 1)).sum())
        fp += int(((y == 0) & (p == 1)).sum())
        fn += int(((y == 1) & (p == 0)).sum())
        tn += int(((y == 0) & (p == 0)).sum())
    return dict(precision=tp / (tp + fp) if tp + fp else 0.0,
                recall=tp / (tp + fn) if tp + fn else 0.0,
                f1=tp / (tp + 0.5 * (fp + fn)) if tp else 0.0,
                far=fp / (fp + tn) if fp + tn else 0.0,
                mar=fn / (fn + tp) if fn + tp else 0.0,
                tp=tp, fp=fp, fn=fn, tn=tn)


def predict_from_residual(train_res, test_res, quantile, scale, k):
    thr = float(np.quantile(train_res, quantile) * scale)
    return consecutive_rule((test_res > thr).astype(int), k), thr


def file_frames(path: Path, train_size: int = 400, window: int = 30):
    df = read_dataset(path)
    train_df, test_df = split_train_test(df, train_size)
    y_tr = window_end_labels(train_df["anomaly"].values.astype(int), window, 1)
    y_te = window_end_labels(test_df["anomaly"].values.astype(int), window, 1)
    return train_df, test_df, y_tr, y_te


# ---------------------------------------------------------------- c1
def stage_c1() -> None:
    df = pd.read_csv(RES / "residuals" / "residuals_w30.csv")
    rows = []
    for q, scale, tag in ((0.95, 1.0, "旧协议阈值规则 q95×1"),
                          (0.95, THRESHOLD_SCALE, "q95×4/3"),
                          (0.99, THRESHOLD_SCALE, "现行默认 q99×4/3")):
        recs = []
        for _, g in df.groupby("file"):
            tr, te = g[g.split == "train"], g[g.split == "test"]
            pred, _ = predict_from_residual(tr.residual_mse.values, te.residual_mse.values, q, scale, 1)
            recs.append((te.y_true.values, pred))
        m = pooled(recs)
        m.update(rule=tag, window=30, residual="mse", quantile=q, scale=scale, consecutive=1)
        rows.append(m)
    out = pd.DataFrame(rows)[["rule", "window", "residual", "quantile", "scale", "consecutive",
                              "precision", "recall", "f1", "far", "mar", "tp", "fp", "fn", "tn"]]
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "threshold_rule.csv", index=False, encoding="utf-8-sig")
    print(out[["rule", "precision", "recall", "f1", "far", "mar"]].to_string(index=False))
    print("→", OUT / "threshold_rule.csv")


# ---------------------------------------------------------------- selection
def stage_selection() -> None:
    """文件级二折：配置只在选择折上挑，评估折完全不参与选择。"""
    frames = {30: pd.read_csv(RES / "residuals" / "residuals_w30.csv"),
              60: pd.read_csv(RES / "residuals" / "residuals_w60.csv")}
    files = sorted(set(frames[30]["file"]))
    folds = [set(files[0::2]), set(files[1::2])]
    grid = [(w, r, q, k) for w in (30, 60)
            for r in ("residual_mse", "residual_mae_sum")
            for q in (0.99, 0.995, 0.999) for k in (1, 5, 20, 60)]

    def per_file_f1(cfg, subset):
        w, r, q, k = cfg
        df = frames[w]
        vals = []
        for f, g in df.groupby("file"):
            if f not in subset:
                continue
            tr, te = g[g.split == "train"], g[g.split == "test"]
            pred, _ = predict_from_residual(tr[r].values, te[r].values, q, THRESHOLD_SCALE, k)
            vals.append(binary_metrics(te.y_true.values, pred)["f1"])
        return float(np.mean(vals))

    def pooled_on(cfg, subset):
        w, r, q, k = cfg
        df = frames[w]
        recs = []
        for f, g in df.groupby("file"):
            if f not in subset:
                continue
            tr, te = g[g.split == "train"], g[g.split == "test"]
            pred, _ = predict_from_residual(tr[r].values, te[r].values, q, THRESHOLD_SCALE, k)
            recs.append((te.y_true.values, pred))
        return pooled(recs)

    picks, honest = [], []
    for i, evalfold in enumerate(folds):
        selfold = folds[1 - i]
        best, best_val = None, -1.0
        for cfg in grid:
            v = per_file_f1(cfg, selfold)
            if v > best_val:
                best, best_val = cfg, v
        m = pooled_on(best, evalfold)
        honest.append(m)
        picks.append(dict(fold=i + 1, n_selection=len(selfold), n_eval=len(evalfold),
                          window=best[0], residual=best[1], quantile=best[2], consecutive=best[3],
                          select_macro_f1=round(best_val, 4),
                          eval_precision=round(m["precision"], 4), eval_recall=round(m["recall"], 4),
                          eval_f1=round(m["f1"], 4), eval_far=round(m["far"], 4),
                          eval_mar=round(m["mar"], 4)))
    tot = {k: sum(m[k] for m in honest) for k in ("tp", "fp", "fn", "tn")}
    tp, fp, fn, tn = tot["tp"], tot["fp"], tot["fn"], tot["tn"]
    overall = dict(precision=tp / (tp + fp), recall=tp / (tp + fn),
                   f1=tp / (tp + 0.5 * (fp + fn)), far=fp / (fp + tn), mar=fn / (fn + tp))
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(picks).to_csv(OUT / "selection_folds.csv", index=False, encoding="utf-8-sig")
    (OUT / "selection_summary.json").write_text(json.dumps(
        {"n_files": len(files), "fold_size": len(folds[0]), "grid_size": len(grid),
         "overall": {k: round(v, 4) for k, v in overall.items()}, "folds": picks},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(pd.DataFrame(picks).to_string(index=False))
    print("无偏汇总：", {k: round(v, 4) for k, v in overall.items()})
    print("→", OUT / "selection_summary.json")


# ---------------------------------------------------------------- other/2
def stage_other2() -> None:
    path = DATA_DIR / "other" / "2.csv"
    train_df, test_df, y_tr, y_te = file_frames(path)
    sc = StandardScaler().fit(train_df[FEATURES].values.astype(np.float32))
    X_tr = sc.transform(train_df[FEATURES].values.astype(np.float32)).astype(np.float32)
    X_te = sc.transform(test_df[FEATURES].values.astype(np.float32)).astype(np.float32)
    rows = []
    for tag, mask in (("全部训练点（现行）", np.ones(len(y_tr), bool)),
                      ("丢弃含异常的窗口", y_tr == 0)):
        w_tr = make_windows(X_tr, WINDOW, 1)[mask]
        w_te = make_windows(X_te, WINDOW, 1)
        model = train_autoencoder(w_tr, WINDOW, epochs=30, batch_size=64, lr=1e-3,
                                  seed=SEED, device=device(), val_split=0.1, patience=5)
        e_tr = reconstruction_error(model, w_tr, batch_size=256, mode="mse")
        e_te = reconstruction_error(model, w_te, batch_size=256, mode="mse")
        pred, thr = predict_from_residual(e_tr, e_te, 0.99, THRESHOLD_SCALE, 1)
        m = binary_metrics(y_te, pred)
        m.update(setting=tag, n_train_windows=int(mask.sum()),
                 n_anomalous_train_windows=int((y_tr == 1).sum()), threshold=thr)
        rows.append(m)
    out = pd.DataFrame(rows)[["setting", "n_train_windows", "n_anomalous_train_windows", "threshold",
                              "precision", "recall", "f1", "far", "mar"]]
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "other2_sensitivity.csv", index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))
    print("→", OUT / "other2_sensitivity.csv")


# ---------------------------------------------------------------- global scaler
def stage_global() -> None:
    norm = np.load(ROOT / "data" / "processed" / "normalization.npz")
    mean, std = norm["mean"], norm["std"]
    rows = []
    for path in discover_files(DATA_DIR):
        rel = path.relative_to(DATA_DIR).as_posix()
        train_df, test_df, _, y_te = file_frames(path)
        X_tr = ((train_df[FEATURES].values - mean) / std).astype(np.float32)
        X_te = ((test_df[FEATURES].values - mean) / std).astype(np.float32)
        w_tr, w_te = make_windows(X_tr, WINDOW, 1), make_windows(X_te, WINDOW, 1)
        model = train_autoencoder(w_tr, WINDOW, epochs=30, batch_size=64, lr=1e-3,
                                  seed=SEED, device=device(), val_split=0.1, patience=5)
        e_tr = reconstruction_error(model, w_tr, batch_size=256, mode="mse")
        e_te = reconstruction_error(model, w_te, batch_size=256, mode="mse")
        pred, _ = predict_from_residual(e_tr, e_te, 0.95, 1.0, 1)
        m = binary_metrics(y_te, pred)
        m.update(file=rel)
        rows.append(m)
        print(f"  [{len(rows)}/34] {rel} F1={m['f1']:.3f}")
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "global_scaler_per_file.csv", index=False, encoding="utf-8-sig")
    tp, fp, fn, tn = int(out.tp.sum()), int(out.fp.sum()), int(out.fn.sum()), int(out.tn.sum())
    s = dict(precision=tp / (tp + fp), recall=tp / (tp + fn), f1=tp / (tp + 0.5 * (fp + fn)),
             far=fp / (fp + tn), mar=fn / (fn + tp), macro_f1=float(out.f1.mean()))
    (OUT / "global_scaler_summary.json").write_text(json.dumps(s, indent=2), encoding="utf-8")
    print("全局标准化 + 逐文件训练（q95×1）：", {k: round(v, 4) for k, v in s.items()})
    print("→", OUT / "global_scaler_summary.json")


# ---------------------------------------------------------------- official 1-channel
class OneChannelAE(nn.Module):
    """复刻官方实现：输入 8 通道，最后一层只输出 1 通道。"""

    def __init__(self, in_features: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_features, 32, 3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv1d(64, 32, 3, padding=1), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 1, 3, padding=1),
        )

    def forward(self, x):
        z = self.encoder(x)
        z = torch.nn.functional.interpolate(z, size=x.shape[-1], mode="linear", align_corners=False)
        return self.decoder(z)


def _train_one_channel(w_tr: np.ndarray, epochs: int = 30) -> OneChannelAE:
    dev = device()
    torch.manual_seed(SEED)
    model = OneChannelAE(in_features=w_tr.shape[-1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = to_channels_first(w_tr).to(dev)
    for _ in range(epochs):
        perm = torch.randperm(len(x), device=dev)
        for s in range(0, len(x), 64):
            b = x[perm[s:s + 64]]
            opt.zero_grad()
            loss = torch.mean((b - model(b)) ** 2)  # (B,8,T) 与 (B,1,T) 广播
            loss.backward()
            opt.step()
    model.eval()
    return model


def _err_one_channel(model: OneChannelAE, w: np.ndarray) -> np.ndarray:
    dev = device()
    x = to_channels_first(w)
    out = []
    with torch.no_grad():
        for s in range(0, len(x), 256):
            b = x[s:s + 256].to(dev)
            out.append(torch.abs(b.cpu() - model(b).cpu()).sum(dim=1).mean(dim=1).numpy())
    return np.concatenate(out)


def stage_official1ch() -> None:
    rows = []
    for window, q, k, epochs in ((30, 0.99, 1, 30), (60, 0.999, 60, 30)):
        recs = []
        for path in discover_files(DATA_DIR):
            rel = path.relative_to(DATA_DIR).as_posix()
            train_df, test_df, _, _ = file_frames(path, window=window)
            sc = StandardScaler().fit(train_df[FEATURES].values.astype(np.float32))
            w_tr = make_windows(sc.transform(train_df[FEATURES].values.astype(np.float32)), window, 1)
            w_te = make_windows(sc.transform(test_df[FEATURES].values.astype(np.float32)), window, 1)
            y_te = window_end_labels(test_df["anomaly"].values.astype(int), window, 1)
            model = _train_one_channel(w_tr, epochs)
            pred, _ = predict_from_residual(_err_one_channel(model, w_tr), _err_one_channel(model, w_te),
                                            q, THRESHOLD_SCALE, k)
            recs.append((y_te, pred))
            print(f"  [w{window}] {rel} F1={binary_metrics(y_te, pred)['f1']:.3f}")
        m = pooled(recs)
        m.update(window=window, quantile=q, consecutive=k)
        rows.append(m)
    out = pd.DataFrame(rows)[["window", "quantile", "consecutive", "precision", "recall", "f1", "far", "mar"]]
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "official_1ch.csv", index=False, encoding="utf-8-sig")
    print(out.to_string(index=False))
    print("→", OUT / "official_1ch.csv")




# ---------------------------------------------------------------- decomp
def stage_decomp() -> None:
    """把「旧方案 → 现行默认」拆成一串单因素步骤。

    旧训练超参：batch 256 / 50 轮 / patience 8（src/train_autoencoder.py）
    新训练超参：batch 32 / 60 轮 / patience 5（skab_protocol.train_autoencoder 默认）
    全局标准化统计量取自 data/processed/normalization.npz（anomaly-free 训练段拟合）。
    """
    norm = np.load(ROOT / "data" / "processed" / "normalization.npz")
    g_mean, g_std = norm["mean"], norm["std"]
    OLD = dict(batch_size=256, epochs=50, patience=8)
    NEW = dict(batch_size=32, epochs=60, patience=5)
    rows = []

    def run(setting, scaler, hyper, quantile, scale, k=1):
        recs = []
        for path in discover_files(DATA_DIR):
            df = read_dataset(path)
            train_df, test_df = split_train_test(df, 400)
            if scaler == "global":
                x_tr = (train_df[FEATURES].values - g_mean) / g_std
                x_te = (test_df[FEATURES].values - g_mean) / g_std
            else:
                sc = StandardScaler().fit(train_df[FEATURES].values)
                x_tr = sc.transform(train_df[FEATURES].values)
                x_te = sc.transform(test_df[FEATURES].values)
            w_tr = make_windows(x_tr.astype(np.float32), WINDOW, 1)
            w_te = make_windows(x_te.astype(np.float32), WINDOW, 1)
            y_te = window_end_labels(test_df["anomaly"].values.astype(int), WINDOW, 1)
            model = train_autoencoder(w_tr, WINDOW, seed=SEED, device=device(), **hyper)
            e_tr = reconstruction_error(model, w_tr, mode="mse")
            e_te = reconstruction_error(model, w_te, mode="mse")
            pred, _ = predict_from_residual(e_tr, e_te, quantile, scale, k)
            recs.append((y_te, pred))
        m = pooled(recs)
        m.update(setting=setting, scaler=scaler, quantile=quantile, scale=scale, consecutive=k,
                 batch_size=hyper["batch_size"], epochs=hyper["epochs"], patience=hyper["patience"])
        rows.append(m)
        print(f"{setting:42s} F1={m['f1']:.4f} FAR={m['far']:.4f} MAR={m['mar']:.4f}")

    run("A 逐文件数据（全局标准化、旧超参、q95×1）", "global", OLD, 0.95, 1.0)
    run("B 再加逐文件标准化（旧超参、q95×1）", "perfile", OLD, 0.95, 1.0)

    # E：新超参 + 逐文件标准化 + q99×1，直接复用残差缓存
    df = pd.read_csv(RES / "residuals" / "residuals_w30.csv")
    recs = []
    for _, g in df.groupby("file"):
        tr, te = g[g.split == "train"], g[g.split == "test"]
        pred, _ = predict_from_residual(tr.residual_mse.values, te.residual_mse.values, 0.99, 1.0, 1)
        recs.append((te.y_true.values, pred))
    m = pooled(recs)
    m.update(setting="E 阈值分位改 q99（系数仍 ×1）", scaler="perfile", quantile=0.99, scale=1.0,
             consecutive=1, batch_size=32, epochs=60, patience=5)
    rows.append(m)
    print(f"{m['setting']:42s} F1={m['f1']:.4f} FAR={m['far']:.4f} MAR={m['mar']:.4f}")

    out = pd.DataFrame(rows)[["setting", "scaler", "quantile", "scale", "consecutive",
                              "batch_size", "epochs", "patience",
                              "precision", "recall", "f1", "far", "mar", "tp", "fp", "fn", "tn"]]
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "decomp.csv", index=False, encoding="utf-8-sig")
    print("→", OUT / "decomp.csv")

STAGES = {"c1": stage_c1, "selection": stage_selection, "other2": stage_other2,
          "global": stage_global, "official1ch": stage_official1ch, "decomp": stage_decomp}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=list(STAGES) + ["all"], default="all")
    args = ap.parse_args()
    for name in (list(STAGES) if args.stage == "all" else [args.stage]):
        print(f"\n===== {name} =====")
        STAGES[name]()


if __name__ == "__main__":
    main()
