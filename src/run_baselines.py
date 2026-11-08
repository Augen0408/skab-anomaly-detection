"""对比基线：PCA 重构、Isolation Forest、LSTM-AE。

三者在与 Conv-AE 完全相同的逐文件协议下评估（前 400 行训练，
StandardScaler 逐文件 fit，窗口 30，阈值 = 训练残差 q=0.99 × 4/3，k=1），
以便直接与 results/ablation/sweep.csv 中的 Conv-AE 行比较。

产物：results/baselines/baseline_metrics.csv、per_file_<model>.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

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
    to_channels_first,
    train_autoencoder,
    window_end_labels,
    resolve_data_dir,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = resolve_data_dir(ROOT)
OUT = ROOT / "results" / "baselines"
WINDOW = 30
EPOCHS = 60
Q = 0.99


# --------------------------------------------------------------------------
# LSTM 自编码器
# --------------------------------------------------------------------------
class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int = 8, hidden: int = 32):
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden, batch_first=True)
        self.decoder = nn.LSTM(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.encoder(x)          # h: (1, B, hidden)
        z = h[-1].unsqueeze(1).repeat(1, x.shape[1], 1)
        y, _ = self.decoder(z)
        return self.out(y)


def train_lstm(x_train: np.ndarray, epochs: int, device, seed: int = 42) -> LSTMAutoencoder:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = LSTMAutoencoder(x_train.shape[-1]).to(device)
    tensor = torch.from_numpy(x_train.astype(np.float32)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    n = tensor.shape[0]
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, 32):
            b = tensor[perm[s : s + 32]]
            opt.zero_grad()
            loss = loss_fn(model(b), b)
            loss.backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def lstm_errors(model: LSTMAutoencoder, x: np.ndarray, batch_size: int = 256) -> np.ndarray:
    device = next(model.parameters()).device
    out = []
    for s in range(0, len(x), batch_size):
        b = torch.from_numpy(x[s : s + batch_size].astype(np.float32)).to(device)
        out.append(((model(b) - b) ** 2).mean(dim=(1, 2)).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="pca,iforest,lstm")
    args = ap.parse_args()
    alias = {"pca": "PCA", "iforest": "IsolationForest", "lstm": "LSTM-AE", "conv": "Conv-AE", "pcat2": "PCA-T2"}
    wanted = {alias[m.strip()] for m in args.models.split(",")}
    OUT.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files = discover_files(DATA_DIR)
    per = {m: [] for m in wanted}
    t0 = time.time()

    for i, path in enumerate(files, 1):
        rel = path.relative_to(DATA_DIR).as_posix()
        df = read_dataset(path)
        train_df, test_df = split_train_test(df)
        scaler = StandardScaler().fit(train_df[FEATURES].values)
        w_tr = make_windows(scaler.transform(train_df[FEATURES].values), WINDOW)
        w_te = make_windows(scaler.transform(test_df[FEATURES].values), WINDOW)
        y_te = window_end_labels(test_df["anomaly"].values.astype(int), WINDOW, 1)

        def record(name: str, e_tr: np.ndarray, e_te: np.ndarray) -> None:
            thr = threshold_from_train(e_tr, Q)
            pred = consecutive_rule((e_te > thr).astype(int), 1)
            m = binary_metrics(y_te, pred)
            m["file"] = rel
            per[name].append(m)

        if "PCA" in wanted:
            flat_tr, flat_te = w_tr.reshape(len(w_tr), -1), w_te.reshape(len(w_te), -1)
            pca = PCA(n_components=8, random_state=42).fit(flat_tr)
            e_tr = ((pca.inverse_transform(pca.transform(flat_tr)) - flat_tr) ** 2).mean(1)
            e_te = ((pca.inverse_transform(pca.transform(flat_te)) - flat_te) ** 2).mean(1)
            record("PCA", e_tr, e_te)

        if "PCA-T2" in wanted:
            flat_tr, flat_te = w_tr.reshape(len(w_tr), -1), w_te.reshape(len(w_te), -1)
            pca = PCA(n_components=8, random_state=42).fit(flat_tr)
            lam = pca.explained_variance_
            # Hotelling T^2：主成分空间中的马氏距离平方
            record("PCA-T2", (pca.transform(flat_tr) ** 2 / lam).sum(1),
                   (pca.transform(flat_te) ** 2 / lam).sum(1))

        if "IsolationForest" in wanted:
            flat_tr, flat_te = w_tr.reshape(len(w_tr), -1), w_te.reshape(len(w_te), -1)
            iso = IsolationForest(n_estimators=200, random_state=42, n_jobs=-1).fit(flat_tr)
            # score_samples 越大越正常，取负作为异常分
            record("IsolationForest", -iso.score_samples(flat_tr), -iso.score_samples(flat_te))

        if "LSTM-AE" in wanted:
            model = train_lstm(w_tr, EPOCHS, device)
            record("LSTM-AE", lstm_errors(model, w_tr), lstm_errors(model, w_te))

        if "Conv-AE" in wanted:
            model = train_autoencoder(
                w_tr, WINDOW, epochs=EPOCHS, batch_size=32, lr=1e-3, seed=42, device=device
            )
            record("Conv-AE", reconstruction_error(model, w_tr), reconstruction_error(model, w_te))

        print(f"[{i:2d}/{len(files)}] {rel}  " + "  ".join(
            f"{n}={per[n][-1]['f1']:.3f}" for n in per))

    rows = []
    for name, res in per.items():
        agg = pool_metrics(res)
        rows.append(dict(model=name, **{k: agg[k] for k in ("precision", "recall", "f1", "far", "mar")},
                         macro_f1=float(np.mean([d["f1"] for d in res]))))
        pd.DataFrame(res)[["file", "precision", "recall", "f1", "far", "mar", "tp", "fp", "fn", "tn"]
                          ].to_csv(OUT / f"per_file_{name.replace('-', '')}.csv",
                                   index=False, encoding="utf-8-sig")

    table = pd.DataFrame(rows).sort_values("f1", ascending=False)
    table.to_csv(OUT / "baseline_metrics.csv", index=False, encoding="utf-8-sig")
    print("\n=== 基线对比（逐文件协议，窗口 30，q=0.99，k=1）===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"用时 {time.time() - t0:.1f}s → {OUT}")


if __name__ == "__main__":
    main()
