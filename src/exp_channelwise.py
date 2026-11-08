"""按通道拆分的重构残差：比较 sum / max / top-2 三种聚合方式的检测效果。

动机：8 个传感器中往往只有 1~2 个对某类故障敏感，sum 聚合会被其余通道稀释。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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
    split_train_test,
    threshold_from_train,
    train_autoencoder,
    window_end_labels,
    resolve_data_dir,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = resolve_data_dir(ROOT)
OUT = ROOT / "results" / "ablation"
WINDOW = 30
EPOCHS = 60


@torch.no_grad()
def channel_errors(model, x: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """返回 (N, F)：每个通道在时间维上的平均绝对误差。"""
    device = next(model.parameters()).device
    model.eval()
    out = []
    for s in range(0, len(x), batch_size):
        batch = torch.from_numpy(
            np.transpose(x[s : s + batch_size], (0, 2, 1)).astype(np.float32)
        ).to(device)
        rec = model(batch)
        out.append((rec - batch).abs().mean(dim=2).detach().cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    files = discover_files(DATA_DIR)
    rules = {
        "sum": lambda e: e.sum(axis=1),
        "max": lambda e: e.max(axis=1),
        "top2": lambda e: np.sort(e, axis=1)[:, -2:].mean(axis=1),
    }
    per_file = {name: [] for name in rules}
    records = []

    for i, path in enumerate(files, 1):
        rel = path.relative_to(DATA_DIR).as_posix()
        df = read_dataset(path)
        train_df, test_df = split_train_test(df)
        scaler = StandardScaler().fit(train_df[FEATURES].values)
        w_tr = make_windows(scaler.transform(train_df[FEATURES].values), WINDOW)
        w_te = make_windows(scaler.transform(test_df[FEATURES].values), WINDOW)
        y_te = window_end_labels(test_df["anomaly"].values.astype(int), WINDOW, 1)

        model = train_autoencoder(
            w_tr, WINDOW, epochs=EPOCHS, batch_size=32, lr=1e-3, seed=42, device=device
        )
        e_tr = channel_errors(model, w_tr)
        e_te = channel_errors(model, w_te)

        rec = {"file": rel}
        for name, fn in rules.items():
            thr = threshold_from_train(fn(e_tr), 0.99)
            pred = consecutive_rule((fn(e_te) > thr).astype(int), 1)
            m = binary_metrics(y_te, pred)
            m["file"] = rel
            per_file[name].append(m)
            rec[name] = m["f1"]
        records.append(rec)

        # 记录每个通道在「异常/正常」上的残差倍数，用于分析哪些通道有效
        for c, feat in enumerate(FEATURES):
            anom = e_te[y_te == 1, c]
            norm = e_te[y_te == 0, c]
            if len(anom) and len(norm):
                rec[f"gain_{feat}"] = float(np.median(anom) / (np.median(norm) + 1e-9))
        print(f"[{i:2d}/{len(files)}] {rel}  " + "  ".join(f"{k}={rec[k]:.3f}" for k in rules))

    rows = []
    for name, per in per_file.items():
        agg = pool_metrics(per)
        rows.append(
            dict(
                rule=name,
                **{k: agg[k] for k in ("precision", "recall", "f1", "far", "mar")},
                macro_f1=float(np.mean([d["f1"] for d in per])),
            )
        )
    table = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "channelwise.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(records).to_csv(
        OUT / "channelwise_per_file.csv", index=False, encoding="utf-8-sig"
    )

    print("\n=== 通道聚合方式对比 ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    gains = pd.DataFrame(records)[[f"gain_{f}" for f in FEATURES]]
    print("\n各通道 异常/正常 残差比（34 文件的中位数）")
    print(gains.median().round(2).to_string())


if __name__ == "__main__":
    main()
