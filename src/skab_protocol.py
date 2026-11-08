"""
SKAB 逐文件评估协议（protocol v2）。

对齐 SKAB 官方实现（core/utils.py + notebooks/Conv_AE.ipynb）：
  * 每个带标签 CSV 前 400 行作为训练段，其余作为测试段（shuffle=False）；
  * StandardScaler 只在各自文件的训练段上 fit；
  * 窗口长度 N_STEPS，步长 1，窗口不跨文件；
  * 阈值 = 训练段重构残差的 quantile(Q) * 4/3。

与旧版（src/prepare_data.py + src/evaluate_autoencoder.py）的区别：
旧版用 anomaly-free 单文件训练一个全局模型、用全局 mean/std 标准化所有文件，
因此测的是「离那次正常实验有多远」，而不是「有多异常」。本模块改为逐文件建模。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "Accelerometer1RMS",
    "Accelerometer2RMS",
    "Current",
    "Pressure",
    "Temperature",
    "Thermocouple",
    "Voltage",
    "Volume Flow RateRMS",
]
LABEL_COLUMNS = ["anomaly", "changepoint"]
TRAIN_SIZE = 400  # 官方协议：每个文件前 400 行做训练
THRESHOLD_SCALE = 4.0 / 3.0  # 官方：UCL = quantile(Q) * 4 / 3


# --------------------------------------------------------------------------
# 数据
# --------------------------------------------------------------------------
from skab_paths import resolve_data_dir  # noqa: E402,F401


def discover_files(data_dir: Path) -> list[Path]:
    """返回 34 个带标签 CSV（anomaly-free 不算）。"""
    return sorted(
        p for p in Path(data_dir).rglob("*.csv") if "anomaly-free" not in p.name.lower()
    )


def read_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    missing = [c for c in FEATURES + LABEL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} 缺少列：{missing}")
    return df


def split_train_test(df: pd.DataFrame, train_size: int = TRAIN_SIZE):
    """按时间顺序切分，不做 shuffle。"""
    return df.iloc[:train_size].copy(), df.iloc[train_size:].copy()


def make_windows(values: np.ndarray, window: int, stride: int = 1) -> np.ndarray:
    """(T, F) -> (N, window, F)，步长滑动。"""
    if len(values) < window:
        raise ValueError(f"序列长度 {len(values)} 小于窗口 {window}")
    n = (len(values) - window) // stride + 1
    idx = np.arange(n)[:, None] * stride + np.arange(window)[None, :]
    return values[idx]


def window_end_labels(labels: np.ndarray, window: int, stride: int = 1) -> np.ndarray:
    """窗口标签 = 窗口最后一个时间点的标签（与旧版一致，因果对齐）。"""
    n = (len(labels) - window) // stride + 1
    ends = np.arange(n) * stride + window - 1
    return labels[ends].astype(np.int64)


# --------------------------------------------------------------------------
# 模型
# --------------------------------------------------------------------------
class CNNAutoencoder(nn.Module):
    """与 src/train_autoencoder.py 完全一致的结构，便于对比。"""

    def __init__(self, input_features: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_features, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, input_features, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = nn.functional.interpolate(
            encoded, size=x.shape[-1], mode="linear", align_corners=False
        )
        return self.decoder(decoded)


def to_channels_first(x: np.ndarray) -> torch.Tensor:
    """(N, W, F) -> (N, F, W)"""
    return torch.from_numpy(np.transpose(x, (0, 2, 1)).astype(np.float32))


def train_autoencoder(
    x_train: np.ndarray,
    window: int,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    seed: int = 42,
    device: torch.device | None = None,
    val_split: float = 0.1,
    patience: int = 5,
    verbose: bool = False,
    split_seed: int | None = None,
    history: list | None = None,
    val_scheme: str = "random",
) -> CNNAutoencoder:
    """在单个文件的训练窗口上训练一个自编码器（10% 验证集 + 早停）。

    seed 控制初始化与批次顺序；split_seed 单独控制训练/验证划分，
    便于在固定划分的前提下做多种子重复。

    val_scheme:
      * "random"  —— 在全部训练窗口里随机抽 val_split（原协议，验证窗口与训练窗口
                     在原始时间点上大量重叠）；
      * "holdout" —— 按时间顺序取末尾 val_split 作验证，并在验证段之前留出
                     window-1 个窗口作为隔离带，使验证窗口与训练窗口的原始时间点
                     完全不重叠（审计用）；
      * "random_matched" —— 与 holdout 训练窗口数相同：随机抽 val_split 作验证，
                     再随机抽同样多的约束窗口整体丢弃（window-1 个），使训练窗
                     数量与 holdout 一致，但保留时间重叠（训练数量对照）。
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split_seed = seed if split_seed is None else split_seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = CNNAutoencoder(input_features=x_train.shape[-1]).to(device)
    tensor = to_channels_first(x_train)
    n = tensor.shape[0]

    n_val = int(round(n * val_split)) if val_split > 0 else 0
    if val_scheme == "holdout":
        # 末尾 n_val 个窗口作验证；其前 window-1 个窗口为隔离带，避免与训练窗口共享时间点
        val_start = n - n_val
        train_end = val_start - (window - 1)
        if train_end <= 0:
            raise ValueError("训练窗口不足，无法构造无重叠的验证集")
        train_idx = torch.arange(0, train_end)
        val_idx = torch.arange(val_start, n)
        val_tensor = tensor[val_idx].to(device)
        train_tensor = tensor[train_idx].to(device)
    elif val_scheme == "random":
        generator = torch.Generator().manual_seed(split_seed)
        perm = torch.randperm(n, generator=generator)
        val_tensor = tensor[perm[:n_val]].to(device)
        train_tensor = tensor[perm[n_val:]].to(device)
    elif val_scheme == "random_matched":
        # 训练窗数量与 holdout 对齐：n - n_val - (window-1)，其余随机丢弃
        n_train = n - n_val - (window - 1)
        if n_train <= 0:
            raise ValueError("训练窗口不足，无法构造匹配数量的训练集")
        generator = torch.Generator().manual_seed(split_seed)
        perm = torch.randperm(n, generator=generator)
        val_tensor = tensor[perm[:n_val]].to(device)
        train_tensor = tensor[perm[n_val : n_val + n_train]].to(device)
    else:
        raise ValueError(f"未知的 val_scheme: {val_scheme}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    n_train = train_tensor.shape[0]

    for epoch in range(epochs):
        model.train()
        ep_perm = torch.randperm(n_train, device=device)
        for start in range(0, n_train, batch_size):
            batch = train_tensor[ep_perm[start : start + batch_size]]
            optimizer.zero_grad()
            loss = criterion(model(batch), batch)
            loss.backward()
            optimizer.step()

        if n_val == 0:
            continue
        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(val_tensor), val_tensor).item()
        if verbose:
            print(f"    epoch {epoch + 1:3d}  val={val_loss:.6f}")
        if history is not None:
            history.append({"epoch": epoch + 1, "val_loss": val_loss,
                            "best_val": min(best_val, val_loss)})
        if val_loss < best_val - 1e-9:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def reconstruction_error(
    model: CNNAutoencoder, x: np.ndarray, batch_size: int = 256, mode: str = "mse"
) -> np.ndarray:
    """每个窗口的重构残差：(N,)。

    mode="mse"      : 时间维与特征维取平均的平方误差（本项目 v2 默认）
    mode="mae_sum"  : 官方 Conv_AE.ipynb 口径，sum_f( mean_t |x - x_hat| )
    """
    device = next(model.parameters()).device
    model.eval()
    out = []
    for start in range(0, len(x), batch_size):
        batch = to_channels_first(x[start : start + batch_size]).to(device)
        recon = model(batch)
        if mode == "mse":
            err = ((recon - batch) ** 2).mean(dim=(1, 2))
        elif mode == "mae_sum":
            err = (recon - batch).abs().mean(dim=2).sum(dim=1)
        else:
            raise ValueError(f"未知的 residual mode: {mode}")
        out.append(err.detach().cpu().numpy())
    return np.concatenate(out).astype(np.float64)


# --------------------------------------------------------------------------
# 评估
# --------------------------------------------------------------------------
def threshold_from_train(train_errors: np.ndarray, quantile: float = 0.99) -> float:
    return float(np.quantile(train_errors, quantile) * THRESHOLD_SCALE)


def consecutive_rule(pred: np.ndarray, k: int) -> np.ndarray:
    """连续 k 个窗口都超阈才报警（抑制孤立误报）。"""
    if k <= 1:
        return pred.astype(np.int64)
    out = np.zeros_like(pred, dtype=np.int64)
    run = 0
    for i, v in enumerate(pred):
        run = run + 1 if v else 0
        if run >= k:
            out[i] = 1
    return out


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    far = fp / (fp + tn) if (fp + tn) else 0.0  # 误报率
    mar = fn / (fn + tp) if (fn + tp) else 0.0  # 漏报率
    return dict(
        tp=tp, fp=fp, fn=fn, tn=tn,
        precision=precision, recall=recall, f1=f1, far=far, mar=mar,
    )


def pool_metrics(per_file: list[dict]) -> dict:
    """按窗口数加权的汇总（等价于把所有文件窗口拼起来算）。"""
    tp = sum(d["tp"] for d in per_file)
    fp = sum(d["fp"] for d in per_file)
    fn = sum(d["fn"] for d in per_file)
    tn = sum(d["tn"] for d in per_file)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return dict(
        tp=tp, fp=fp, fn=fn, tn=tn,
        precision=precision, recall=recall, f1=f1,
        far=fp / (fp + tn) if (fp + tn) else 0.0,
        mar=fn / (fn + tp) if (fn + tp) else 0.0,
    )


@dataclass
class FileResult:
    file: str
    n_train: int
    n_test: int
    anomaly_ratio: float
    threshold: float
    metrics: dict
    residual: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    window_end: np.ndarray
