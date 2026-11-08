"""生成报告用图。所有图输出到 results/figures/（PNG, 200 dpi，中文字体）。"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from skab_protocol import resolve_data_dir, FEATURES, discover_files, read_dataset  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = resolve_data_dir(ROOT)
FIG = ROOT / "results" / "figures"
RES = ROOT / "results"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 200


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"→ {path}")


# --------------------------------------------------------------------------
def fig_series() -> None:
    """图 1：一个带标签文件的关键传感器曲线与异常区间。"""
    df = read_dataset(DATA_DIR / "valve1" / "1.csv")
    t = np.arange(len(df))
    chans = ["Volume Flow RateRMS", "Accelerometer1RMS", "Temperature"]
    fig, axes = plt.subplots(len(chans), 1, figsize=(8, 5.2), sharex=True)
    for ax, c in zip(axes, chans):
        ax.plot(t, df[c].values, lw=0.9, color="#1f4e79")
        ax.set_ylabel(c, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.25, lw=0.4)
    anom = df["anomaly"].values == 1
    for ax in axes:
        ax.fill_between(t, *ax.get_ylim(), where=anom, color="#d62728", alpha=0.12, lw=0)
        ax.axvline(400, color="#7f7f7f", ls="--", lw=1)
    axes[0].annotate("训练段 ← 400 → 测试段", xy=(400, 0.02), xycoords=("data", "axes fraction"),
                     fontsize=7, color="#7f7f7f", ha="center")
    axes[0].set_title("valve1/1.csv：前 400 行为训练段，红色阴影为真实异常区间", fontsize=9)
    axes[-1].set_xlabel("采样点")
    save(fig, "fig1_series.png")


def fig_residual() -> None:
    """图 2：训练段 / 测试正常 / 测试异常的残差分布。"""
    df = pd.read_csv(RES / "residuals" / "residuals_w30.csv")
    sub = df[df.file == "valve1/1.csv"]
    tr = sub[sub.split == "train"].residual_mse.values
    te = sub[sub.split == "test"]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    bins = np.linspace(0, np.percentile(sub.residual_mse, 99.5), 60)
    ax.hist(tr, bins=bins, alpha=0.55, label="训练段", color="#4c72b0", density=True)
    ax.hist(te[te.y_true == 0].residual_mse, bins=bins, alpha=0.55,
            label="测试段·正常", color="#55a868", density=True)
    ax.hist(te[te.y_true == 1].residual_mse, bins=bins, alpha=0.55,
            label="测试段·异常", color="#c44e52", density=True)
    ax.axvline(np.quantile(tr, 0.99) * 4 / 3, color="k", ls="--", lw=1.2, label="阈值 (q=0.99)")
    ax.set_xlabel("窗口重构残差 (MSE)")
    ax.set_ylabel("密度")
    ax.set_title("valve1/1.csv 的残差分布：训练段与测试段整体错位", fontsize=9)
    ax.legend(fontsize=7)
    save(fig, "fig2_residual.png")


def fig_per_file() -> None:
    """图 3：逐文件 F1 —— Conv-AE vs PCA。"""
    conv = pd.read_csv(RES / "baselines" / "per_file_ConvAE.csv")
    pca = pd.read_csv(RES / "baselines" / "per_file_PCA.csv")
    m = conv[["file", "f1"]].merge(pca[["file", "f1"]], on="file", suffixes=("_conv", "_pca"))
    m = m.sort_values("f1_conv").reset_index(drop=True)
    x = np.arange(len(m))
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.bar(x - 0.2, m.f1_conv, width=0.4, label="Conv-AE", color="#4c72b0")
    ax.bar(x + 0.2, m.f1_pca, width=0.4, label="PCA", color="#dd8452")
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace(".csv", "") for f in m.file], rotation=90, fontsize=6)
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25, lw=0.4)
    ax.set_title("逐文件 F1：PCA 在多数文件上不弱于 Conv-AE", fontsize=9)
    save(fig, "fig3_per_file.png")


def fig_failure() -> None:
    """图 4：三类典型失败模式。"""
    pred = pd.read_csv(RES / "protocol_v2" / "predictions.csv")
    metr = pd.read_csv(RES / "protocol_v2" / "per_file_metrics.csv").set_index("file")
    cases = [
        ("other/13.csv", "(a) 完全漏检：异常区间的残差与正常段无差别"),
        ("valve1/12.csv", "(b) 大段误报：残差整体被抬到阈值之上"),
        ("other/4.csv", "(c) 部分漏检：只有部分异常段被检出"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(8, 6.4))
    for ax, (fname, title) in zip(axes, cases):
        sub = pred[pred.file == fname].sort_values("window_end")
        x = sub.window_end.values
        thr = float(metr.loc[fname, "threshold"])
        ax.fill_between(x, 0, 1, where=sub.y_true.values == 1, transform=ax.get_xaxis_transform(),
                        color="#c44e52", alpha=0.15, lw=0, label="真实异常")
        ax.fill_between(x, 0, 1, where=sub.predicted.values == 1, transform=ax.get_xaxis_transform(),
                        color="#4c72b0", alpha=0.15, lw=0, label="预测异常")
        ax.plot(x, sub.residual.values, lw=0.9, color="#1f4e79", label="重构残差")
        ax.axhline(thr, color="k", ls="--", lw=1, label=f"阈值 {thr:.3f}")
        ax.set_ylabel("残差", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title(title, fontsize=8.5, loc="left")
        ax.grid(alpha=0.25, lw=0.4)
    axes[0].legend(fontsize=7, loc="upper right", ncol=2)
    axes[-1].set_xlabel("窗口末端采样点")
    save(fig, "fig4_failure.png")


def fig_ablation() -> None:
    """图 5：消融热力图（窗口 30 / MSE 下的 F1）。"""
    sweep = pd.read_csv(RES / "ablation" / "sweep.csv")
    sub = sweep[(sweep.window == 30) & (sweep.residual == "mse")]
    piv = sub.pivot_table(index="consecutive", columns="quantile", values="f1")
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    im = ax.imshow(piv.values, cmap="YlGnBu", aspect="auto", vmin=0.70, vmax=0.80)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels([f"q={c}" for c in piv.columns], fontsize=8)
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels([f"k={c}" for c in piv.index], fontsize=8)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.3f}", ha="center", va="center", fontsize=7)
    ax.set_title("F1 对阈值分位与连续窗口数不敏感", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.85, label="F1")
    save(fig, "fig5_ablation.png")


def fig_cross_condition() -> None:
    """图 6：旧协议的根因——各文件相对 anomaly-free 统计量的平均 z 值。"""
    z_mean = np.load(ROOT / "data" / "processed" / "normalization.npz")
    mean, std = z_mean["mean"], z_mean["std"]
    rows = []
    for path in discover_files(DATA_DIR):
        rel = path.relative_to(DATA_DIR).as_posix()
        vals = read_dataset(path)[FEATURES].values
        z = (vals - mean) / std
        rows.append(dict(file=rel, **{f: z[:, i].mean() for i, f in enumerate(FEATURES)}))
    t = pd.DataFrame(rows).sort_values("Volume Flow RateRMS")
    fig, ax = plt.subplots(figsize=(9, 3.0))
    x = np.arange(len(t))
    for f, c in zip(["Accelerometer1RMS", "Volume Flow RateRMS", "Temperature"],
                    ["#4c72b0", "#dd8452", "#55a868"]):
        ax.plot(x, t[f].values, marker="o", ms=2.5, lw=0.9, label=f, color=c)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace(".csv", "") for f in t.file], rotation=90, fontsize=6)
    ax.set_ylabel("平均 z 值")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25, lw=0.4)
    ax.set_title("各文件相对 anomaly-free 统计量的平均 z 值：稳定的工况级平移", fontsize=9)
    save(fig, "fig6_cross_condition.png")


if __name__ == "__main__":
    fig_series()
    fig_residual()
    fig_per_file()
    fig_failure()
    fig_ablation()
    fig_cross_condition()
