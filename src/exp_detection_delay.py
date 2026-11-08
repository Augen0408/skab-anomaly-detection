"""检测延迟分析：故障起点到首次报警之间隔了多少个采样点。

输入 results/protocol_v2/predictions.csv（逐文件协议、窗长 30、q=0.99、k=1）
输出 results/ablation/detection_delay.csv 与终端汇总。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def analyse(pred_path: Path, label: str) -> pd.DataFrame:
    pred = pd.read_csv(pred_path)
    rows = []
    for fname, sub in pred.groupby("file"):
        sub = sub.sort_values("window_end")
        y = sub.y_true.to_numpy()
        if not (y == 1).any():
            continue
        x = sub.window_end.to_numpy()
        alarm = sub.predicted.to_numpy()
        onset = int(x[np.argmax(y == 1)])
        after = np.where((x >= onset) & (alarm == 1))[0]
        detected = len(after) > 0
        first = int(x[after[0]]) if detected else None
        rows.append(
            {
                "config": label,
                "file": fname,
                "onset": onset,
                "first_alarm": first,
                "delay_samples": (first - onset) if detected else None,
                "detected": bool(detected),
                "anomaly_len": int((y == 1).sum()),
            }
        )
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, label: str) -> None:
    det = df[df.detected]
    print(f"\n=== 检测延迟 · {label} ===")
    print(f"含故障文件 {len(df)} 个，首次报警命中 {len(det)} 个（{len(det) / len(df) * 100:.1f}%）")
    if len(det):
        d = det.delay_samples
        print(f"延迟（采样点）：中位 {d.median():.0f}，均值 {d.mean():.1f}，"
              f"IQR {d.quantile(.25):.0f}–{d.quantile(.75):.0f}，最大 {d.max():.0f}")
    for group in ("valve1", "valve2", "other"):
        g = df[df.file.str.startswith(group)]
        gd = g[g.detected]
        if len(gd):
            print(f"  {group}: 命中 {len(gd)}/{len(g)}，延迟中位 {gd.delay_samples.median():.0f}")
    miss = df[~df.detected]
    if len(miss):
        print("  完全未报警：" + "、".join(miss.file.tolist()))


if __name__ == "__main__":
    v2 = analyse(RES / "protocol_v2" / "predictions.csv", "v2(w30,q0.99,k1)")
    off = analyse(RES / "protocol_v2_official" / "predictions.csv", "official(w60,q0.999,k60)")
    out = pd.concat([v2, off], ignore_index=True)
    out.to_csv(RES / "ablation" / "detection_delay.csv", index=False, encoding="utf-8-sig")
    summarise(v2, "v2")
    summarise(off, "official")
    print("\n→", RES / "ablation" / "detection_delay.csv")
