"""对比三种验证集划分：random / random_matched / holdout。

读取三个结果目录的 per_seed_metrics.csv、per_file_metrics.csv 与
training_history.csv，输出统一的对照表到 results/validation_schemes/。
不重训、不改动任何已有结果。
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCHEMES = {
    "random": ROOT / "results" / "repeats",
    "random_matched": ROOT / "results" / "repeats_matched",
    "holdout": ROOT / "results" / "repeats_holdout",
}
METRICS = ["precision", "recall", "f1", "far", "mar", "macro_f1"]
OUT = ROOT / "results" / "validation_schemes"


def best_epoch_and_loss(d: Path) -> pd.DataFrame:
    frames = []
    for seed_dir in sorted(d.glob("seed_*")):
        hist = pd.read_csv(seed_dir / "training_history.csv")
        hist["seed"] = int(seed_dir.name.split("_")[1])
        frames.append(hist)
    hist = pd.concat(frames, ignore_index=True)
    idx = hist.groupby(["seed", "file"])["val_loss"].idxmin()
    best = hist.loc[idx, ["seed", "file", "epoch", "val_loss"]]
    return best.rename(columns={"epoch": "best_epoch", "val_loss": "min_val_loss"})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    per_seed, per_file, vl = [], [], []
    for tag, d in SCHEMES.items():
        seed_df = pd.read_csv(d / "per_seed_metrics.csv")
        seed_df["scheme"] = tag
        per_seed.append(seed_df)

        for seed_dir in sorted(d.glob("seed_*")):
            pf = pd.read_csv(seed_dir / "per_file_metrics.csv")
            pf["scheme"] = tag
            pf["seed"] = int(seed_dir.name.split("_")[1])
            per_file.append(pf[["scheme", "seed", "file", "precision", "recall", "f1", "far", "mar"]])

        be = best_epoch_and_loss(d)
        be["scheme"] = tag
        vl.append(be)

    seeds = pd.concat(per_seed, ignore_index=True)
    seeds.to_csv(OUT / "per_seed_metrics.csv", index=False, encoding="utf-8-sig")

    files = pd.concat(per_file, ignore_index=True)
    files.to_csv(OUT / "per_file_metrics.csv", index=False, encoding="utf-8-sig")

    best = pd.concat(vl, ignore_index=True)
    best.to_csv(OUT / "best_epoch.csv", index=False, encoding="utf-8-sig")

    rows = []
    for tag in SCHEMES:
        sdf = seeds[seeds.scheme == tag]
        bdf = best[best.scheme == tag]
        for m in METRICS:
            rows.append(dict(scheme=tag, metric=m, mean=sdf[m].mean(), std=sdf[m].std(ddof=1),
                             min=sdf[m].min(), max=sdf[m].max(), n=len(sdf)))
        for m, col in (("min_val_loss", "min_val_loss"), ("best_epoch", "best_epoch")):
            rows.append(dict(scheme=tag, metric=m, mean=bdf[col].mean(), std=bdf[col].std(ddof=1),
                             min=bdf[col].min(), max=bdf[col].max(), n=len(bdf)))
    stat = pd.DataFrame(rows)
    stat.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")

    wide = stat.pivot(index="metric", columns="scheme", values="mean")[list(SCHEMES)]
    print("=== 均值（三种子）===")
    print(wide.round(4).to_string())

    print("\n=== 逐种子 F1 ===")
    print(seeds.pivot(index="seed", columns="scheme", values="f1").round(4).to_string())

    base = wide["random"]
    print("\n=== 相对 random 的差值 ===")
    print((wide.sub(base, axis=0)).round(4).to_string())


if __name__ == "__main__":
    main()
