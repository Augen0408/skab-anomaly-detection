"""数据目录定位（不依赖 torch，供报告/图表脚本使用）。"""
from __future__ import annotations

from pathlib import Path


def resolve_data_dir(root: Path | None = None) -> Path:
    """定位 SKAB CSV 根目录，兼容不同解压层级。

    优先 data/raw/SKAB-master/data；其次在 data/raw 下递归寻找第一个
    含 valve1/ 或 other/ 子目录的目录，保证文件名标签仍是 valve1/1.csv。
    """
    root = Path(root) if root else Path(__file__).resolve().parents[1]
    for cand in (root / "data" / "raw" / "SKAB-master" / "data", root / "data" / "raw"):
        if not cand.is_dir():
            continue
        for sub in [cand, *sorted(p for p in cand.rglob("*") if p.is_dir())]:
            if (sub / "valve1").is_dir() or (sub / "other").is_dir():
                return sub
    raise FileNotFoundError(
        f"未找到 SKAB 数据目录，请把 CSV 放到 {root / 'data' / 'raw'} 下（见 README「数据」一节）"
    )
