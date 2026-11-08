
import pandas as pd
from pathlib import Path

root = Path("data/raw/SKAB-master/data")

files = sorted(
    list(root.glob("other/*.csv"))
    + list(root.glob("valve1/*.csv"))
    + list(root.glob("valve2/*.csv"))
)

rows = []

for f in files:
    df = pd.read_csv(f, sep=";")

    rows.append({
        "文件": str(f.relative_to(root)),
        "行数": len(df),
        "异常数量": int((df["anomaly"] == 1).sum()),
        "正常数量": int((df["anomaly"] == 0).sum()),
        "异常比例(%)": round(
            (df["anomaly"] == 1).mean() * 100, 2
        ),
        "开始时间": df["datetime"].iloc[0],
        "结束时间": df["datetime"].iloc[-1],
    })

result = pd.DataFrame(rows)

print("测试文件总数：", len(result))
print("=" * 100)
print(result.to_string(index=False))

print("\n总体统计：")
print("总行数：", result["行数"].sum())
print("总异常数量：", result["异常数量"].sum())
print(
    "总体异常比例(%)：",
    round(
        result["异常数量"].sum()
        / result["行数"].sum() * 100,
        2
    )
)

output = Path("results/metrics")
output.mkdir(parents=True, exist_ok=True)

result.to_csv(
    output / "sequence_inventory.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n统计文件已保存：")
print("results/metrics/sequence_inventory.csv")
