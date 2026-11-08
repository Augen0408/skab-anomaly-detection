import pandas as pd
from pathlib import Path

root = Path("data/raw/SKAB-master/data")
files = sorted(root.rglob("*.csv"))

print(f"数据文件总数：{len(files)}")
print("=" * 80)

summary = []

for file in files:
    try:
        df = pd.read_csv(file, sep=";")

        has_anomaly = "anomaly" in df.columns
        has_changepoint = "changepoint" in df.columns

        summary.append({
            "文件": str(file.relative_to(root)),
            "行数": len(df),
            "列数": len(df.columns),
            "有anomaly标签": has_anomaly,
            "有changepoint标签": has_changepoint,
            "列名": ", ".join(df.columns),
        })

    except Exception as e:
        print(f"读取失败：{file}")
        print(f"原因：{e}")

result = pd.DataFrame(summary)

print("\n各文件结构统计：")
print(
    result[
        ["文件", "行数", "列数", "有anomaly标签", "有changepoint标签"]
    ].to_string(index=False)
)

print("\n列数分布：")
print(result["列数"].value_counts().sort_index().to_string())

print("\n不含 anomaly 标签的文件：")
print(
    result.loc[~result["有anomaly标签"], "文件"].to_string(index=False)
)

output = Path("results/metrics")
output.mkdir(parents=True, exist_ok=True)

result.to_csv(
    output / "dataset_inventory.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n完整检查结果已保存到：results/metrics/dataset_inventory.csv")