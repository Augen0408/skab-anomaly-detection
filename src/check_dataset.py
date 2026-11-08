import pandas as pd
from pathlib import Path

# 数据集目录
root = Path("data/raw/SKAB-master/data")

# 查找所有 CSV 文件
files = sorted(root.rglob("*.csv"))

print(f"CSV 文件总数：{len(files)}")

# 统计前 10 个文件
rows = []

for file in files[:10]:
    df = pd.read_csv(file, sep=";")

    anomaly_ratio = None
    if "anomaly" in df.columns:
        anomaly_ratio = round((df["anomaly"] == 1).mean() * 100, 2)

    rows.append({
        "文件": str(file.relative_to(root)),
        "行数": len(df),
        "列数": len(df.columns),
        "异常比例(%)": anomaly_ratio,
    })

print("\n前 10 个文件的统计结果：")
print(pd.DataFrame(rows).to_string(index=False))