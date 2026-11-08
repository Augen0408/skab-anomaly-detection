import pandas as pd
import numpy as np
from pathlib import Path

file = Path("data/raw/SKAB-master/data/anomaly-free/anomaly-free.csv")

df = pd.read_csv(file, sep=";")

sensor_cols = [
    "Accelerometer1RMS",
    "Accelerometer2RMS",
    "Current",
    "Pressure",
    "Temperature",
    "Thermocouple",
    "Voltage",
    "Volume Flow RateRMS"
]

df["datetime"] = pd.to_datetime(df["datetime"])

print("=" * 60)
print("正常数据详细检查")
print("=" * 60)

print("\n数据形状：", df.shape)

print("\n时间范围：")
print("开始：", df["datetime"].min())
print("结束：", df["datetime"].max())

print("\n时间间隔统计（秒）：")
print(df["datetime"].diff().dt.total_seconds().describe())

print("\n传感器数据统计：")
print(df[sensor_cols].describe().round(4).to_string())

print("\n是否存在缺失值：")
print(df[sensor_cols].isna().sum().to_string())

print("\n是否存在无穷值：")
print(np.isinf(df[sensor_cols].to_numpy()).sum())

print("\n检查完成。")