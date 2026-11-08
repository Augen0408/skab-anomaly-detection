import pandas as pd

normal_file = "data/raw/SKAB-master/data/anomaly-free/anomaly-free.csv"
test_file = "data/raw/SKAB-master/data/valve1/0.csv"

normal = pd.read_csv(normal_file, sep=";")
test = pd.read_csv(test_file, sep=";")

print("=" * 50)
print("正常数据文件")
print("形状：", normal.shape)
print("列名：")
for col in normal.columns:
    print("  ", col)

print("\n" + "=" * 50)
print("带标签的数据文件")
print("形状：", test.shape)
print("列名：")
for col in test.columns:
    print("  ", col)

print("\n" + "=" * 50)
print("两类文件共有的列：")
print(sorted(set(normal.columns) & set(test.columns)))

print("\n正常文件前3行：")
print(normal.head(3).to_string(index=False))

print("\n带标签文件前3行：")
print(test.head(3).to_string(index=False))