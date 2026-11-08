
from pathlib import Path

import numpy as np
import pandas as pd


# ==================================================
# 1. 路径与实验参数
# ==================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = ROOT / "data" / "raw" / "SKAB-master" / "data"

NORMAL_FILE = DATA_ROOT / "anomaly-free" / "anomaly-free.csv"

OUTPUT_DIR = ROOT / "data" / "processed"

WINDOW_SIZE = 30
TRAIN_RATIO = 0.8

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


# ==================================================
# 2. 读取 CSV 文件
# ==================================================

def read_csv_file(file_path):
    """读取使用分号分隔的 SKAB CSV 文件。"""

    df = pd.read_csv(file_path, sep=";")

    required = ["datetime"] + FEATURES

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(
            f"文件 {file_path.name} 缺少列：{missing}"
        )

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        errors="raise"
    )

    df[FEATURES] = df[FEATURES].apply(
        pd.to_numeric,
        errors="raise"
    )

    if df[FEATURES].isna().any().any():
        raise ValueError(f"文件存在缺失值：{file_path}")

    if not np.isfinite(df[FEATURES].to_numpy()).all():
        raise ValueError(f"文件存在无穷值：{file_path}")

    return df


# ==================================================
# 3. 时间序列窗口生成
# ==================================================

def make_windows(data, window_size):
    """
    将二维时间序列转换成三维窗口。

    输入形状：[时间步数, 特征数]
    输出形状：[窗口数量, 窗口长度, 特征数]
    """

    if len(data) < window_size:
        return np.empty(
            (0, window_size, data.shape[1]),
            dtype=np.float32
        )

    windows = np.lib.stride_tricks.sliding_window_view(
        data,
        window_size,
        axis=0
    )

    windows = np.moveaxis(windows, -1, 1)

    return np.ascontiguousarray(
        windows,
        dtype=np.float32
    )


# ==================================================
# 4. 读取正常数据并划分训练集、验证集
# ==================================================

def prepare_normal_data():

    print("\n========== 正常数据准备 ==========")

    df = read_csv_file(NORMAL_FILE)

    data = df[FEATURES].to_numpy(dtype=np.float32)

    split_index = int(len(data) * TRAIN_RATIO)

    train_data = data[:split_index]
    val_data = data[split_index:]

    print("正常数据总行数：", len(data))
    print("训练数据行数：", len(train_data))
    print("验证数据行数：", len(val_data))

    # 只使用训练数据计算标准化参数
    mean = train_data.mean(axis=0)
    std = train_data.std(axis=0)

    std[std < 1e-8] = 1.0

    train_scaled = (train_data - mean) / std
    val_scaled = (val_data - mean) / std

    train_windows = make_windows(
        train_scaled,
        WINDOW_SIZE
    )

    val_windows = make_windows(
        val_scaled,
        WINDOW_SIZE
    )

    print("训练窗口形状：", train_windows.shape)
    print("验证窗口形状：", val_windows.shape)

    return train_windows, val_windows, mean, std


# ==================================================
# 5. 准备带标签的测试数据
# ==================================================

def prepare_test_data(mean, std):

    print("\n========== 测试数据准备 ==========")

    test_files = sorted(
        DATA_ROOT.rglob("*.csv")
    )

    test_files = [
        f for f in test_files
        if "anomaly-free" not in f.parts
    ]

    all_windows = []
    all_labels = []
    file_names = []

    for file_path in test_files:

        df = read_csv_file(file_path)

        if "anomaly" not in df.columns:
            print("跳过没有 anomaly 标签的文件：", file_path.name)
            continue

        data = df[FEATURES].to_numpy(dtype=np.float32)

        labels = pd.to_numeric(
            df["anomaly"],
            errors="raise"
        ).to_numpy(dtype=np.int64)

        if not np.isin(labels, [0, 1]).all():
            raise ValueError(
                f"文件 {file_path.name} 的 anomaly 标签不是 0/1"
            )

        scaled = (data - mean) / std

        windows = make_windows(
            scaled,
            WINDOW_SIZE
        )

        # 每个窗口使用最后一个时间点的异常标签
        window_labels = labels[WINDOW_SIZE - 1:]

        if len(windows) != len(window_labels):
            raise ValueError(
                f"窗口和标签数量不一致：{file_path.name}"
            )

        all_windows.append(windows)
        all_labels.append(window_labels)

        file_names.extend(
            [str(file_path.relative_to(DATA_ROOT))]
            * len(window_labels)
        )

        print(
            f"{file_path.relative_to(DATA_ROOT)}："
            f"{len(windows)} 个窗口，"
            f"{int(window_labels.sum())} 个异常标签"
        )

    if not all_windows:
        raise RuntimeError("没有生成任何测试窗口！")

    X_test = np.concatenate(all_windows, axis=0)
    y_test = np.concatenate(all_labels, axis=0)

    print("\n测试窗口总数：", len(X_test))
    print("异常窗口数量：", int(y_test.sum()))
    print("正常窗口数量：", int((y_test == 0).sum()))

    return X_test, y_test, np.asarray(file_names)


# ==================================================
# 6. 主程序：保存处理结果
# ==================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    X_train, X_val, mean, std = prepare_normal_data()

    X_test, y_test, test_files = prepare_test_data(
        mean,
        std
    )

    # 保存标准化参数，后续测试新数据时也要使用
    np.savez(
        OUTPUT_DIR / "normalization.npz",
        mean=mean,
        std=std,
        features=np.asarray(FEATURES)
    )

    # 保存训练、验证和测试数据
    np.savez_compressed(
        OUTPUT_DIR / "skab_windows.npz",
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_test=y_test,
        test_files=test_files
    )

    print("\n========== 数据准备完成 ==========")
    print("训练集：", X_train.shape)
    print("验证集：", X_val.shape)
    print("测试集：", X_test.shape)
    print("测试标签：", y_test.shape)

    print("\n保存位置：")
    print(OUTPUT_DIR / "skab_windows.npz")
    print(OUTPUT_DIR / "normalization.npz")


if __name__ == "__main__":
    main()
