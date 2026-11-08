from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)


# =====================================================
# 1. 基础设置
# =====================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = Path("data/processed/skab_windows.npz")
MODEL_PATH = Path("checkpoints/cnn_autoencoder_best.pth")

OUTPUT_DIR = Path("results/metrics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 使用验证集正常数据重构误差的 95% 分位数作为阈值
THRESHOLD_PERCENTILE = 95


print("=" * 60)
print("CNN 自编码器异常检测评估程序")
print("=" * 60)

print("使用设备：", DEVICE)

if DEVICE.type == "cuda":
    print("GPU 名称：", torch.cuda.get_device_name(0))


# =====================================================
# 2. 定义模型
# 必须与训练代码中的模型结构保持一致
# =====================================================

class CNNAutoencoder(nn.Module):

    def __init__(self, input_features=8):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(
                input_features, 32,
                kernel_size=3, padding=1
            ),
            nn.ReLU(),

            nn.Conv1d(
                32, 64,
                kernel_size=3, stride=2, padding=1
            ),
            nn.ReLU(),

            nn.Conv1d(
                64, 32,
                kernel_size=3, padding=1
            ),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Conv1d(
                32, 32,
                kernel_size=3, padding=1
            ),
            nn.ReLU(),

            nn.Conv1d(
                32, input_features,
                kernel_size=3, padding=1
            )
        )

    def forward(self, x):

        encoded = self.encoder(x)

        decoded = nn.functional.interpolate(
            encoded,
            size=x.shape[-1],
            mode="linear",
            align_corners=False
        )

        reconstructed = self.decoder(decoded)

        return reconstructed


# =====================================================
# 3. 读取数据
# =====================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(f"找不到数据文件：{DATA_PATH}")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"找不到模型文件：{MODEL_PATH}")


with np.load(DATA_PATH, allow_pickle=False) as data:

    X_val = data["X_val"].astype(np.float32)
    X_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].astype(np.int64)


print("\n数据读取完成")
print("验证集形状：", X_val.shape)
print("测试集形状：", X_test.shape)
print("测试标签形状：", y_test.shape)

print("测试集异常窗口数：", int((y_test == 1).sum()))
print("测试集正常窗口数：", int((y_test == 0).sum()))


# 原始形状是：
# (样本数量, 时间长度, 特征数量)
#
# Conv1d 要求：
# (样本数量, 特征数量, 时间长度)

X_val = np.transpose(X_val, (0, 2, 1))
X_test = np.transpose(X_test, (0, 2, 1))


# =====================================================
# 4. 加载训练好的模型
# =====================================================

model = CNNAutoencoder(input_features=8).to(DEVICE)

state_dict = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(state_dict)

model.eval()

print("\n模型加载成功")


# =====================================================
# 5. 计算每个窗口的重构误差
# =====================================================

def calculate_reconstruction_errors(data, batch_size=256):

    errors = []

    with torch.no_grad():

        for start in range(0, len(data), batch_size):

            end = min(start + batch_size, len(data))

            batch = torch.from_numpy(
                data[start:end]
            ).to(DEVICE)

            reconstructed = model(batch)

            # 每个窗口在所有特征和时间点上的平均平方误差
            batch_errors = torch.mean(
                (reconstructed - batch) ** 2,
                dim=(1, 2)
            )

            errors.extend(
                batch_errors.cpu().numpy()
            )

    return np.asarray(errors)


print("\n正在计算验证集重构误差……")

val_errors = calculate_reconstruction_errors(X_val)

print("验证集误差计算完成")
print("验证集误差形状：", val_errors.shape)


print("\n正在计算测试集重构误差……")

test_errors = calculate_reconstruction_errors(X_test)

print("测试集误差计算完成")
print("测试集误差形状：", test_errors.shape)


# =====================================================
# 6. 使用验证集确定异常阈值
# =====================================================

threshold = np.percentile(
    val_errors,
    THRESHOLD_PERCENTILE
)

print("\n" + "=" * 60)
print("异常阈值")
print("=" * 60)

print(
    f"验证集误差的 {THRESHOLD_PERCENTILE}% 分位数："
    f"{threshold:.8f}"
)


# =====================================================
# 7. 根据阈值进行异常检测
# =====================================================

# 重构误差大于阈值：判定为异常
# 否则：判定为正常

y_pred = (test_errors > threshold).astype(np.int64)


# =====================================================
# 8. 计算评估指标
# =====================================================

tn, fp, fn, tp = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
).ravel()

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

accuracy = accuracy_score(
    y_test,
    y_pred
)

# 误报率：正常窗口中被误判为异常的比例
false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0


# =====================================================
# 9. 输出结果
# =====================================================

print("\n" + "=" * 60)
print("测试集评估结果")
print("=" * 60)

print(f"阈值：             {threshold:.8f}")
print(f"Accuracy：         {accuracy:.4f}")
print(f"Precision：        {precision:.4f}")
print(f"Recall：           {recall:.4f}")
print(f"F1：               {f1:.4f}")
print(f"误报率：           {false_alarm_rate:.4f}")

print("\n混淆矩阵统计：")

print(f"TN（正常判正常）： {tn}")
print(f"FP（正常判异常）： {fp}")
print(f"FN（异常判正常）： {fn}")
print(f"TP（异常判异常）： {tp}")


# =====================================================
# 10. 保存预测结果
# =====================================================

prediction_path = OUTPUT_DIR / "test_predictions.csv"

prediction_df = pd.DataFrame({
    "true_label": y_test,
    "reconstruction_error": test_errors,
    "predicted_label": y_pred
})

prediction_df.to_csv(
    prediction_path,
    index=False,
    encoding="utf-8-sig"
)


# =====================================================
# 11. 保存评估指标
# =====================================================

metrics_path = OUTPUT_DIR / "evaluation_metrics.csv"

metrics_df = pd.DataFrame([{
    "threshold_percentile": THRESHOLD_PERCENTILE,
    "threshold": threshold,
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "false_alarm_rate": false_alarm_rate,
    "TN": tn,
    "FP": fp,
    "FN": fn,
    "TP": tp
}])

metrics_df.to_csv(
    metrics_path,
    index=False,
    encoding="utf-8-sig"
)


print("\n" + "=" * 60)
print("评估完成")
print("=" * 60)

print("预测结果保存位置：", prediction_path)
print("评估指标保存位置：", metrics_path)