
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# =====================================================
# 1. 基础设置
# =====================================================

SEED = 42
BATCH_SIZE = 256
EPOCHS = 50
LEARNING_RATE = 0.001
PATIENCE = 8

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("CNN 自编码器训练程序")
print("=" * 60)
print("使用设备：", device)

if device.type == "cuda":
    print("GPU 名称：", torch.cuda.get_device_name(0))


# =====================================================
# 2. 读取已经准备好的数据
# =====================================================

data_path = Path("data/processed/skab_windows.npz")

if not data_path.exists():
    raise FileNotFoundError(f"找不到数据文件：{data_path}")

with np.load(data_path, allow_pickle=False) as data:
    X_train = data["X_train"].astype(np.float32)
    X_val = data["X_val"].astype(np.float32)

print("\n数据读取完成")
print("训练集形状：", X_train.shape)
print("验证集形状：", X_val.shape)

# 原始数据形状为：
# (样本数量, 时间长度, 特征数量)
#
# Conv1d 要求：
# (样本数量, 特征数量, 时间长度)

X_train = np.transpose(X_train, (0, 2, 1))
X_val = np.transpose(X_val, (0, 2, 1))

print("转换后训练集形状：", X_train.shape)
print("转换后验证集形状：", X_val.shape)


# =====================================================
# 3. 创建 DataLoader
# =====================================================

train_tensor = torch.from_numpy(X_train)
val_tensor = torch.from_numpy(X_val)

train_dataset = TensorDataset(train_tensor)
val_dataset = TensorDataset(val_tensor)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# =====================================================
# 4. 定义 CNN 自编码器
# =====================================================

class CNNAutoencoder(nn.Module):

    def __init__(self, input_features=8):
        super().__init__()

        # 编码器：提取时间序列特征
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

        # 解码器：还原原始时间序列
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

        # 将时间长度恢复到原始的 30
        decoded = nn.functional.interpolate(
            encoded,
            size=x.shape[-1],
            mode="linear",
            align_corners=False
        )

        reconstructed = self.decoder(decoded)

        return reconstructed


model = CNNAutoencoder(input_features=8).to(device)

print("\n模型创建成功")
print(model)


# =====================================================
# 5. 定义损失函数和优化器
# =====================================================

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# =====================================================
# 6. 创建训练和验证函数
# =====================================================

def train_one_epoch():

    model.train()
    total_loss = 0.0

    for (batch,) in train_loader:

        batch = batch.to(device)

        optimizer.zero_grad()

        output = model(batch)

        loss = criterion(output, batch)

        loss.backward()

        optimizer.step()

        total_loss += loss.item() * batch.size(0)

    return total_loss / len(train_dataset)


def validate():

    model.eval()
    total_loss = 0.0

    with torch.no_grad():

        for (batch,) in val_loader:

            batch = batch.to(device)

            output = model(batch)

            loss = criterion(output, batch)

            total_loss += loss.item() * batch.size(0)

    return total_loss / len(val_dataset)


# =====================================================
# 7. 正式训练
# =====================================================

checkpoint_dir = Path("checkpoints")
metrics_dir = Path("results/metrics")

checkpoint_dir.mkdir(parents=True, exist_ok=True)
metrics_dir.mkdir(parents=True, exist_ok=True)

best_val_loss = float("inf")
wait = 0
history = []

best_model_path = checkpoint_dir / "cnn_autoencoder_best.pth"

print("\n开始训练")
print("=" * 60)

for epoch in range(1, EPOCHS + 1):

    train_loss = train_one_epoch()
    val_loss = validate()

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss
    })

    print(
        f"Epoch [{epoch:02d}/{EPOCHS}] "
        f"Train Loss: {train_loss:.6f} "
        f"Val Loss: {val_loss:.6f}"
    )

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        wait = 0

        torch.save(
            model.state_dict(),
            best_model_path
        )

        print("  -> 保存当前最佳模型")

    else:

        wait += 1

    if wait >= PATIENCE:

        print("\n验证集损失连续多轮没有改善，提前停止训练。")
        break


# =====================================================
# 8. 保存训练记录
# =====================================================

history_path = metrics_dir / "training_history.csv"

pd.DataFrame(history).to_csv(
    history_path,
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 60)
print("训练结束")
print("=" * 60)

print("最佳验证损失：", round(best_val_loss, 6))
print("最佳模型保存位置：", best_model_path)
print("训练记录保存位置：", history_path)

print("\n训练记录：")
print(pd.DataFrame(history).to_string(index=False))

if device.type == "cuda":
    print("\nGPU 最大显存占用：",
          round(torch.cuda.max_memory_allocated() / 1024**3, 3),
          "GB")

print("\n全部完成。")
