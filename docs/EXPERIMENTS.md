# 实验记录

本文件由 `results/` 下的结果文件生成，记录课程设计全部实验的配置与指标，供复现与查证。

## 1. 环境与协议

- Windows 11 / Python 3.15.2 / PyTorch 2.14.1+cu132 / RTX 5070 Ti Laptop
- 逐文件协议：每个 CSV 前 400 行训练、其余测试；标准化统计量仅由该文件训练段拟合；滑窗步长 1、不跨文件；窗口标签取末端点
- 阈值 `UCL = Q_q(r_train) × 4/3`；报警为连续 k 窗超阈
- 指标与 SKAB 官方 `core/metrics.py` 一致：`FAR = FP/(FP+TN)`、`MAR = FN/(FN+TP)`、`F1 = TP/(TP+(FN+FP)/2)`

## 2. 主实验（results/protocol_v2）

配置：窗长 30，epochs 30，batch 64，lr 0.001，q 0.99，连续 1，seed 42，耗时 11.7 s

| precision | recall | f1 | far | mar | macro_f1 | TP | FP | FN | TN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.7301 | 0.8401 | 0.7812 | 0.3928 | 0.1599 | 0.7491 | 10704 | 3957 | 2038 | 6116 |

逐文件 F1：最好 valve1/15.csv = 0.9467，最差 other/13.csv = 0.0075，中位数 0.8026

## 3. 后处理扫描（results/ablation/sweep.csv，共 48 组，按 F1 排序前 10）

| window | residual | quantile | consecutive | precision | recall | f1 | far | mar | macro_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 30 | mse | 0.9990 | 5 | 0.7309 | 0.8714 | 0.7950 | 0.4059 | 0.1286 | 0.7615 |
| 30 | mse | 0.9950 | 5 | 0.7254 | 0.8755 | 0.7934 | 0.4192 | 0.1245 | 0.7605 |
| 30 | mse | 0.9900 | 5 | 0.7210 | 0.8817 | 0.7933 | 0.4316 | 0.1183 | 0.7615 |
| 30 | mse | 0.9990 | 1 | 0.7134 | 0.8910 | 0.7924 | 0.4528 | 0.1090 | 0.7617 |
| 30 | mse | 0.9950 | 1 | 0.7079 | 0.8947 | 0.7904 | 0.4671 | 0.1053 | 0.7607 |
| 30 | mse | 0.9900 | 20 | 0.7416 | 0.8448 | 0.7899 | 0.3723 | 0.1552 | 0.7567 |
| 30 | mse | 0.9900 | 1 | 0.7027 | 0.8989 | 0.7888 | 0.4812 | 0.1011 | 0.7599 |
| 60 | mae_sum | 0.9990 | 5 | 0.7095 | 0.8811 | 0.7860 | 0.5050 | 0.1189 | 0.7485 |
| 60 | mae_sum | 0.9990 | 1 | 0.6960 | 0.9010 | 0.7853 | 0.5508 | 0.0990 | 0.7499 |
| 60 | mae_sum | 0.9950 | 5 | 0.7027 | 0.8880 | 0.7846 | 0.5257 | 0.1120 | 0.7474 |

## 4. 留出验证段定阈值（results/ablation/holdout_threshold.csv，前 5 行）

| window | threshold_from | quantile | consecutive | precision | recall | f1 | far | mar | macro_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 30 | val | 0.9500 | 1 | 0.7354 | 0.8611 | 0.7933 | 0.3919 | 0.1389 | 0.7694 |
| 30 | val | 0.9900 | 1 | 0.7432 | 0.8485 | 0.7924 | 0.3708 | 0.1515 | 0.7640 |
| 30 | val | 0.9000 | 1 | 0.7272 | 0.8700 | 0.7922 | 0.4129 | 0.1300 | 0.7697 |
| 30 | val | 0.9500 | 5 | 0.7492 | 0.8358 | 0.7901 | 0.3540 | 0.1642 | 0.7608 |
| 30 | val | 0.9000 | 5 | 0.7429 | 0.8430 | 0.7898 | 0.3690 | 0.1570 | 0.7623 |

## 5. 基线（results/baselines/baseline_metrics.csv）

| model | precision | recall | f1 | far | mar | macro_f1 |
| --- | --- | --- | --- | --- | --- | --- |
| PCA | 0.7564 | 0.8750 | 0.8114 | 0.3565 | 0.1250 | 0.7791 |
| ConvAE | 0.7003 | 0.8989 | 0.7873 | 0.4865 | 0.1011 | 0.7596 |
| LSTMAE | 0.6306 | 0.9571 | 0.7603 | 0.7092 | 0.0429 | 0.7422 |
| IsolationForest | 0.9933 | 0.0232 | 0.0452 | 0.0002 | 0.9768 | 0.0249 |

## 6. 通道聚合（results/ablation/channelwise.csv）

| rule | precision | recall | f1 | far | mar | macro_f1 |
| --- | --- | --- | --- | --- | --- | --- |
| sum | 0.7633 | 0.8106 | 0.7863 | 0.3180 | 0.1894 | 0.7488 |
| top2 | 0.7809 | 0.7224 | 0.7505 | 0.2564 | 0.2776 | 0.6987 |
| max | 0.8022 | 0.6978 | 0.7464 | 0.2176 | 0.3022 | 0.6808 |

## 7. 自适应阈值（results/ablation/adaptive_threshold.csv）

滚动 W 窗分位数在线更新阈值，`shift(1)` 保证只用历史窗口。

| mode | width | quantile | precision | recall | f1 | far | mar | macro_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rolling | 200 | 0.9900 | 0.5616 | 0.0089 | 0.0176 | 0.0088 | 0.9911 | 0.0175 |
| rolling | 400 | 0.9900 | 0.5600 | 0.0088 | 0.0173 | 0.0087 | 0.9912 | 0.0172 |
| rolling | 100 | 0.9900 | 0.4591 | 0.0057 | 0.0113 | 0.0085 | 0.9943 | 0.0114 |
| rolling | 200 | 0.9950 | 0.4552 | 0.0052 | 0.0102 | 0.0078 | 0.9948 | 0.0104 |
| rolling | 400 | 0.9950 | 0.4476 | 0.0050 | 0.0099 | 0.0078 | 0.9950 | 0.0102 |
| rolling | 100 | 0.9950 | 0.3636 | 0.0035 | 0.0068 | 0.0076 | 0.9965 | 0.0070 |

结论：持续型故障会污染滚动窗口内的分位数（自遮蔽），阈值随故障上移，MAR 达 0.99，不可用。

## 9. 基线（含 PCA-T²，results/baselines/baseline_metrics.csv）

| model | precision | recall | f1 | far | mar | macro_f1 |
| --- | --- | --- | --- | --- | --- | --- |
| PCA | 0.7564 | 0.8750 | 0.8114 | 0.3565 | 0.1250 | 0.7791 |
| ConvAE | 0.7003 | 0.8989 | 0.7873 | 0.4865 | 0.1011 | 0.7596 |
| LSTMAE | 0.6306 | 0.9571 | 0.7603 | 0.7092 | 0.0429 | 0.7422 |
| PCAT2 | 0.6535 | 0.6366 | 0.6449 | 0.4269 | 0.3634 | 0.5712 |
| IsolationForest | 0.9933 | 0.0232 | 0.0452 | 0.0002 | 0.9768 | 0.0249 |

## 10. 检测延迟（results/ablation/detection_delay.csv）

以故障起点到首次报警的采样点间隔计。

| config | hit | median_delay | iqr | max_delay |
| --- | --- | --- | --- | --- |
| official(w60,q0.999,k60) | 32/34 | 76 | 21–113 | 316 |
| v2(w30,q0.99,k1) | 34/34 | 14 | 0–51 | 80 |

结论：延迟主要由连续 k 规则决定（k=60 至少带来 60 步下限），与模型结构无关。

## 11. 图表与报告

```bash
python src/make_figures.py    # results/figures/*.png
python src/build_report.py    # report/课程设计报告.docx
```

## 受控实验（审稿补充）

命令：`python src/exp_controlled.py --stage all`，输出到 `results/controlled/`。

### C1 阈值规则受控（同一逐文件模型，只换阈值口径）

| 规则 | P | R | F1 | FAR | MAR |
| --- | --- | --- | --- | --- | --- |
| 旧协议阈值规则 q95×1 | 0.5929 | 0.9745 | 0.7372 | 0.8464 | 0.0255 |
| q95×4/3 | 0.6767 | 0.9166 | 0.7785 | 0.5541 | 0.0834 |
| 现行默认 q99×4/3 | 0.7027 | 0.8989 | 0.7888 | 0.4812 | 0.1011 |

### 文件级二折选参（配置只在选择半上选，评估半不参与）

| 折 | 选中配置 | 选择折 macro-F1 | 评估折 P | R | F1 | FAR | MAR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | w30/mse/q0.999/k5 | 0.8233 | 0.6928 | 0.8605 | 0.7676 | 0.4538 | 0.1395 |
| 2 | w30/mse/q0.99/k1 | 0.7057 | 0.7388 | 0.9021 | 0.8123 | 0.4281 | 0.0979 |

两折合并（34 文件点级）：P 0.7168 / R 0.8824 / F1 0.7910 / FAR 0.4411 / MAR 0.1176。

### other/2.csv 训练段污染敏感性

| 设置 | 训练窗口 | 含异常窗口 | 阈值 | P | R | F1 | FAR | MAR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 全部训练点（现行） | 371 | 296 | 0.5365 | 0.0833 | 0.2034 | 0.1182 | 0.4521 | 0.7966 |
| 丢弃含异常的窗口 | 75 | 296 | 0.8403 | 0.1681 | 1.0000 | 0.2878 | 1.0000 | 0.0000 |

### 全局标准化 + 逐文件训练（q95×1）

P 0.6908 / R 0.7724 / F1 0.7293 / FAR 0.4374 / MAR 0.2276 / macro-F1 0.6672。

### 官方单通道实现复刻（filters=1）

| 窗长 | q | k | P | R | F1 | FAR | MAR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 30 | 0.99 | 1 | 0.7159 | 0.7808 | 0.7469 | 0.3919 | 0.2192 |
| 60 | 0.999 | 60 | 0.7070 | 0.6745 | 0.6903 | 0.3913 | 0.3255 |

### 复现性

- `python src/run_experiments.py --stage sweep` 后在干净克隆上 `git diff --stat` 为空：sweep 可逐字节复现。
- `python src/make_figures.py` 重生成的 6 张图与原图逐像素一致（maxdiff = 0）。
- 需要注意：同一 seed 重训仍会因 GPU 非确定性带来 0.002–0.02 的 F1 波动（主配置最小、通道聚合最大）。

## 单因素分解（results/controlled/decomp.csv）

命令：`python src/exp_controlled.py --stage decomp`。旧训练超参 = batch 256 / 50 轮 / patience 8，新训练超参 = batch 32 / 60 轮 / patience 5；全局标准化统计量取自 `data/processed/normalization.npz`。

| 步骤 | P | R | F1 | FAR | MAR |
| --- | --- | --- | --- | --- | --- |
| ① 旧方案（全局数据/全局标准化/旧超参/q95×1） | 0.3791 | 0.9723 | 0.5455 | 0.8912 | 0.0277 |
| A 逐文件数据（全局标准化、旧超参、q95×1） | 0.6701 | 0.5974 | 0.6316 | 0.3721 | 0.4026 |
| B 再加逐文件标准化（旧超参、q95×1） | 0.6285 | 0.9633 | 0.7607 | 0.7204 | 0.0367 |
| E 阈值分位改 q99（系数仍 ×1） | 0.6146 | 0.9617 | 0.7499 | 0.7628 | 0.0383 |
| ④ 换现行训练超参（阈值仍 q95×1） | 0.5929 | 0.9745 | 0.7372 | 0.8464 | 0.0255 |
| ⑥ 松弛系数 ×1→4/3（现行默认） | 0.7027 | 0.8989 | 0.7888 | 0.4812 | 0.1011 |

注：各步收益不可加——标准化在旧超参下 +0.129、在现行超参下 +0.008；换超参在全局标准化下 +0.098、在逐文件标准化下 −0.023。

## 报告派生统计（src/exp_report_stats.py）

- PCA：240 维输入取 8 个主成分，逐文件累计解释方差平均 38.3%（28.3%–52.6%）。
- 现行默认配置的误报结构：FP 4847 / 正常窗口 10073（FAR 0.4812），257 个连续段，平均 18.9 窗、中位 2 窗、最长 174 窗，前 10 段占 35.5%。
- 剔除 other/2.csv 后：F1 0.7940、FAR 0.4820、MAR 0.0969（33 个文件）。
- 二折划分：评估折 A = 17 个文件（other/1.csv, other/11.csv, other/13.csv …），评估折 B = 17 个文件（other/10.csv, other/12.csv, other/14.csv …）；完整清单见 `results/controlled/fold_files.json`。

## 表 3 各行配置审记（2026-11-04 核对）

| 行 | 模型 | 训练数据 | 验证集 | 标准化 | batch / epochs / patience | 窗口 | q | ×系数 | seed | 结果文件 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ① 旧方案 | CNN-AE | anomaly-free 前 80% | anomaly-free 后 20% | 全局（anomaly-free 训练段） | 256 / 50 / 8 | 30 | 0.95 | 1 | 42 | results/metrics/evaluation_metrics.csv |
| ② A | CNN-AE | 各文件前 400 点 | 该文件训练窗口随机 10% | 全局（同上 npz） | 256 / 50 / 8 | 30 | 0.95 | 1 | 42 | results/controlled/decomp.csv |
| ③ B | CNN-AE | 各文件前 400 点 | 该文件训练窗口随机 10% | 逐文件 StandardScaler | 256 / 50 / 8 | 30 | 0.95 | 1 | 42 | results/controlled/decomp.csv |
| ④ 训练超参数组合对照 | CNN-AE | 各文件前 400 点 | 该文件训练窗口随机 10% | 逐文件 StandardScaler | 32 / 60 / 5 | 30 | 0.95 | 1 | 42 | results/residuals/residuals_w30.csv → threshold_rule.csv |
| ⑤ E | CNN-AE | 同上（复用 ④ 残差） | 同上 | 逐文件 StandardScaler | 32 / 60 / 5 | 30 | 0.99 | 1 | 42 | results/controlled/decomp.csv |
| ⑥ | CNN-AE | 同上（复用 ④ 残差） | 同上 | 逐文件 StandardScaler | 32 / 60 / 5 | 30 | 0.99 | 4/3 | 42 | results/controlled/threshold_rule.csv |

结论：
- ③→④ 是**训练超参数组合对照**：batch、epochs、patience 三者同时变化，其余条件一一对应，F1 0.7607 → 0.7372（−0.023）。据此不需要重跑，但报告不对单个超参的作用作独立归因。
- ②→③ 也是严格单因素（只换标准化统计量），F1 0.6316 → 0.7607（+0.129）。
- ①→② 同时改了训练数据与验证集来源（anomaly-free 20% → 目标文件训练窗口随机 10%），两者同时变化，故不能把该步差异全部归因于训练数据；已作为局限写明。
- 「全局标准化 × 现行超参」这一格**没有跑过**；`results/controlled/global_scaler_summary.json`（F1 0.7293）用的是 batch 64 / 30 轮，**不属于**本表，故不作为交互证据。

## 图表数据来源（与报告图注一致）

| 图 | 生成脚本 | 数据来源 |
| --- | --- | --- |
| 图 1 逐文件 F1 | src/make_figures.py: fig_per_file | results/baselines/per_file_ConvAE.csv、per_file_PCA.csv |
| 图 2 三类失败模式 | src/make_figures.py: fig_failure | results/protocol_v2/predictions.csv、per_file_metrics.csv（阈值列） |
| 图 3 各文件平均 z 值 | src/make_figures.py: fig_cross_condition | 原始 CSV + data/processed/normalization.npz（anomaly-free 统计量） |

## 已知局限（与报告 §6.5 一致）

- 文件级二折只有一组按文件名奇偶的固定划分，0.7910 与事后最优的 0.005 差距仅对该划分成立，不代表一般意义上的选优偏差很小；未做重复划分或 K 折。
- PCA 主成分数量未做消融；8 个是该配置的实测取值，38.3% 的累计解释方差只描述该配置，不构成「充分或最优」的依据。
- ①→② 的训练数据与验证集来源同时变化，无法分离。

## 第三轮审计补充（表 4 求和行、q 消融、重跑口径）

**表 4 的「8 通道 MAE 求和（独立训练）」**
- 代码：`src/exp_channelwise.py`，每文件 `train_autoencoder(w_tr, 30, epochs=60, batch_size=32, lr=1e-3, seed=42)`，逐文件 `StandardScaler`；残差为 `|x - x_hat|` 在时间维取均值后按通道求和（L1 求和）；阈值 `np.quantile(train, 0.99) * 4/3`，k=1。结果 `results/ablation/channelwise.csv`。
- 与表 2 默认行（MSE 均值，来自 `results/residuals/residuals_w30.csv`）不是同一口径：**逐通道 MSE 求和 = 8 × MSE 均值**，阈值同比例放大，判定完全相同（恒等式）；而表中求和行用的是 L1，故差异来自 L1/L2 对尾部误差的敏感度，加上另一次独立训练。
- 因此报告已把该行标注为「独立训练」，并删除「与表 2 默认行同配置、差异来自重训波动」的旧说法。

**q 的匹配消融（`results/ablation/sweep.csv`，w30 / mse / k=1 / ×4/3 固定）**

| q | F1 | FAR | MAR |
| --- | --- | --- | --- |
| 0.99 | 0.7888 | 0.4812 | 0.1011 |
| 0.995 | 0.7904 | 0.4671 | 0.1053 |
| 0.999 | 0.7924 | 0.4528 | 0.1090 |

结论：q 0.99→0.999 只让 FAR 降 0.028、MAR 升 0.008；此前「FAR 减半 / MAR 翻倍」的说法来自不同 k 的跨配置比较，已删除。

**重跑波动**
- 没有系统的多种子实验；同一 seed 重跑同一脚本时观察到 F1 约 0.01 量级的浮动，这些重复未逐次留档。
- 曾把 `results/protocol_v2/summary.json`（30 轮 / batch 64）与 sweep 行（60 轮 / batch 32）当成「同一配置两次独立训练」，实为两组不同超参，该说法已从报告中删除。

## 多种子重复（seed 42 / 43 / 44）

配置完全固定：CNN-AE、batch 32、60 轮、patience 5、window 30、逐文件 StandardScaler、前 400 点训练、
`q99 × 4/3`、k=1；**验证集划分固定为 seed 42 的 `randperm`**，每次运行只改初始化与批次顺序
（`train_autoencoder(..., seed=s, split_seed=42)`）。脚本 `src/run_repeats.py`，结果目录：

- `results/repeats/seed_42/`、`seed_43/`、`seed_44/`：各含 `config.json`、`training_history.csv`、
  `test_predictions.csv`、`evaluation_metrics.csv`、`per_file_metrics.csv`、`checkpoints/`。
- `results/repeats/per_seed_metrics.csv`、`repeat_summary.csv`（n=3，ddof=1）、`per_file_seed_matrix.csv`。

| 指标 | mean | std | min | max |
| --- | --- | --- | --- | --- |
| precision | 0.7117 | 0.0044 | 0.7070 | 0.7158 |
| recall | 0.8908 | 0.0127 | 0.8778 | 0.9032 |
| f1 | 0.7912 | 0.0024 | 0.7886 | 0.7931 |
| far | 0.4565 | 0.0163 | 0.4409 | 0.4734 |
| mar | 0.1092 | 0.0127 | 0.0968 | 0.1222 |
| macro_f1 | 0.7655 | 0.0029 | 0.7630 | 0.7687 |

三次 pooled F1 分别为 0.7931 / 0.7886 / 0.7919。逐文件跨种子方差最大的文件见
`per_file_seed_matrix.csv` 的 `f1_std` 列（other/2.csv 0.079、valve2/0.csv 0.064）。

说明：seed 42 这次重跑得 0.7931，与主实验残差缓存行 0.7888 差 0.0043，属同配置重跑离散；
样本只有 3 次，均值 ± 标准差仅作量级参考，报告中未据此挑种子、也未重选阈值。

## 验证集划分对照（随机 10% vs 无重叠 holdout）

原方案在 371 个训练窗口里随机抽 37 个作验证（`skab_protocol.py:167`），因步长 1、窗长 30，
**1258 个验证窗口 100% 与训练窗口共享原始时间点**，每个平均与 50.6 个训练窗口重叠（详见
`docs/AUDIT_VALIDATION_OVERLAP.md`）。无重叠方案按时间顺序取末尾 37 个窗口作验证，并在其前留
29 个窗口隔离带：训练窗口 0–304（覆盖 0–333 点）、隔离带 305–333、验证窗口 334–370（覆盖 334–399 点）。

同一配置（CNN-AE、batch 32、60 轮、patience 5、w30、逐文件 StandardScaler、q99 × 4/3、k=1）
在 seed 42/43/44 下两套划分的结果：

| 指标 | random（原） | holdout（无重叠） | Δ |
| --- | --- | --- | --- |
| Precision | 0.7117 ± 0.0044 | 0.7136 ± 0.0071 | +0.0019 |
| Recall | 0.8908 ± 0.0127 | 0.8791 ± 0.0146 | −0.0116 |
| F1 | 0.7912 ± 0.0024 | 0.7877 ± 0.0052 | −0.0035 |
| FAR | 0.4565 ± 0.0163 | 0.4465 ± 0.0204 | −0.0100 |
| MAR | 0.1092 ± 0.0127 | 0.1209 ± 0.0146 | +0.0116 |
| macro-F1 | 0.7655 ± 0.0029 | 0.7623 ± 0.0077 | −0.0032 |

- 早停验证损失：holdout 的逐文件 `min(val_loss)`（三种子平均）是 random 的 **1.72 倍**，34 个文件中 33 个更高
  → 原划分的验证损失系统性偏乐观。
- 测试集指标几乎不变：pooled F1 −0.0035，逐文件 |ΔF1| 中位 0.0154、最大 0.1292。
- 种子间方差变大（F1 标准差 0.0024 → 0.0052），因为原方案的验证窗口固定、且与训练重叠。

产物：`results/repeats_holdout/`（seed_42/43/44 + repeat_summary.csv + per_seed_metrics.csv +
per_file_seed_matrix.csv）、`comparison.csv`、`comparison_summary.csv`、`val_loss_comparison.csv`；
运行与对比脚本：`src/run_repeats.py --val-scheme holdout --out results/repeats_holdout`、
`src/compare_validation_schemes.py`。旧结果 `results/repeats/` 未被覆盖。


## 验证集划分三方案对照（random / random_matched / holdout）

- 方案定义与代码位置见 `docs/AUDIT_VALIDATION_OVERLAP.md` 第七节；`random_matched` 由 `src/skab_protocol.py` 的 `train_autoencoder(val_scheme="random_matched")` 实现，`src/run_repeats.py --val-scheme random_matched` 调用。
- 结果目录：`results/repeats_matched/`（种子 42/43/44，配置见各 `seed_XX/config.json`）。
- 汇总：`results/validation_schemes/{summary.csv, per_seed_metrics.csv, per_file_metrics.csv, best_epoch.csv}`，由 `src/compare_validation_schemes.py` 生成。
- 结论：random F1 0.7912 → matched 0.7896（训练窗 334→305，−0.0016）→ holdout 0.7877（同 305 窗，去重叠并改成时间顺序，−0.0019）；验证损失中位 1.05 倍 → 1.50 倍，说明损失抬升主要来自验证段的时间位置，重叠的解释力有限但未被单独识别。

## 阈值残差集合的口径（补充）

- 代码位置：`src/run_repeats.py:88-90`。`w_tr = make_windows(scaler.transform(train_df[FEATURES].values), WINDOW, STRIDE)` 取的是每个文件前 400 点的**全部 371 个窗口**，随后 `e_tr = reconstruction_error(model, w_tr)`，`thr = threshold_from_train(e_tr, 0.99) * 4/3`。
- 因此阈值使用的残差集合 = 模型训练窗口 + 早停验证窗口 + 划分方案丢弃的窗口；**不含** 400 点之后的测试窗口。测试段的 `e_te` 只用于判定与指标计算。
- 由此推论：holdout 只保证「模型训练窗口」与「早停验证窗口」在原始采样点上不重叠，阈值校准与 StandardScaler 仍基于前 400 点全部数据，不构成完全独立的标定与预处理流程；这与测试数据无关，不构成测试数据泄漏。

## 表 5 最小验证损失的统计口径

- 每个（种子, 文件）在该文件的 60 轮训练曲线上取最小 `val_loss`（即早停所依据的那条曲线的最小值）。
- 先在 34 个文件上等权平均，再对 3 个种子取均值；表中 ± 为跨种子的样本标准差（n = 3，ddof = 1）。
- 三种子合并后的跨文件标准差约为 0.03 / 0.03 / 0.15（random / matched / holdout），远大于跨种子离散，因此表中只用跨种子 sd 衡量种子稳定性，不把跨文件离散当作误差棒。
- 数值：random 0.189 ± 0.006、random_matched 0.198 ± 0.005、holdout 0.322 ± 0.004。

## §5.3 定阈值实验的 StandardScaler 范围（代码证据）

源码：`src/run_experiments.py` 的 `stage_holdout()`（第 208–256 行），关键片段：

```python
fit_df  = df.iloc[:FIT_END]        # 0–320 行
val_df  = df.iloc[FIT_END:VAL_END] # 320–400 行
test_df = df.iloc[VAL_END:]        # 400 行之后

scaler  = StandardScaler().fit(fit_df[FEATURES].values)   # 只用 0–320 行估计 mean/std
x_fit   = scaler.transform(fit_df[FEATURES].values)
x_val   = scaler.transform(val_df[FEATURES].values)
x_test  = scaler.transform(test_df[FEATURES].values)

win_fit  = make_windows(x_fit, window)   # 291 个窗口
win_val  = make_windows(x_val, window)   # 51 个窗口
win_test = make_windows(x_test, window)

model = train_autoencoder(win_fit, window, epochs=HOLDOUT_EPOCHS, batch_size=BATCH,
                          lr=LR, seed=SEED, device=device, val_split=0.0, patience=5)
err_val = reconstruction_error(model, win_val, mode="mse")
thr = float(np.quantile(err_val, q) * 4 / 3)   # 阈值只用 320–400 段的 51 个残差
```

逐条回答：

| 问题 | 结论 |
| --- | --- |
| `StandardScaler.fit()` 用了哪些原始行 | 只用 0–320 行（`fit_df`，320 行） |
| 320–400 行是否参与均值/标准差 | **否** |
| 320–400 行是否只参与阈值计算 | **是**：不参与参数更新，`err_val` 仅用于 `np.quantile` 定阈值 |
| 测试段是否进入阈值计算 | **否**：`err_test` 只用于逐窗判定与指标 |

窗口数由 `results/residuals/residuals_holdout.csv` 复核：fit 291 窗 / val 51 窗 / test 随文件长度变化。报告中该段已按此改写（原「标定残差由模型在训练窗口上算出」的说法不再使用）。
