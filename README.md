# SKAB 工业多变量时序异常检测

深度学习课程设计（2026 秋）· 复现研究

在 SKAB v0.9 阀门设备数据集上研究**无监督重构式异常检测**：为什么一个 Conv-AE 会给出 89% 的误报率，
把数据划分换成官方逐文件协议后能达到什么水平，以及阈值与后处理对指标的影响有多大。

## 结论速览

| 方案 | F1 | FAR | MAR |
| --- | --- | --- | --- |
| 旧全局协议（anomaly-free 训练 + 全局标准化） | 0.5455 | 0.8912 | 0.0277 |
| 逐文件协议 · Conv-AE（默认 30 窗 / q=0.99） | 0.7888 | 0.4812 | 0.1011 |
| 逐文件协议 · 官方 notebook 配置复现 | 0.7263 | 0.4278 | 0.2554 |
| 逐文件协议 · PCA 重构（对照） | **0.8114** | 0.3565 | 0.1250 |
| 逐文件协议 · LSTM-AE（对照） | 0.7603 | 0.7092 | 0.0429 |
| 逐文件协议 · PCA-T² 统计量（对照） | 0.6449 | 0.4269 | 0.3634 |
| 逐文件协议 · Isolation Forest（对照） | 0.0452 | 0.0002 | 0.9768 |

四点主要发现：

1. **旧方案的失效不是模型能力问题，而是「只学了一条别的工况」。** `anomaly-free.csv` 是单一稳态工况，
   Accelerometer1RMS 的标准差只有 0.005，用它做标准化后，其他文件的工况差异被放大成 −40σ ~ +27σ 的平移。
   但受控分解显示：切换到目标文件自身的训练与验证数据后，F1 由 0.5455 升至 0.6316；随后采用逐文件标准化，
   F1 进一步升至 0.7607。两步均带来明显改善，但该结果仅代表本实验顺序下的对照，不能推断变量间的交互或普遍因果关系。
2. **阈值是最大的杠杆。** 48 组后处理配置下 F1 只在 0.669–0.795 之间移动，FAR 从 0.225 变化至 0.889，说明不同阈值与后处理组合会带来较大的误报率差异；
   「用哪段数据估阈值」的影响大于「换模型」的影响。
3. **线性 PCA 反超 Conv-AE。** 每个文件前 400 点可生成 371 个标定窗口（窗长 30、步长 1，相邻窗口高度重叠，不能当作独立样本），
   其中随机划分方案用 334 个窗口做参数训练，输入为 240 维展平特征。故障表现为少数通道的低频电平偏移，
   线性子空间已足够；同一 PCA 的 T² 统计量只有 F1 = 0.6449，说明统计量的选择比模型本身更关键。
4. **延迟由后处理规则而非模型决定。** k = 1 时 34 个文件全部在故障起点后报警，中位延迟 14.5 个采样点；
   k = 60 时只有 32/34 命中、中位延迟 75.5 点，但 FAR 从 0.4812 降到 0.4278。

以上数字都建立在**直推式自校准**（用目标文件前 400 点训练）之上，不检验跨工况泛化；
报告 §3.1 与 §6.1 对这个边界有专门说明。本仓库不含课程设计报告文件。

## 目录结构

```
src/                    实验代码
  skab_protocol.py      数据读取、切分、滑窗、Conv-AE、阈值与指标（唯一数据入口）
  skab_paths.py         定位数据目录（容忍不同解压层级）
  run_protocol_v2.py    按 preset 跑完整协议（v2 / official 两套配置）
  run_experiments.py    三阶段：缓存残差 → 48 组后处理扫描 → 留出验证段定阈值
  exp_channelwise.py    通道聚合对照（sum / top2 / max）
  exp_detection_delay.py 检测延迟（故障起点到首次报警的间隔）
  exp_controlled.py     受控实验（阈值规则 / 二折选参 / 训练段污染 / 全局标准化 / 单通道复刻）
  run_baselines.py      PCA、PCA-T²、Isolation Forest、LSTM-AE 基线
  run_repeats.py        三种验证划分 × 3 个随机种子的重复实验
  compare_validation_schemes.py  三方案汇总与逐文件对照
  exp_report_stats.py   报告用的辅助统计（PCA 解释方差、误报段结构等）
  build_result_tables.py 汇总基线与自适应阈值实验
  make_figures.py       从结果文件生成全部图表
  build_report.py       由结果文件生成报告 DOCX（需 python-docx）
  verify_report.py      把报告里的数字逐项比对回 results/（自检，需 python-docx）
  —— 以下为旧方案（全局标准化 + 单文件划分）脚本，仅用于复现报告 §1 的 0.5455 结果 ——
  prepare_data.py / train_autoencoder.py / evaluate_autoencoder.py
  check_dataset.py / compare_columns.py / inspect_*.py
results/                实验输出（csv / json / figures）
docs/                   EXPERIMENTS.md（实验记录）、AUDIT_VALIDATION_OVERLAP.md（验证集重叠审计）
```

## 环境

Windows 11 · Python 3.15 · PyTorch 2.14.1+cu132 · RTX 5070 Ti Laptop

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 复现

```bash
# 1. 跑主实验（逐文件协议 + 残差缓存 + 48 组扫描 + 留出段阈值）
python src/run_experiments.py --stage all

# 2. 基线与通道聚合
python src/run_baselines.py --models pca,iforest,lstm,conv,pcat2
python src/exp_channelwise.py
python src/exp_detection_delay.py
python src/exp_controlled.py --stage all
python src/build_result_tables.py

# 3. 三种验证集划分 × 3 个随机种子（约 36 s/种子）
python src/run_repeats.py --val-scheme random          --out results/repeats
python src/run_repeats.py --val-scheme random_matched  --out results/repeats_matched
python src/run_repeats.py --val-scheme holdout         --out results/repeats_holdout
python src/compare_validation_schemes.py

# 4. 图表与报告
python src/make_figures.py
python src/build_report.py
```

`--val-scheme` 的默认输出目录只区分 random（`results/repeats`）与其他（`results/repeats_holdout`），
因此 random_matched 需显式传 `--out results/repeats_matched`（与仓库中已有结果一致）。

单次全量训练约 1.5 min；由于残差只算一次并缓存为 CSV，48 组后处理配置的扫描不足 5 s。

报告里的每个数字都在生成时从 `results/` 现读，不手抄。改过代码或重跑实验后，可用

```bash
python src/verify_report.py     # 需带 python-docx 的解释器
```

把 docx 里的表格与正文关键值逐项比对回原始 CSV/JSON（当前 184 项，全部通过），
数字不一致会以非零退出码报出具体单元格。

## 数据

SKAB v0.9（[Kaggle: SKAB](https://www.kaggle.com/datasets/katser/skoltech-anomaly-benchmark-skab)，
官方实现：[waico/SKAB](https://github.com/waico/SKAB)），
34 个带标签文件（valve1 ×16、valve2 ×4、other ×14）+ `anomaly-free.csv`。
每个文件的分号分隔 CSV 含 8 个传感器通道与 `anomaly`、`changepoint` 两个标签列；标签不作为模型输入。

仓库不包含数据（`data/` 已在 `.gitignore` 中）。解压后把 SKAB 的 `data/` 目录放到

```
data/raw/SKAB-master/data/{valve1,valve2,other}/  +  anomaly-free.csv
```

即可直接运行。若解压层级不同（例如 `data/raw/SKAB-master/SKAB/data/`），
`src/skab_paths.py` 的 `resolve_data_dir()` 会在 `data/raw/` 下递归查找第一个同时包含
`valve1/` 或 `other/` 的目录，因此无需改代码；查找失败时会直接报错并提示期望路径。

## 评估口径

与 SKAB 官方 `core/metrics.py` 对齐的点级指标：

- `FAR = FP / (FP + TN)`，`MAR = FN / (FN + TP)`，`F1 = TP / (TP + (FN + FP) / 2)`
- 阈值 `UCL = Q_q(r_cal) × 4/3`。`r_cal` 是每个文件前 400 个标定点生成的全部 371 个窗口的重构残差；
  该集合包含模型训练窗口、早停验证窗口与被划分方案丢弃的窗口，**不包含** 400 点之后的测试数据。
  模型参数更新只使用对应方案指定的训练窗口。
- 报警规则为连续 k 个窗口超阈；汇总指标按窗口数加权，另报 34 个文件的 macro-F1
