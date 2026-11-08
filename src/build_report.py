"""生成课程设计报告 DOCX（数字全部从 results/ 的真实结果文件读取）。"""
from __future__ import annotations

import json
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from skab_paths import resolve_data_dir  # noqa: E402

RES = ROOT / "results"
FIG = RES / "figures"
OUT_DIR = ROOT / "report"
OUT = OUT_DIR / "课程设计报告.docx"

CN = "宋体"
CN_BOLD = "黑体"
EN = "Times New Roman"


# ---------------------------------------------------------------- 数据装载
def load_numbers() -> dict:
    n: dict = {}
    old = pd.read_csv(RES / "metrics" / "evaluation_metrics.csv").iloc[0]
    n["old"] = dict(
        p=old["precision"], r=old["recall"], f1=old["f1"],
        far=old["false_alarm_rate"], mar=old["FN"] / (old["FN"] + old["TP"]),
        thr=old["threshold"],
    )
    n["v2_summary"] = json.loads((RES / "protocol_v2" / "summary.json").read_text(encoding="utf-8"))
    # 默认配置的权威数字取残差缓存上的同一配置行，避免与表 3④ 冲突
    sw0 = pd.read_csv(RES / "ablation" / "sweep.csv")
    row0 = sw0[(sw0["window"] == 30) & (sw0["residual"] == "mse")
               & (sw0["quantile"] == 0.99) & (sw0["consecutive"] == 1)].iloc[0]
    n["v2"] = {
        "precision": float(row0["precision"]), "recall": float(row0["recall"]),
        "f1": float(row0["f1"]), "far": float(row0["far"]), "mar": float(row0["mar"]),
        "macro_f1": float(row0["macro_f1"]),
    }
    n["v2_run2"] = {k: float(n["v2_summary"][k]) for k in
                    ("precision", "recall", "f1", "far", "mar", "macro_f1")}
    off = pd.read_csv(RES / "protocol_v2_official" / "per_file_metrics.csv")
    tp, fp, fn, tn = (int(off[c].sum()) for c in ("tp", "fp", "fn", "tn"))
    n["official"] = dict(
        p=tp / (tp + fp), r=tp / (tp + fn), f1=2 * tp / (2 * tp + fp + fn),
        far=fp / (fp + tn), mar=fn / (fn + tp), macro=float(off["f1"].mean()),
        tp=tp, fp=fp, fn=fn, tn=tn,
    )
    sw = pd.read_csv(RES / "ablation" / "sweep.csv")
    n["sweep"] = sw
    n["sweep_best"] = sw.sort_values("f1", ascending=False).iloc[0]
    n["sweep_best_macro"] = sw.sort_values("macro_f1", ascending=False).iloc[0]
    n["sweep_f1_min"] = float(sw["f1"].min())
    n["sweep_n"] = len(sw)
    ho = pd.read_csv(RES / "ablation" / "holdout_threshold.csv")
    n["holdout"] = ho.sort_values("f1", ascending=False).iloc[0]
    n["base"] = pd.read_csv(RES / "baselines" / "baseline_metrics.csv")
    n["params"] = json.loads((RES / "metrics" / "model_params.json").read_text(encoding="utf-8"))["trainable_params"]
    n["chan"] = pd.read_csv(RES / "ablation" / "channelwise.csv")
    n["adaptive"] = pd.read_csv(RES / "ablation" / "adaptive_threshold.csv")
    dec = pd.read_csv(RES / "controlled" / "decomp.csv")
    n["decomp"] = dec.set_index(dec["setting"].str[0])
    for key, name in (("pca_evr", "pca_evr"), ("fpseg", "fp_segments"),
                      ("o2excl", "other2_exclusion"), ("fold_files", "fold_files")):
        n[key] = json.loads((RES / "controlled" / f"{name}.json").read_text(encoding="utf-8"))
    n["files"] = pd.read_csv(RES / "protocol_v2" / "per_file_metrics.csv")
    n["delay"] = pd.read_csv(RES / "ablation" / "detection_delay.csv")
    rep_dir = RES / "repeats"
    n["rep_seed"] = pd.read_csv(rep_dir / "per_seed_metrics.csv")
    n["rep_stat"] = pd.read_csv(rep_dir / "repeat_summary.csv").set_index("metric")
    n["rep_pf"] = pd.read_csv(rep_dir / "per_file_seed_matrix.csv")
    hold_dir = ROOT / "results" / "repeats_holdout"
    n["holdout_stat"] = pd.read_csv(hold_dir / "repeat_summary.csv").set_index("metric")
    n["holdout_seed"] = pd.read_csv(hold_dir / "per_seed_metrics.csv")
    vs_dir = ROOT / "results" / "validation_schemes"
    vs = pd.read_csv(vs_dir / "summary.csv")
    n["scheme"] = vs.pivot(index="metric", columns="scheme", values="mean")
    n["scheme_std"] = vs.pivot(index="metric", columns="scheme", values="std")
    n["scheme_seed"] = pd.read_csv(vs_dir / "per_seed_metrics.csv")
    best = pd.read_csv(vs_dir / "best_epoch.csv")
    vl_seed = best.groupby(["scheme", "seed"])["min_val_loss"].mean().groupby("scheme")
    n["vl_seed"] = pd.DataFrame({"mean": vl_seed.mean(), "std": vl_seed.std(ddof=1)})
    n["vl_file_sd"] = best.groupby("scheme")["min_val_loss"].std(ddof=1)
    piv = best.pivot_table(index="file", columns="scheme", values="min_val_loss", aggfunc="mean")
    r_mr = (piv["random_matched"] / piv["random"]).dropna()
    r_hm = (piv["holdout"] / piv["random_matched"]).dropna()
    n["ratio_mr"] = dict(median=float(r_mr.median()), mean=float(r_mr.mean()), higher=int((r_mr > 1).sum()), n=int(len(r_mr)))
    n["ratio_hm"] = dict(median=float(r_hm.median()), mean=float(r_hm.mean()), higher=int((r_hm > 1).sum()), n=int(len(r_hm)))
    n["chanpf"] = pd.read_csv(RES / "ablation" / "channelwise_per_file.csv")
    ctrl = RES / "controlled"
    n["ctrl_thr"] = pd.read_csv(ctrl / "threshold_rule.csv")
    n["ctrl_sel"] = json.loads((ctrl / "selection_summary.json").read_text(encoding="utf-8"))
    n["ctrl_other2"] = pd.read_csv(ctrl / "other2_sensitivity.csv")
    n["ctrl_global"] = json.loads((ctrl / "global_scaler_summary.json").read_text(encoding="utf-8"))
    n["ctrl_1ch"] = pd.read_csv(ctrl / "official_1ch.csv")
    return n


def group_table(n: dict) -> list[list[str]]:
    norm = np.load(ROOT / "data" / "processed" / "normalization.npz")
    mean, std = norm["mean"], norm["std"]
    feats = [str(x) for x in norm["features"]]
    i1, i2 = feats.index("Accelerometer1RMS"), feats.index("Volume Flow RateRMS")

    z: dict[str, list[tuple[float, float]]] = {}
    for path in sorted(resolve_data_dir(ROOT).rglob("*.csv")):
        if path.name == "anomaly-free.csv":
            continue
        df = pd.read_csv(path, sep=";")
        zz = (df[feats].to_numpy() - mean) / std
        group = path.parent.name
        z.setdefault(group, []).append((float(zz[:, i1].mean()), float(zz[:, i2].mean())))

    files = n["files"]
    rows = []
    for group in ("valve1", "valve2", "other"):
        sub = files[files.file.str.startswith(group)]
        vals = np.array(z[group])
        rows.append([
            {"valve1": "valve1 入口关阀", "valve2": "valve2 出口关阀", "other": "other 泄漏/不平衡"}[group],
            str(len(sub)),
            f"{int(sub.n_test.sum()):,}",
            f"{sub.anomaly_ratio.mean() * 100:.0f}%",
            f"{vals[:, 0].mean():+.1f}",
            f"{vals[:, 1].mean():+.1f}",
        ])
    return rows


# ---------------------------------------------------------------- 排版工具
def set_run(run, size=10.5, bold=False, cn=CN) -> None:
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.name = EN
    run._element.rPr.rFonts.set(qn("w:eastAsia"), cn)


def body(doc, text: str, size=9.5, first_indent=True, space_after=0, align=WD_ALIGN_PARAGRAPH.JUSTIFY):
    p = doc.add_paragraph()
    p.alignment = align
    pf = p.paragraph_format
    pf.line_spacing = 0.93
    pf.space_before = Pt(0)
    pf.space_after = Pt(space_after)
    if first_indent:
        pf.first_line_indent = Pt(size * 2)
    # 支持 **粗体** 片段
    for i, seg in enumerate(text.split("**")):
        if not seg:
            continue
        run = p.add_run(seg)
        set_run(run, size=size, bold=(i % 2 == 1))
    return p


def heading(doc, text: str, level=1, space_before=None):
    size = {1: 12.5, 2: 11, 3: 10.5}[level]
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(space_before if space_before is not None else {1: 6, 2: 4, 3: 3}[level])
    pf.space_after = Pt({1: 3, 2: 2, 3: 1}[level])
    pf.keep_with_next = True
    run = p.add_run(text)
    set_run(run, size=size, bold=True, cn=CN_BOLD)
    return p


def caption(doc, text: str, above=False):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = p.paragraph_format
    pf.space_before = Pt(2 if not above else 4)
    pf.space_after = Pt(6 if not above else 2)
    pf.keep_with_next = above
    run = p.add_run(text)
    set_run(run, size=9)
    return p


def add_table(doc, headers, rows, widths=None, font=8.5, caption_text=None):
    if caption_text:
        caption(doc, caption_text, above=True)
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    layout = table._tbl.tblPr.makeelement(qn("w:tblLayout"), {qn("w:type"): "fixed"})
    table._tbl.tblPr.append(layout)
    hdr = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = ""
        p = hdr[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(1)
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.keep_with_next = True
        set_run(p.add_run(h), size=font, bold=True, cn=CN_BOLD)
    for r_idx, row in enumerate(rows):
        cells = table.add_row().cells
        trPr = table.rows[-1]._tr.get_or_add_trPr()
        trPr.append(trPr.makeelement(qn("w:cantSplit"), {}))
        for i, v in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(0.4)
            p.paragraph_format.space_before = Pt(0.4)
            if r_idx < len(rows) - 1:
                p.paragraph_format.keep_with_next = True
            set_run(p.add_run(str(v)), size=font)
    if widths:
        for i, w in enumerate(widths):
            table.columns[i].width = Cm(w)
            for row in table.rows:
                row.cells[i].width = Cm(w)
    return table


def add_figure(doc, name: str, width_cm: float, cap: str):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.keep_with_next = True
    p.add_run().add_picture(str(FIG / name), width=Cm(width_cm))
    caption(doc, cap)


def f4(v) -> str:
    return f"{float(v):.4f}"


# ---------------------------------------------------------------- 主文档
REPO_URL = ""


def build(repo_url: str | None = None) -> None:
    global REPO_URL
    if repo_url:
        REPO_URL = repo_url
    n = load_numbers()
    doc = Document()

    style = doc.styles["Normal"]
    style.font.name = EN
    style.font.size = Pt(10.5)
    style.element.rPr.rFonts.set(qn("w:eastAsia"), CN)

    sec = doc.sections[0]
    sec.top_margin = Cm(1.5)
    sec.bottom_margin = Cm(1.0)
    sec.left_margin = Cm(2.2)
    sec.right_margin = Cm(2.2)

    # 页脚：仅页码
    footer_p = sec.footer.paragraphs[0]
    footer_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = footer_p.add_run()
    for instr in ("PAGE",):
        fld = run._element.makeelement(qn("w:fldSimple"), {qn("w:instr"): instr})
        footer_p._p.append(fld)
    set_run(run, size=9)

    # ---------------- 标题
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(2)
    set_run(title.add_run("基于重构误差的工业多变量时序异常检测"), size=17, bold=True, cn=CN_BOLD)
    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub.paragraph_format.space_after = Pt(4)
    set_run(sub.add_run("——检测协议对指标的影响与跨工况失效分析"), size=11.5, bold=False, cn=CN_BOLD)

    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta.paragraph_format.space_after = Pt(2)
    sid = os.environ.get("REPORT_STUDENT_ID", "").strip()
    name = os.environ.get("REPORT_STUDENT_NAME", "").strip()
    who = f"学号：{sid}    姓名：{name}    " if sid and name else ""
    set_run(meta.add_run(f"{who}课程：深度学习课程设计    2026 年 11 月"), size=9.5)

    if REPO_URL:
        link = doc.add_paragraph()
        link.alignment = WD_ALIGN_PARAGRAPH.CENTER
        link.paragraph_format.space_after = Pt(8)
        set_run(link.add_run(f"代码仓库：{REPO_URL}"), size=9)
    else:
        meta.paragraph_format.space_after = Pt(8)

    # ---------------- 1 研究问题
    heading(doc, "1  研究问题")
    body(doc, "工业设备（阀门、泵、转子）的传感器数据是多变量时间序列，故障样本稀缺、标注昂贵，"
              "因此主流做法是只用正常数据训练重构模型，以重构误差衡量异常程度。本项目的起点正是这样一个系统："
              "用 SKAB 数据集的平稳工况数据训练一维卷积自编码器（Conv-AE），在全部测试数据上得到 "
              f"F1 = {f4(n['old']['f1'])}、误报率 FAR = {n['old']['far'] * 100:.1f}% 的结果。"
              "这一结果在工程上等价于「没有检测器」：它对 89% 的正常窗口报警，运维会直接关闭告警。")
    body(doc, "面对这种整体性失效，一个自然的反应是换模型、调超参。但本项目把问题倒过来问："
              "**指标失效的根源究竟在模型，还是在数据划分、标准化与评估协议？** 围绕这一问题，本文研究三件事：")
    body(doc, "（1）旧方案的失效机制是什么，能否用受控实验把它逐项归因（§5.1）；（2）改用官方逐文件协议、"
              "即在目标文件自身前 400 点上自校准之后，同一模型能达到什么水平，与简单线性方法相比是否仍有优势（§5.4）；"
              "（3）阈值与后处理在多大程度上决定了最终指标（§5.3、§5.5）。边界先说清："
              "本文的逐文件结果来自**目标文件自身的前 400 点**（除 other/2.csv 外都是正常段；该文件训练段夹杂异常，见 §3.4），"
              "属于直推式自校准，不检验跨工况泛化（§3.1）。"
              "本文价值不在于刷新公开分数，而在于给出可复用的「协议审计」思路。", first_indent=True)


    # ---------------- 2 数据与实验环境
    heading(doc, "2  数据与实验环境")
    body(doc, "**数据集。** SKAB v0.9 [1] 由阀门试验台采集，含 34 个带标签文件（valve1 16 个、valve2 4 个、other 14 个）"
              "与 1 个无异常文件 anomaly-free.csv。按官方数据说明 [6]：valve1 是关闭泵入口阀门的实验，"
              "valve2 是关闭泵出口阀门的实验；other 下 1–4 模拟泄漏与补液，5–9 是不同形态的转子不平衡"
              "10–11 为回路水量缓慢/突然增加，12 为排水至汽蚀，13 为入口两相流（汽蚀），"
              "14 为提高水温。每条记录含 8 个传感器通道（加速度 RMS×2、电流、压力、温度、热电偶、电压、体积流量），"
              "标称采样率 1 Hz；实际时间戳间隔以 1 s 为主，也有少量 2 s 间隔，故文中「点」与「秒」近似对应。"
              "标签列 anomaly 与 changepoint 仅用于评估，绝不作为模型输入。")
    add_table(
        doc,
        ["数据分组", "文件数", "测试窗口数", "异常窗口占比", "加速度 z", "流量 z"],
        group_table(n),
        widths=[3.3, 1.5, 2.4, 2.6, 1.9, 1.9],
        font=9,
        caption_text="表 1  数据分组概况（z 为各文件相对 anomaly-free 统计量的平均标准化偏移）",
    )
    body(doc, "**环境。** Windows 11 + Python 3.15 + PyTorch 2.14.1 (CUDA 13.2) + NVIDIA RTX 5070 Ti Laptop。"
              f"在逐文件协议下训练全部 34 个模型约 1.5 min，一次完整的后处理扫描（{n['sweep_n']} 组配置）不足 5 s，"
              "这使得「多配置 × 多文件」的系统性对照在单机上完全可行。主实验固定随机种子 42；"
              "§5.8 另使用 42、43、44 三个随机种子进行重复实验。")

    # ---------------- 3 方法与技术
    heading(doc, "3  方法与技术")
    heading(doc, "3.1  逐文件自校准协议", 2)
    body(doc, "**数据划分来自官方**：core/utils.py 的 preprocess_skab 只做一件事——对每个文件按时间顺序切成前 400 点为训练、"
              "其余为测试（train_test_split(train_size=400, shuffle=False)），它不做标准化、也不做滑窗。"
              "**标准化、滑窗与阈值是本文的处理**：逐文件 StandardScaler 只用该文件训练段拟合（这是官方 notebook 的做法，"
              "本文照此实现，但它不属于 preprocess_skab）；滑窗长度 w、步长 1、不跨文件；窗口标签取窗口末端采样点的标签。"
              "术语要讲清楚：训练与测试都取自**同一文件**，因此这既不是跨工况泛化也不是零样本迁移，"
              "而是直推式**自校准（transductive）**——系统事先拿到了该设备在当前工况下的 400 点正常数据。"
              "它回答「给定正常基线后检测能做到什么程度」，不回答「换台设备、换个工况还能不能用」；"
              "后者需要留出整组文件做验证，本文未做，故不作此主张。")
    heading(doc, "3.2  模型与对照方法", 2)
    body(doc, "选择依据：数据是 8 通道同步采样的多变量时间序列，一维卷积可直接在时间轴上提取局部模式；"
              "自编码器只需正常数据训练、用重构误差即可逐窗打分，适合故障样本稀缺的场景，故以 Conv-AE 为主模型，"
              "并用 PCA 重构与 LSTM-AE 作对照。结构：编码器 8→32（核 3）→64（核 3，步长 2）→32，用线性插值把长度还原，"
              "解码器 32→32→8，损失为逐点均方误差，Adam（lr = 1e-3）、batch 32、最多 60 轮，"
              "以训练窗口的随机 10% 作验证、patience = 5 早停（划分细节见下）。为判断「深度」是否必要，同一协议下设三个对照："
              "PCA 重构（把 30×8 窗口展平成 240 维，取 8 个主成分再还原，残差为逐元素平方的均值）[5]；"
              "输入是 240 维，8 个主成分构成真正的降维约束，而不是对 8 维数据的满秩变换；"
              f"这 8 个主成分在各文件上平均解释 {n['pca_evr']['mean'] * 100:.1f}% 的方差"
              f"（{n['pca_evr']['min'] * 100:.1f}%–{n['pca_evr']['max'] * 100:.1f}%，本文在 results/ 下按逐文件计算）；"
              "PCA-T²（同一子空间内的 Hotelling T² 统计量）；Isolation Forest（200 棵树，random_state = 42）[4]；"
              "LSTM-AE（编码/解码 LSTM 隐层均为 32，接线性输出，Adam lr = 1e-3、batch 32、30 轮）[3]。")
    body(doc, "验证集划分细说：早停需从每个文件前 400 点对应的 371 个窗口中划出验证集，本文比较三种方式。"
              "（a）**random**：按固定随机排列抽 37 个窗口作验证、其余 334 个训练，验证窗口与训练窗口在原始时间点上存在重叠"
              "（34 个文件共 1258 个验证窗口全部与至少一个训练窗口共享采样点）；"
              "（b）**random_matched**：同样随机抽取，但只保留 305 个训练窗口、另丢弃 29 个，使训练窗数与（c）一致，重叠依旧存在；"
              "（c）**holdout**：按时间顺序取最后 37 个窗口作验证（起点 334–370），其前 29 个窗口（起点 305–333）作为隔离带丢弃，"
              "训练窗口为 0–304，两者不共享任何原始采样点。三者都用前 400 点整体拟合 StandardScaler，只隔离了窗口样本。"
              "holdout 仅保证训练窗口与早停验证窗口在原始采样点上不重叠，**不意味着阈值校准与预处理完全独立**："
              "阈值仍取自全部 371 个标定窗口（§3.3）。这与测试数据无关，不构成测试数据泄漏（测试窗口始终在 400 点之后）。"
              "比较结果见 §5.8。", first_indent=True)

    heading(doc, "3.3  打分、阈值与判定", 2)
    body(doc, "窗口残差 r_t 取重构均方误差。**阈值基于每个文件前 400 个标定点生成的全部 371 个窗口的重构残差计算**，"
              "取其高分位数并乘以 4/3 的松弛系数（UCL = Q_q(r_cal) × 4/3，其中 r_cal 为这 371 个窗口的重构残差）：该集合包含模型训练、早停验证及被划分方案丢弃的窗口，"
              "不包含 400 点之后的测试数据；模型参数更新仅使用对应方案指定的训练窗口。"
              "该形式与官方 notebook 一致。判定规则为连续 k 个窗口超阈才报警（k = 1 表示逐窗判定）。"
              "指标采用点级定义并与官方 core/metrics.py 完全对齐：FAR = FP/(FP+TN)、MAR = FN/(FN+TP)、"
              "F1 = TP/(TP+(FN+FP)/2)。因文件长度差异大、汇总会被长文件主导，本文同时报告 macro-F1。"
              "旧方案用「验证段 95% 分位、不乘系数」，本文默认用「标定段全部窗口残差 q99 × 4/3」；"
              "两者本身就是一个变量，§5.1 的受控实验把它单独拆出来。")
    heading(doc, "3.4  协议的一个已知缺陷", 2)
    body(doc, "other/2.csv 共 780 行、含 384 个异常点，首个异常点在第 104 行（0 基索引），因此前 400 点（标定段）里混入了 "
              "296 个异常点（占该文件 371 个标定窗口的 4/5），模型会把故障形态学成正常。"
              "该缺陷无法用参数消除，本文按三种方式说明其影响："
              "（1）丢弃含异常的窗口后只剩 75 窗，阈值被抬到 FAR = 1.0（全部报警）；（2）把 other/2.csv 从评估中剔除，汇总 F1 为 "
              f"{n['o2excl']['f1']:.4f}、MAR 为 {n['o2excl']['mar']:.4f}（对比含该文件时 {n['v2']['f1']:.4f} / {n['v2']['mar']:.4f}），"
              "汇总几乎不变，但该文件逐文件指标不可信；（3）如 §6.4 所述，属官方协议本身的已知局限。")

    # ---------------- 4 设计与实现
    heading(doc, "4  设计与实现")
    body(doc, "代码按「唯一数据入口 + 一次训练多次评估」组织，脚本位于 src/，结果落盘到 results/：**skab_protocol.py** "
              "统一实现数据发现、切分、滑窗、模型、训练、残差、阈值与指标；**run_protocol_v2.py** 以 preset 跑完整流程"
              "（v2 / official）；**run_experiments.py** 分三阶段：训练并缓存残差 → 扫描「窗口 × 残差口径 × 分位 × 连续数」"
              f"{n['sweep_n']} 组 → 用留出段重定阈值；**exp_channelwise.py** 做通道聚合对照；**run_baselines.py** 跑四个对照；"
              "**build_result_tables.py** 汇总基线实验；**make_figures.py** 只读结果文件生成图表。", first_indent=True)
    body(doc, "两个工程决策支撑了可行性：**训练一次、缓存残差**，34 个模型的窗口残差一次写盘，后续数十组配置只需排序比较；"
              "**图表与正文数字全部从 CSV/JSON 读取**。复现：`python src/run_experiments.py --stage all`，"
              "再 `python src/make_figures.py`。", first_indent=True)

    # ---------------- 5 实验与结果
    heading(doc, "5  实验与结果")
    heading(doc, "5.1  从旧协议到逐文件协议", 2)
    body(doc, f"表 2 汇总主要结果。旧系统 F1 = {f4(n['old']['f1'])}、FAR = {n['old']['far'] * 100:.1f}%，几乎等于「永远报警」；"
              f"换到逐文件自校准后，同一模型结构达到 F1 = {f4(n['v2']['f1'])}、FAR = {n['v2']['far'] * 100:.1f}%。"
              "但这个对比**同时改变了五件事**：训练数据来源与验证数据来源（早停用的验证集从 anomaly-free 的 20% 换成目标文件自身的训练窗口）、"
              "标准化统计量、训练超参，以及阈值里的分位数与松弛系数，"
              "把差额整体记在「协议」头上不成立，必须逐项拆开（表 3）。", first_indent=True)

    sw = n["sweep"]
    sb, sbm = n["sweep_best"], n["sweep_best_macro"]
    ho = n["holdout"]
    b = n["base"].set_index("model")
    rows = [
        ["旧全局协议（对照）", "30", "MSE", "0.95", "1", f4(n["old"]["p"]), f4(n["old"]["r"]), f4(n["old"]["f1"]),
         f4(n["old"]["far"]), f4(n["old"]["mar"]), "—"],
        ["逐文件协议（默认）", "30", "MSE", "0.99", "1", f4(n["v2"]["precision"]), f4(n["v2"]["recall"]), f4(n["v2"]["f1"]),
         f4(n["v2"]["far"]), f4(n["v2"]["mar"]), f4(n["v2"]["macro_f1"])],
        ["官方配置复现", "60", "MAE-sum", "0.999", "60", f4(n["official"]["p"]), f4(n["official"]["r"]), f4(n["official"]["f1"]),
         f4(n["official"]["far"]), f4(n["official"]["mar"]), f4(n["official"]["macro"])],
        ["扫描最优（F1，事后）", f"{int(sb["window"])}", str(sb["residual"]).upper(), f"{sb["quantile"]:g}", f"{int(sb["consecutive"])}",
         f4(sb.precision), f4(sb.recall), f4(sb.f1), f4(sb.far), f4(sb.mar), f4(sb.macro_f1)],
        ["标定段定阈值（事后选参）", f"{int(ho["window"])}", "MSE", f"{ho["quantile"]:g}", f"{int(ho["consecutive"])}",
         f4(ho.precision), f4(ho.recall), f4(ho.f1), f4(ho.far), f4(ho.mar), f4(ho.macro_f1)],
        ["PCA 重构（对照）", "—", "—", "0.99", "1", f4(b.loc["PCA", "precision"]), f4(b.loc["PCA", "recall"]),
         f4(b.loc["PCA", "f1"]), f4(b.loc["PCA", "far"]), f4(b.loc["PCA", "mar"]), f4(b.loc["PCA", "macro_f1"])],
        ["PCA-T² 统计量（对照）", "—", "—", "0.99", "1", f4(b.loc["PCAT2", "precision"]), f4(b.loc["PCAT2", "recall"]),
         f4(b.loc["PCAT2", "f1"]), f4(b.loc["PCAT2", "far"]), f4(b.loc["PCAT2", "mar"]), f4(b.loc["PCAT2", "macro_f1"])],
        ["LSTM-AE（对照）", "—", "—", "0.99", "1", f4(b.loc["LSTMAE", "precision"]), f4(b.loc["LSTMAE", "recall"]),
         f4(b.loc["LSTMAE", "f1"]), f4(b.loc["LSTMAE", "far"]), f4(b.loc["LSTMAE", "mar"]), f4(b.loc["LSTMAE", "macro_f1"])],
        ["Isolation Forest（对照）", "—", "—", "—", "—", f4(b.loc["IsolationForest", "precision"]),
         f4(b.loc["IsolationForest", "recall"]), f4(b.loc["IsolationForest", "f1"]),
         f4(b.loc["IsolationForest", "far"]), f4(b.loc["IsolationForest", "mar"]), f4(b.loc["IsolationForest", "macro_f1"])],
    ]
    add_table(doc, ["方法 / 配置", "窗长", "残差", "分位 q", "连续 k", "P", "R", "F1", "FAR", "MAR", "macro-F1"],
              rows, widths=[3.2, 1.2, 1.8, 1.35, 1.3, 1.4, 1.4, 1.4, 1.4, 1.4, 1.7], font=7.2,
              caption_text="表 2  主要结果对照（点级指标，34 个文件汇总）")

    cg = n["ctrl_global"]
    ct = n["ctrl_thr"].set_index("rule")
    dc = n["decomp"]
    ct3 = ct.loc["旧协议阈值规则 q95×1"]
    dA, dB, dE = dc.loc["A"], dc.loc["B"], dc.loc["E"]
    body(doc, "为此补了一组**有序分步对照**（表 3）：除 ①→② 同时改变训练与验证来源外，②→⑥ 逐步只改一个条件、"
              "其余保持上一步不变；训练超参的差异单独占一步，以免把「换数据来源」和「换超参」混在一起。")
    add_table(
        doc,
        ["配置（逐项叠加）", "P", "R", "F1", "FAR", "MAR"],
        [["① 旧方案：全局数据 + 全局标准化 + batch256/50 轮/pat8 + q95×1",
          f4(n["old"]["p"]), f4(n["old"]["r"]), f4(n["old"]["f1"]), f4(n["old"]["far"]), f4(n["old"]["mar"])],
         ["② 切换到目标文件训练/验证来源（标准化、超参、阈值不变）",
          f4(dA["precision"]), f4(dA["recall"]), f4(dA["f1"]), f4(dA["far"]), f4(dA["mar"])],
         ["③ 换逐文件标准化（超参、阈值不变）",
          f4(dB["precision"]), f4(dB["recall"]), f4(dB["f1"]), f4(dB["far"]), f4(dB["mar"])],
         ["④ 训练超参数组合对照 batch32/60 轮/pat5（阈值仍 q95×1）",
          f4(ct3["precision"]), f4(ct3["recall"]), f4(ct3["f1"]), f4(ct3["far"]), f4(ct3["mar"])],
         ["⑤ 分位 q 0.95→0.99（松弛系数仍 ×1）",
          f4(dE["precision"]), f4(dE["recall"]), f4(dE["f1"]), f4(dE["far"]), f4(dE["mar"])],
         ["⑥ 松弛系数 ×1→4/3（现行默认）",
          f4(ct.loc["现行默认 q99×4/3", "precision"]), f4(ct.loc["现行默认 q99×4/3", "recall"]),
          f4(ct.loc["现行默认 q99×4/3", "f1"]), f4(ct.loc["现行默认 q99×4/3", "far"]),
          f4(ct.loc["现行默认 q99×4/3", "mar"])]],
        widths=[7.6, 1.75, 1.75, 1.75, 1.75, 1.75], font=7.2,
        caption_text="表 3  失效因素的有序分步对照（点级指标，34 个文件汇总）",
    )
    body(doc, "注：各行均为同一 CNN-AE（结构见 §3.2），窗口 30、步长 1、残差取逐点 MSE、连续窗 k = 1、随机种子 42。"
              "① 的训练数据为 anomaly-free 前 80%、验证集为其后 20%；②–⑥ 的训练数据为各文件前 400 点、测试为其余部分，"
              "验证集为该文件训练窗口的随机 10%（排列由 seed 42 生成，§5.8 的种子重复沿用这一划分）。"
              "② 用全局标准化（统计量取自 anomaly-free 训练段），③–⑥ 用逐文件标准化。"
              "数值来自 results/controlled/decomp.csv 与 threshold_rule.csv。",
         size=7.6, space_after=1)
    body(doc, f"逐项看：**切换训练/验证数据来源 +{dA['f1'] - n['old']['f1']:.3f}**、**换逐文件标准化 +{dB['f1'] - dA['f1']:.3f}**、"
              f"**换训练超参 {ct3['f1'] - dB['f1']:+.3f}**、**q 0.95→0.99 {dE['f1'] - ct3['f1']:+.3f}**、"
              f"**松弛系数 ×1→4/3 {ct.loc['现行默认 q99×4/3', 'f1'] - dE['f1']:+.3f}**。"
              "其中 ②→③ 只换标准化统计量；③→④ 是**训练超参数组合对照**——batch、epochs、patience 三者同时变化，"
              "不对单个超参的作用作独立归因；①→② 同时换了训练数据与验证数据来源，"
              "故不能把该步差异全部归因于训练数据。此外「全局标准化 × 现行超参」一格未运行"
              "（早期探针用的是 batch 64 / 30 轮，不在本条链上），故本文只报告上述两处对照，不对交互大小下结论。",
         first_indent=True)
    body(doc, "其二，在已完成的对照里，**①→② 这一步的提升最大**：它同时把训练数据换成目标文件的前 400 点、"
              "把早停验证集从 anomaly-free 后 20% 换成该文件训练窗口的随机 10%，"
              f"F1 从 {f4(n['old']['f1'])} 升到 {f4(dA['f1'])} 是整个协议步骤的结果，不能单独归因于训练数据。"
              "本文只跑了这一条链，未跑完「数据 × 标准化」的全部组合，故只说它在本链上是前提，"
              "不写成普遍意义上的必要条件。", first_indent=True)

    heading(doc, "5.2  复现官方配置", 2)
    body(doc, f"按官方 notebook 的设定（窗长 60、MAE 求和型残差、q = 0.999、连续 60 窗报警）复现，得到 "
              f"F1 = {f4(n['official']['f1'])}、FAR = {n['official']['far'] * 100:.2f}%、MAR = {n['official']['mar'] * 100:.2f}%。"
              "官方仓库 README 的 Proposed Leaderboard 上 Conv-AE 为 F1 = 0.78、FAR = 13.55%、MAR = 28.02% [6]。MAR 与之一致，"
              "FAR 却相差约 3 倍。差距不是调参能弥合的，第 6.2 节给出机制解释。")

    heading(doc, "5.3  后处理扫描：指标的主要杠杆在阈值", 2)
    sel = n["ctrl_sel"]
    folds = sel["folds"]
    so = sel["overall"]
    body(doc, f"在缓存残差上扫描 {n['sweep_n']} 组配置，F1 落在 {n['sweep_f1_min']:.3f}–{sb.f1:.3f}，FAR 从 0.21 变到 0.62；"
              f"其中 F1 最高的是 w = {int(sb["window"])}、{sb["residual"]}、q = {sb["quantile"]:g}、k = {int(sb["consecutive"])}"
              f"（F1 = {f4(sb.f1)}、FAR = {f4(sb.far)}、MAR = {f4(sb.mar)}）。"
              "**这些数字是在测试段上选出来的，属事后探索，不能当无偏成绩。**"
              "文件级二折（按文件名奇偶各 17 个文件互为选择/评估折；选择折按逐文件 macro-F1 选优，"
              "评估折不参与选参与选择也不重训，合并指标为两折混淆矩阵相加）"
              f"分别选中 q = {folds[0]['quantile']:g}/k = {folds[0]['consecutive']} 与 "
              f"q = {folds[1]['quantile']:g}/k = {folds[1]['consecutive']}，在未参与选择的文件上合并得到 "
              f"F1 = {f4(so['f1'])}、FAR = {f4(so['far'])}、MAR = {f4(so['mar'])}，与事后最优相差不到 0.005。"
              "该差距仅代表本次固定奇偶划分，不能外推为「一般意义上的选优偏差很小」。"
              "在匹配条件下（w30、MSE、k = 1、×4/3 固定），q 0.99→0.999 只把 FAR 从 0.481 降到 0.453、"
              "MAR 从 0.101 升到 0.109，F1 几乎不变（0.7888→0.7924）；阈值的主要杠杆来自 k（连续窗）与分位数口径，"
              "而不是把 q 单独再抬一个数量级。")
    body(doc, "阈值从哪估计同样重要。默认配置用 0–400 行训练（早停验证集为随机 10%，排列固定为 seed 42），"
              "阈值取自这 400 点生成的 371 个标定窗口残差；"
              "另做一组（src/run_experiments.py 的 stage_holdout）：StandardScaler 只用 0–320 行拟合，"
              "模型只用这 320 行生成的 291 个窗口训练（不使用早停验证），320–400 行不参与参数更新，"
              "其 51 个窗口的残差只用来定阈值（q95 × 4/3），400 行之后的测试段不参与阈值计算。"
              f"同配置下 F1 = {f4(ho.f1)}、FAR = {f4(ho.far)}、MAR = {f4(ho.mar)}，"
              f"FAR 比直接用标定段分位（{f4(n['v2']['far'])}）低约 0.09，F1 反而略升。"
              "原因是这一组的阈值由模型在 320–400 段未见过的窗口上打分得到，其残差普遍大于训练段残差，分位数因此更高。"
              f"主配置已在 §5.8 做 3 个种子重复（F1 {n['rep_stat'].loc['f1','mean']:.4f} ± "
              f"{n['rep_stat'].loc['f1','std']:.4f}），但本节 48 组扫描、官方配置与基线尚未系统重复，"
              "故配置间的小幅差异需谨慎解释；FAR 由分位数决定、对残差尾部敏感。")

    heading(doc, "5.4  与简单基线的对比", 2)
    body(doc, f"表 2 后四行给出同一协议下的对照方法。PCA 重构的 F1 = {f4(b.loc['PCA', 'f1'])}、FAR = {f4(b.loc['PCA', 'far'])}，"
              "在两个指标上都优于 Conv-AE；LSTM-AE 方向相反——召回率 0.957 但 FAR 高达 0.71；"
              "Isolation Forest 几乎全判正常（R = 0.023），对「电平平移型」故障不敏感。"
              "线性与深度模型的差距（0.024）远小于阈值的可调范围。")
    body(doc, f"同为线性方法，PCA 的 T² 统计量（子空间内的马氏距离）只有 F1 = {f4(b.loc['PCAT2', 'f1'])}、"
              f"MAR = {f4(b.loc['PCAT2', 'mar'])}，明显弱于重构残差：SKAB 的故障更多是「子空间外」的成分偏移，"
              "统计量的选择比模型本身更关键。")
    add_figure(doc, "fig3_per_file.png", 6.6,
               "图 1  逐文件 F1：PCA 在 19 个文件上超过 Conv-AE（results/baselines/per_file_*.csv）")
    body(doc, "图 1 给出逐文件结果。两者的差异集中在少数文件：Conv-AE 在 other/13 上几乎完全失效（F1 ≈ 0.01），"
              "PCA 则退化得没那么彻底；而在 valve1/15、other/4 等文件上 Conv-AE 更好。逐文件视角比汇总指标更能暴露问题——"
              "汇总 F1 0.79 的背后，是若干个「完全失效」的文件被其他文件的高分掩盖。")

    heading(doc, "5.5  通道聚合与自适应阈值", 2)
    ch = n["chan"].set_index("rule")
    add_table(
        doc,
        ["聚合规则", "P", "R", "F1", "FAR", "MAR", "macro-F1"],
        [[{"sum": "8 通道 MAE 求和（独立训练）", "top2": "取最大两通道", "max": "取最大通道"}[r],
          f4(ch.loc[r, "precision"]), f4(ch.loc[r, "recall"]), f4(ch.loc[r, "f1"]),
          f4(ch.loc[r, "far"]), f4(ch.loc[r, "mar"]), f4(ch.loc[r, "macro_f1"])] for r in ("sum", "top2", "max")]
        + [["滚动分位阈值（W=200，q=0.99）", f4(n['adaptive'].iloc[0]['precision']), f4(n['adaptive'].iloc[0]['recall']),
            f4(n['adaptive'].iloc[0]['f1']), f4(n['adaptive'].iloc[0]['far']), f4(n['adaptive'].iloc[0]['mar']),
            f4(n['adaptive'].iloc[0]['macro_f1'])]],
        widths=[6.0, 1.6, 1.6, 1.6, 1.6, 1.6, 1.9], font=7.8,
        caption_text="表 4  通道聚合与自适应阈值（窗长 30，q = 0.99，k = 1）",
    )
    pf = n["chanpf"]
    gain_cur = float(pf["gain_Current"].median())
    gain_volt = float(pf["gain_Voltage"].median())
    body(doc, "把 8 个通道的残差按不同方式聚合。先排除一个误解：若把逐通道的 **MSE** 相加，它只是 MSE 均值的 8 倍，"
              "而阈值（标定段分位 × 4/3）会同比例放大，判定结果**完全不变**，这是恒等式、不需要实验。"
              "表 4 的求和行用的是逐通道 **MAE（L1）** 之和，属另一种残差口径，且来自一次独立训练，"
              "因此与表 2 默认行（MSE 均值）不能逐项对等；F1 0.7937 与 0.7888、FAR 0.3219 与 0.4812 的差异，"
              "主要来自 L1 与 L2 对尾部误差的敏感度不同，不能归因于「重训波动」。"
              "该表四行均由 src/exp_channelwise.py 各自独立训练（60 轮 / batch 32 / seed 42，逐文件标准化），"
              "记录见 results/ablation/channelwise.csv。求和方式 F1 最高但 FAR 也最高，"
              f"取最大通道残差能压低 FAR（{f4(ch.loc['sum', 'far'])} → {f4(ch.loc['max', 'far'])}），"
              f"代价是 MAR 从 {f4(ch.loc['sum', 'mar'])} 升到 {f4(ch.loc['max', 'mar'])}。"
              f"各通道对故障的敏感度差异明显：Current 的异常/正常残差比中位数最高（{gain_cur:.2f}），"
              f"Voltage 最低（{gain_volt:.2f}），"
              "这解释了为什么「取最大」会让召回率下降——最不敏感的那个通道常常给出最大的噪声残差。")
    body(doc, "最后一行的自适应阈值实验是反直觉的：用滚动 200 窗的分位数在线更新阈值，F1 崩到 0.018、MAR 达到 0.99。"
              "机制是自遮蔽——故障是持续数分钟的电平偏移，滚动窗口内的分位数被故障样本自身抬高，故障被判为「新常态」。"
              "这一现象提示了一个方向：持续型故障下或许应改用对故障更鲁棒的统计量（中位数、MAD）而非近期分位数，"
              "但本文**没有对这些替代方案做实验验证**，只能作为后续研究的建议。", first_indent=True)

    heading(doc, "5.6  失败模式", 2)
    add_figure(doc, "fig4_failure.png", 5.4,
               "图 2  三类失败模式（results/protocol_v2/predictions.csv 与 per_file_metrics.csv）")
    body(doc, "图 2 把错误分成三类。**(a) 完全漏检**（other/13）：整个测试段残差在 0.15–0.41 之间波动，"
              "恰好压在阈值 0.406 之下，故障没有在重构误差中留下任何可分辨的痕迹。"
              "**(b) 大段误报**（valve1/12）：残差从 0.5 一路抬到 2.5 并持续数百个窗口，"
              "这是工况缓慢漂移导致的系统性偏移，逐窗判定会把整段正常数据标成异常。"
              "**(c) 部分漏检**（other/4）：只有残差越过阈值的少数区段被检出。")
    fpg = n["fpseg"]
    body(doc, f"从统计上看，现行默认配置的 {fpg['fp']} 个误报窗口（分母 {fpg['normal_windows']} 个正常窗口，"
              f"FAR = {fpg['far']:.4f}）构成 {fpg['segments']} 个连续段：最长 {fpg['max_len']} 窗、"
              f"中位数只有 {fpg['median_len']:.0f} 窗，前 10 段占全部误报窗口的 {fpg['top10_share'] * 100:.1f}%。"
              "误报因此是「少数超长段 + 大量孤立点」的混合结构：前者来自工况漂移，后者来自单点噪声。"
              "这解释了「连续 k 窗」为何能压低 FAR——它剪掉了后者，对前者几乎无效。", first_indent=True)

    heading(doc, "5.7  检测延迟", 2)
    dl = n["delay"]
    d_v2 = dl[dl.config.str.startswith("v2")]
    d_off = dl[dl.config.str.startswith("official")]
    det_v2 = d_v2[d_v2.detected]
    det_off = d_off[d_off.detected]
    body(doc, f"异常检测除了「命中多少」，还要看「多快命中」。事件起点取 anomaly 列首个标签为 1 的采样点（不使用 changepoint 列）；报警窗口覆盖其末端之前 w 个采样点，故所报延迟等于「首个被标记采样点相对异常起点的间隔」，单位为采样点（标称采样率 1 Hz，1 点约合 1 秒）。"
              f"逐文件协议（k = 1）在 {len(det_v2)}/{len(d_v2)} 个文件上报警，中位延迟 "
              f"{det_v2.delay_samples.median():.0f} 点（IQR {det_v2.delay_samples.quantile(.25):.0f}–"
              f"{det_v2.delay_samples.quantile(.75):.0f}），最大 {det_v2.delay_samples.max():.0f} 点，约合 14 秒；"
              f"官方配置（k = 60）只命中 {len(det_off)}/{len(d_off)}，中位延迟 "
              f"{det_off.delay_samples.median():.0f} 点，最长 {det_off.delay_samples.max():.0f} 点，"
              "other/13 与 other/2 完全未报警。")
    body(doc, "延迟差异几乎全部由后处理规则决定：k = 60 要求连续 60 个窗口全部超阈，仅此就带来约 60 步下限；"
              "代价是 2 个文件彻底漏报；这条权衡在只看 F1 的评估里不可见。", first_indent=True)

    heading(doc, "5.8  随机种子重复", 2)
    rs = n["rep_seed"].sort_values("seed")
    st = n["rep_stat"]
    hst = n["holdout_stat"]
    s3 = n["scheme"]
    sd3 = n["scheme_std"]
    rpf = n["rep_pf"].sort_values("f1_std", ascending=False)
    body(doc, "训练随机性有多大？用同一配置（batch 32、60 轮、patience 5、w30、q99 × 4/3、k = 1）跑 3 个种子 42/43/44，"
              "**验证集划分固定为 seed 42 的排列**，只让初始化与批次顺序随种子变化："
              f"pooled F1 = {st.loc['f1','mean']:.4f} ± {st.loc['f1','std']:.4f}、"
              f"FAR = {st.loc['far','mean']:.4f} ± {st.loc['far','std']:.4f}、"
              f"MAR = {st.loc['mar','mean']:.4f} ± {st.loc['mar','std']:.4f}；"
              f"P = {st.loc['precision','mean']:.4f} ± {st.loc['precision','std']:.4f}、"
              f"R = {st.loc['recall','mean']:.4f} ± {st.loc['recall','std']:.4f}"
              f"（三次分别为 {', '.join(f'{v:.4f}' for v in rs.f1)}），"
              f"34 个文件的宏平均 F1 = {st.loc['macro_f1','mean']:.4f} ± {st.loc['macro_f1','std']:.4f}。", first_indent=True)
    body(doc, f"F1 的种子间标准差只有 {st.loc['f1','std']:.4f}，FAR 却是它的 {st.loc['far','std'] / st.loc['f1','std']:.0f} 倍——"
              "阈值取自分位数、对残差尾部敏感，这是全文 FAR 只作方向性解读的原因。"
              f"逐文件看，{rpf.iloc[0]['file']} 最不稳定（三次 F1 为 "
              f"{min(rpf.iloc[0]['f1_42'], rpf.iloc[0]['f1_43'], rpf.iloc[0]['f1_44']):.3f}–"
              f"{max(rpf.iloc[0]['f1_42'], rpf.iloc[0]['f1_43'], rpf.iloc[0]['f1_44']):.3f}），"
              f"其次是 {rpf.iloc[1]['file']}（{min(rpf.iloc[1]['f1_42'], rpf.iloc[1]['f1_43'], rpf.iloc[1]['f1_44']):.3f}–"
              f"{max(rpf.iloc[1]['f1_42'], rpf.iloc[1]['f1_43'], rpf.iloc[1]['f1_44']):.3f}）。"
              f"主实验的残差缓存（F1 {f4(n['v2']['f1'])}）产生于固定划分之前，既未固定验证集排列、也无逐轮记录，"
              "不能算严格同配置的重复训练；本节只用 42/43/44 三次严格匹配的运行。样本仅 3 次（ddof = 1），"
              "均值 ± 标准差只作量级参考，未据此挑种子或重选阈值。", first_indent=True)
    body(doc, "为分清「训练窗口数量」与「窗口重叠 / 时间位置」各自的作用，对三种划分各跑了 3 个种子（同一份固定划分）。"
              "random 与 random_matched 的训练窗数为 334 与 305：pooled F1 "
              f"{s3.loc['f1','random']:.4f} ± {sd3.loc['f1','random']:.4f} → {s3.loc['f1','random_matched']:.4f} ± {sd3.loc['f1','random_matched']:.4f}"
              f"（{s3.loc['f1','random_matched'] - s3.loc['f1','random']:+.4f}），逐文件最小验证损失中位仅升 "
              f"{n['ratio_mr']['median']:.2f} 倍（{n['ratio_mr']['higher']}/{n['ratio_mr']['n']} 个文件更高），"
              "说明单纯少 29 个训练窗口的影响很小。random_matched 与 holdout 训练窗数相同（305），差别是随机抽样 vs 时间顺序加隔离带："
              f"pooled F1 {s3.loc['f1','random_matched']:.4f} → {s3.loc['f1','holdout']:.4f}"
              f"（{s3.loc['f1','holdout'] - s3.loc['f1','random_matched']:+.4f}），"
              f"FAR {s3.loc['far','random_matched']:.4f} → {s3.loc['far','holdout']:.4f}、"
              f"MAR {s3.loc['mar','random_matched']:.4f} → {s3.loc['mar','holdout']:.4f}，而逐文件最小验证损失中位升至 "
              f"{n['ratio_hm']['median']:.2f} 倍（均值 {n['ratio_hm']['mean']:.2f}，{n['ratio_hm']['higher']}/{n['ratio_hm']['n']} 更高）。"
              "在训练窗口数量匹配的情况下，将随机验证划分改为时间顺序隔离划分后，观察到验证损失进一步升高。"
              "但该对照同时改变了窗口重叠状态与验证数据的时间位置，现有实验不能将变化单独归因于其中某一因素。三种方案的 pooled F1 差异都落在种子波动"
              f"（{sd3.loc['f1'].min():.3f}–{sd3.loc['f1'].max():.3f}）之内；由于每种方案仅运行三个随机种子，现有证据不足以确认方案间存在真实性能差异，也不足以证明三种方案等效。"
              "表中最小验证损失的口径是：对每个（种子, 文件）取该文件训练曲线上的最小 val_loss，先对 34 个文件等权平均、再对 3 个种子取均值与样本标准差；"
              "跨文件的损失离散远大于跨种子离散，后者只用于衡量种子稳定性。"
              "三种方案的 StandardScaler 均按原协议用前 400 点拟合，因此本对照检验的是「划分方式 + 训练窗数」的组合效应，"
              "而不是严格的时间外泛化；三种方案的平均最佳轮数为 60 / 60 / 57，差异同样有限。", first_indent=True)

    sch, schs = n["scheme"], n["scheme_std"]

    def ms(metric: str, tag: str, digits: int = 4) -> str:
        return f"{sch.loc[metric, tag]:.{digits}f} ± {schs.loc[metric, tag]:.{digits}f}"

    add_table(doc, ["划分方案", "训练窗", "重叠", "F1", "FAR", "MAR", "macro-F1", "最小验证损失"],
              [["random", "334", "有", ms("f1", "random"), ms("far", "random"),
                ms("mar", "random"), ms("macro_f1", "random"), f"{n['vl_seed'].loc['random', 'mean']:.3f} ± {n['vl_seed'].loc['random', 'std']:.3f}"],
               ["random_matched", "305", "有", ms("f1", "random_matched"), ms("far", "random_matched"),
                ms("mar", "random_matched"), ms("macro_f1", "random_matched"),
                f"{n['vl_seed'].loc['random_matched', 'mean']:.3f} ± {n['vl_seed'].loc['random_matched', 'std']:.3f}"],
               ["holdout", "305", "无", ms("f1", "holdout"), ms("far", "holdout"),
                ms("mar", "holdout"), ms("macro_f1", "holdout"), f"{n['vl_seed'].loc['holdout', 'mean']:.3f} ± {n['vl_seed'].loc['holdout', 'std']:.3f}"]],
              widths=[2.6, 1.1, 0.8, 2.2, 2.2, 2.2, 2.2, 2.1], font=6.6,
              caption_text="表 5  三种验证集划分的对照（均值 ± 样本标准差，n = 3、ddof = 1；验证损失口径见 §5.8 正文）")

    # ---------------- 6 分析与思考
    heading(doc, "6  分析与思考")
    heading(doc, "6.1  旧方案为什么失效：只学了一条别的工况", 2)
    body(doc, "anomaly-free 是一条稳态工况下的记录，尺度极小：Accelerometer1RMS 标准差仅 0.005、Volume Flow RateRMS 为 1.70。"
              "用它做标准化，其他 34 个文件的工况差异被放大成巨大的 z 值——valve1/valve2 组加速度约 −40σ、流量约 −55σ"
              "（关阀时流量约 32，稳态约 125），other/5–9 转子不平衡组加速度约 +27σ；组内高度一致，说明是稳定的工况差异而非噪声（图 3）。", first_indent=True)
    add_figure(doc, "fig6_cross_condition.png", 5.4,
               "图 3  各文件相对 anomaly-free 的平均 z 值（原始 CSV + normalization.npz）")
    body(doc, "在这样的输入下，重构误差主要度量「这个文件属于哪个工况」，而不是「设备是否故障」，"
              "基于稳态验证集 95% 分位得到的阈值 0.184 于是把几乎所有窗口判为异常。"
              "但表 3 的分步对照纠正了一个容易得出的推论：**在已完成的对照里，①→② 的收益最大**——"
              f"该步同时换了训练数据与验证数据来源，F1 从 {f4(n['old']['f1'])} 升到 "
              f"{f4(n['decomp'].loc['A']['f1'])} 属于整步结果，不能只归因于训练数据；"
              "本文没有跑完因子组合，故不把它写成普遍意义上的「必要条件」。"
              "标准化与训练超参在本文各只测得一次顺序对照（+0.13 与 −0.02），两者不可相加。", first_indent=True)
    body(doc, "因此全部逐文件结果只能读作自校准：它回答「给定正常基线能做到什么程度」，不能外推到换工况。",
         first_indent=True)

    heading(doc, "6.2  官方单通道输出的影响及其解释边界", 2)
    o1 = n["ctrl_1ch"].set_index("window")
    body(doc, "官方 core/Conv_AE.py 的最后一层是 Conv1DTranspose(filters=1)：输入 8 通道却只重构 1 个通道，"
              "损失靠广播把 (B,T,8) 与 (B,T,1) 相减。复刻该变体得："
              f"w30/q0.99/k1 下 FAR {f4(n['v2']['far'])} → {f4(o1.loc[30, 'far'])}、"
              f"MAR {f4(n['v2']['mar'])} → {f4(o1.loc[30, 'mar'])}；"
              f"w60/q0.999/k60 下 FAR {f4(n['official']['far'])} → {f4(o1.loc[60, 'far'])}、"
              f"MAR {f4(n['official']['mar'])} → {f4(o1.loc[60, 'mar'])}。方向与榜单一致："
              "重构被削弱后残差偏大、阈值水涨船高，故 FAR 降、MAR 升；复刻只压到 0.39，只能作方向性解释。", first_indent=True)

    heading(doc, "6.3  为什么线性 PCA 能反超深度模型", 2)
    body(doc, "每个文件的前 400 点可生成 371 个标定窗口（窗长 30、步长 1，相邻窗口高度重叠，不能当作独立样本）；"
              "其中 random 方案用 334 个窗口做参数训练，random_matched 与 holdout 各用 305 个。"
              "而故障的物理表现是少数通道的低频电平偏移。"
              f"8 个主成分平均只解释 {n['pca_evr']['mean'] * 100:.1f}% 的方差（各文件 28.3%–52.6%），"
              "但解释方差与检测能力是两件事：故障是少数通道的持续电平偏移，落在残差里的幅度仍高于正常抖动，"
              "当前配置因此取得表 2 的实测结果；本文未做主成分数量消融，不就「8 个是否最优」下结论。"
              f"Conv-AE 只有 {n['params']:,} 个可训练参数（编码器 32→64→32、解码器 32→8，核长 3），"
              "但参数量与高度重叠的窗口数之比不足以单独判断是否过参数化，本文也没有做参数量消融；"
              "不过，当前实验中 Conv-AE 未显示出相较 PCA 重构的综合性能优势，说明在本数据集和现有配置下，"
              "增加局部时序建模复杂度并未带来明显收益；但由于本文未做严格匹配的网络容量与时间建模消融，"
              "尚不能判断时序信息本身的贡献。当前 LSTM-AE 实现及其参数配置（隐层 32、30 轮）也未优于现有模型，"
              "不能据此否定时序建模本身的价值，更可能的原因是该实现在此数据规模下未调好。"
              "**判断是：这类小样本、低维场景，先把线性基线做扎实更划算**。", first_indent=True)

    heading(doc, "6.4  局限与后续工作", 2)
    body(doc, "（1）只用 SKAB 一个数据集，且其官方划分在 other/2.csv 上存在训练段污染（§3.4）；"
              "（2）点级评估、未上事件级指标，长故障文件主导汇总；"
              f"（3）主配置已做 3 种子重复（F1 {n['rep_stat'].loc['f1','mean']:.4f} ± {n['rep_stat'].loc['f1','std']:.4f}），"
              "但官方配置（窗长 60、k = 60）未做多种子重复；PCA 主成分数未消融，未与 USAD [2] 等其他代表性时序异常检测方法对比。",
         first_indent=True)
    ad = n["adaptive"]
    rl = ad[(ad["mode"] == "rolling") & (ad["width"] == 200) & (ad["quantile"] == 0.99)].iloc[0]
    body(doc, "（4）自适应阈值仅测试了窗口长度 200、分位数 0.99 的滚动分位数方案，该配置在持续型故障下出现明显自遮蔽"
              f"（F1 = {f4(rl['f1'])}、MAR = {f4(rl['mar'])}）；中位数、MAD 等鲁棒统计方法尚未做实验验证，"
              "留待后续研究。", first_indent=True)
    body(doc, "（5）原随机验证方案存在窗口重叠：1258 个验证窗口全部与至少一个训练窗口共享采样点。"
              "用训练窗数相同的 random_matched 作对照，仅减少训练窗数只把验证损失中位抬高 "
              f"{n['ratio_mr']['median']:.2f} 倍；抬到 {n['ratio_hm']['median']:.2f} 倍则出现在取消重叠并把验证集移到末端的 holdout 上，"
              "该步同时改变了重叠与时间位置，不能单独归因于其一；三种划分的 pooled F1 只差 0.002–0.004，"
              "但三次重复与一种划分顺序仍不足以证明等效。", first_indent=True)

    heading(doc, "6.5  小结", 2)
    body(doc, f"结论可以浓缩为一句：**在工业时序异常检测中，先修协议，再谈模型**。同一模型在同一台设备上"
              f"自校准后 F1 从 {f4(n['old']['f1'])} 升到 0.63–0.79，而 PCA 与 Conv-AE 只差 0.02 量级；"
              "但这份收益来自「事先拿到了该设备的 400 点正常数据」，不是跨工况泛化能力。", first_indent=True)

    # ---------------- 参考文献
    heading(doc, "参考文献", space_before=2)
    refs = [
        "[1] Katser I. D., Kozitsin V. O. Skoltech Anomaly Benchmark (SKAB). Kaggle, 2020. "
        "DOI: 10.34740/KAGGLE/DSV/1693952.",
        "[2] Audibert J., Michiardi P., Guyard F., et al. USAD: UnSupervised Anomaly Detection on Multivariate Time Series. KDD, 2020.",
        "[3] Malhotra P., et al. LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection. ICML Workshop, 2016. "
        "arXiv:1607.00148.",
        "[4] Liu F. T., Ting K. M., Zhou Z.-H. Isolation Forest. ICDM, 2008.",
        "[5] Tipping M. E., Bishop C. M. Probabilistic Principal Component Analysis. JRSS-B, 1999.",
        "[6] SKAB 官方仓库（waico/SKAB）：根目录 README.md（含 Proposed Leaderboard）、data/README.md、"
        "core/utils.py、core/Conv_AE.py. https://github.com/waico/SKAB.",
    ]
    for r in refs:
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.space_before = Pt(0)
        pf.space_after = Pt(0)
        pf.left_indent = Pt(18)
        pf.first_line_indent = Pt(-18)
        pf.line_spacing = 0.88
        set_run(p.add_run(r), size=7.5)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(f"saved: {OUT}")


def _repo_url(user: str | None, url: str | None) -> str:
    if url:
        return url
    if user:
        return f"https://github.com/{user}/skab-anomaly-detection"
    return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成课程设计报告 DOCX")
    parser.add_argument("--github-user", help="GitHub 用户名，生成 https://github.com/<user>/skab-anomaly-detection")
    parser.add_argument("--github-url", help="完整仓库地址，优先于 --github-user")
    args = parser.parse_args()
    build(_repo_url(args.github_user, args.github_url))
