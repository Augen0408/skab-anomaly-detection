"""核对报告（docx）里的数字是否与 results/ 下的原始结果一致。

用法（需带 python-docx 的解释器，项目 .venv 默认没有）：
    python src/verify_report.py

任一数字对不上即返回非零退出码，便于在改代码或重跑实验后自检。
"""

from __future__ import annotations

import sys
from pathlib import Path

import docx

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_report import load_numbers  # noqa: E402  只依赖 pandas/docx，不 import torch

ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / "report" / "课程设计报告.docx"

TOL_NOTE = "保留 4 位小数后逐字比对"


def f4(v) -> str:
    return f"{float(v):.4f}"


def table_rows(doc, index: int) -> list[list[str]]:
    return [[c.text.strip() for c in r.cells] for r in doc.tables[index].rows]


def main() -> int:
    if not DOCX.exists():
        print(f"找不到报告：{DOCX}")
        return 2

    n = load_numbers()
    doc = docx.Document(str(DOCX))
    problems: list[str] = []
    checked = 0

    def check_metrics(prefix: str, cells: list[str], p, r, f1, far, mar, macro,
                      start: int = 5) -> None:
        nonlocal checked
        # 列顺序：… P R F1 FAR MAR macro-F1
        for name, got, exp in zip(("P", "R", "F1", "FAR", "MAR", "macro-F1"),
                                  cells[start:start + 6], (p, r, f1, far, mar, macro)):
            checked += 1
            want = "—" if exp is None else f4(exp)
            if got != want:
                problems.append(f"{prefix}/{name}: 报告={got!r} 结果={want!r}")

    # ---- 表 2：主结果对照（含表头 10 行） ----
    t2 = table_rows(doc, 1)
    if len(t2) != 10:
        problems.append(f"表 2 行数异常：{len(t2)}（应为 10）")
    else:
        old = n["old"]
        check_metrics("旧全局协议", t2[1], old["p"], old["r"], old["f1"],
                      old["far"], old["mar"], None)

        v2 = n["v2"]
        check_metrics("逐文件协议", t2[2], v2["precision"], v2["recall"], v2["f1"],
                      v2["far"], v2["mar"], v2["macro_f1"])

        off = n["official"]
        check_metrics("官方配置复现", t2[3], off["p"], off["r"], off["f1"],
                      off["far"], off["mar"], off["macro"])

        sb = n["sweep_best"]
        check_metrics("扫描最优", t2[4], sb["precision"], sb["recall"], sb["f1"],
                      sb["far"], sb["mar"], sb["macro_f1"])

        ho = n["holdout"]
        check_metrics("留出段重定阈值", t2[5], ho["precision"], ho["recall"], ho["f1"],
                      ho["far"], ho["mar"], ho["macro_f1"])

        base = n["base"].set_index("model")
        for row_idx, model in ((6, "PCA"), (7, "PCAT2"), (8, "LSTMAE"), (9, "IsolationForest")):
            rec = base.loc[model]
            check_metrics(model, t2[row_idx], rec["precision"], rec["recall"], rec["f1"],
                          rec["far"], rec["mar"], rec["macro_f1"])

    # ---- 表 3：单因素分解（含表头 7 行，6 列） ----
    tc = table_rows(doc, 2)
    if len(tc) != 7:
        problems.append(f"表 3 行数异常：{len(tc)}（应为 7）")
    else:
        dc = n["decomp"]
        ct = n["ctrl_thr"].set_index("rule")
        old_n = n["old"]
        check_metrics("受控/①旧", tc[1], old_n["p"], old_n["r"], old_n["f1"],
                      old_n["far"], old_n["mar"], None, start=1)
        for row_idx, tag in ((2, "A"), (3, "B"), (5, "E")):
            rec = dc.loc[tag]
            check_metrics(f"受控/{tag}", tc[row_idx], rec["precision"], rec["recall"],
                          rec["f1"], rec["far"], rec["mar"], None, start=1)
        r4 = ct.loc["旧协议阈值规则 q95×1"]
        check_metrics("受控/④换超参", tc[4], r4["precision"], r4["recall"], r4["f1"],
                      r4["far"], r4["mar"], None, start=1)
        r6 = ct.loc["现行默认 q99×4/3"]
        check_metrics("受控/⑥q99×4/3", tc[6], r6["precision"], r6["recall"], r6["f1"],
                      r6["far"], r6["mar"], None, start=1)

    # ---- 表 4：通道聚合与自适应阈值（含表头 5 行） ----
    t3 = table_rows(doc, 3)
    if len(t3) != 5:
        problems.append(f"表 4 行数异常：{len(t3)}（应为 5）")
    else:
        ch = n["chan"].set_index("rule")
        for row_idx, rule in ((1, "sum"), (2, "top2"), (3, "max")):
            rec = ch.loc[rule]
            check_metrics(f"通道聚合/{rule}", t3[row_idx], rec["precision"], rec["recall"],
                          rec["f1"], rec["far"], rec["mar"], rec["macro_f1"], start=1)

        adf = n["adaptive"]
        ad = adf[(adf["mode"].astype(str) == "rolling")
                 & (adf["width"] == 200)
                 & ((adf["quantile"] - 0.99).abs() < 1e-9)].iloc[0]
        check_metrics("自适应阈值", t3[4], ad["precision"], ad["recall"], ad["f1"],
                      ad["far"], ad["mar"], ad["macro_f1"], start=1)

    # ---- 正文里的关键数字与表述 ----
    text = "\n".join(p.text for p in doc.paragraphs)
    for tb in doc.tables:
        for row in tb.rows:
            text += "\n" + " ".join(c.text for c in row.cells)

    delay = n["delay"]
    v2d = delay[delay["config"].astype(str).str.startswith("v2")]
    med = int(v2d["delay_samples"].median())
    hits = int(v2d["detected"].sum())

    st = n["rep_stat"]
    must = {
        "旧协议 F1": f4(n["old"]["f1"]),
        "逐文件 F1": f4(n["v2"]["f1"]),
        "PCA F1": f4(n["base"].set_index("model").loc["PCA", "f1"]),
        "PCA-T² 行": "PCA-T²",
        "检测延迟命中数": f"{hits}/{len(v2d)}",
        "检测延迟中位数": f"中位延迟 {med} 点",
        "多种子现状": "本节 48 组扫描、官方配置与基线尚未系统重复",
        "受控分解小节": "有序分步对照",
        "事后探索声明": "属事后探索",
        "二折无偏 F1": f4(n["ctrl_sel"]["overall"]["f1"]),
        "自校准 F1(A)": f4(n["decomp"].loc["A"]["f1"]),
        "PCA 解释方差": f"{n['pca_evr']['mean'] * 100:.1f}% 的方差",
        "误报窗口数": str(n["fpseg"]["fp"]),
        "误报段数": f"{n['fpseg']['segments']} 个连续段",
        "other/2 剔除 F1": f4(n["o2excl"]["f1"]),
        "other/2 起始索引": "第 104 行",
        "单通道节标题": "官方单通道输出的影响及其解释边界",
        "MAE 求和行": "8 通道 MAE 求和（独立训练）",
        "多种子小节": "5.8  随机种子重复",
        "验证集口径": "以训练窗口的随机 10% 作验证",
        "验证集重叠说明": "1258 个验证窗口全部与至少一个训练窗口共享采样点",
        "无重叠对照": "训练窗口为 0–304",
        "随机划分对照": "random_matched",
        "阈值窗口口径": "阈值基于每个文件前 400 个标定点生成的全部 371 个窗口的重构残差计算",
        "阈值不含测试": "不包含 400 点之后的测试数据",
        "holdout 边界": "不意味着阈值校准与预处理完全独立",
        "无泄漏声明": "不构成测试数据泄漏",
        "LSTM 谨慎结论": "不能据此否定时序建模本身的价值",
        "阈值公式": "UCL = Q_q(r_cal) × 4/3",
        "标定窗口": "阈值仍取自全部 371 个标定窗口",
        "榜单来源": "Proposed Leaderboard 上 Conv-AE 为 F1 = 0.78",
        "5.8 谨慎一": "现有证据不足以确认方案间存在真实性能差异",
        "5.8 谨慎二": "现有实验不能将变化单独归因于其中某一因素",
        "阈值术语": "标定段全部窗口残差 q99 × 4/3",
        "5.3 scaler 范围": "StandardScaler 只用 0–320 行拟合",
        "5.3 阈值集合": "51 个窗口的残差只用来定阈值",
        "6.3 时序建模边界": "尚不能判断时序信息本身的贡献",
        "PCA 方差出处": "本文在 results/ 下按逐文件计算",
        "参考文献 6 构成": "根目录 README.md（含 Proposed Leaderboard）",
        "6.3 窗口口径": "371 个标定窗口（窗长 30、步长 1，相邻窗口高度重叠",
        "6.3 参数量谨慎": "不足以单独判断是否过参数化",
        "种子口径": "主实验固定随机种子 42；§5.8 另使用 42、43、44 三个随机种子进行重复实验",
        "早停验证集": "早停验证集为随机 10%，排列固定为 seed 42",
        "①→② 不可单归因": "整个协议步骤的结果，不能单独归因于训练数据",
        "表3脚注划分": "排列由 seed 42 生成，§5.8 的种子重复沿用这一划分",
        "缓存非严格重复": "不能算严格同配置的重复训练",
        "主配置三种子": "主配置已做 3 种子重复",
        "自适应阈值局限": "自适应阈值仅测试了窗口长度 200、分位数 0.99 的滚动分位数方案",
        "参数量": f"{17064:,} 个可训练参数",
        "表3行②": "② 切换到目标文件训练/验证来源（标准化、超参、阈值不变）",
        "切换来源增益": "切换训练/验证数据来源 +0.086",
        "自适应阈值建议": "但本文没有对这些替代方案做实验验证",
        "小结编号": "6.5  小结",
        "官方配置未重复": "官方配置（窗长 60、k = 60）未做多种子重复",
        "多种子 pooled F1": f"{st.loc['f1','mean']:.4f} ± {st.loc['f1','std']:.4f}",
        "多种子 FAR": f"{st.loc['far','mean']:.4f} ± {st.loc['far','std']:.4f}",
        "验证划分固定": "验证集划分固定为 seed 42 的排列",
    }
    for label, needle in must.items():
        checked += 1
        if needle not in text:
            problems.append(f"正文缺少 {label}：{needle!r}")

    forbids = {
        "首页占位提示": "提交时请填写实际的 GitHub 仓库地址",
        "首页假用户名": "<username>",
        "TranAD 具体名称": "TranAD",
    }
    for label, needle in forbids.items():
        checked += 1
        if needle in text:
            problems.append(f"正文仍出现 {label}：{needle!r}")

    # ---- 表 5：三种验证集划分对照（含表头 4 行） ----
    if len(doc.tables) >= 5:
        t5 = table_rows(doc, 4)
        if len(t5) != 4:
            problems.append(f"表 5 行数异常：{len(t5)}（应为 4）")
        else:
            sch, schs = n["scheme"], n["scheme_std"]
            for row_idx, tag in ((1, "random"), (2, "random_matched"), (3, "holdout")):
                # F1 / FAR / MAR / macro-F1 四列均为「均值 ± 样本标准差」
                for col, metric in ((3, "f1"), (4, "far"), (5, "mar"), (6, "macro_f1")):
                    checked += 1
                    want = f"{sch.loc[metric, tag]:.4f} ± {schs.loc[metric, tag]:.4f}"
                    if t5[row_idx][col] != want:
                        problems.append(f"表 5 {tag}/{metric}：docx={t5[row_idx][col]!r} 结果={want!r}")
                # 最小验证损失：跨种子均值 ± 样本标准差（n = 3）
                checked += 1
                want_vl = (f"{n['vl_seed'].loc[tag, 'mean']:.3f} ± "
                           f"{n['vl_seed'].loc[tag, 'std']:.3f}")
                if t5[row_idx][7] != want_vl:
                    problems.append(f"表 5 {tag}/min_val_loss：docx={t5[row_idx][7]!r} 结果={want_vl!r}")
    else:
        problems.append(f"表数量为 {len(doc.tables)}（应含表 5）")

    checked += 1
    if len(doc.inline_shapes) != 3:
        problems.append(f"内嵌图片数 {len(doc.inline_shapes)}（应为 3）")

    print(f"核对 {checked} 项（{TOL_NOTE}）")
    if problems:
        print("发现问题：")
        for item in problems:
            print("  -", item)
        return 1

    print("全部一致：报告中的数字与 results/ 下的原始结果吻合。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
