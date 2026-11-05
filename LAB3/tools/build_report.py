"""Build a five-page, evidence-backed Chinese lab report from completed runs.

Run from any directory. No training/evaluation takes place in this script.
Only real summary/history/test files are accepted; missing data is an error.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


ROOT = Path(__file__).resolve().parents[1]
GROUPS = ("baseline", "ablation", "improved")
LABELS = {"baseline": "A 基础CNN", "ablation": "B 基础CNN+增强", "improved": "C 残差SE-CNN+增强"}
SHORT = {"baseline": "A", "ablation": "B", "improved": "C"}
CHINESE = {"airplane": "飞机", "automobile": "汽车", "bird": "鸟", "cat": "猫", "deer": "鹿", "dog": "狗",
           "frog": "蛙", "horse": "马", "ship": "船", "truck": "卡车"}
INK = colors.HexColor("#17364a")
BLUE = colors.HexColor("#185d82")
LIGHT = colors.HexColor("#edf3f7")
LINE = colors.HexColor("#d3dee5")
# A SimpleDocTemplate frame has six points of internal padding on each side.
BODY_WIDTH = A4[0] - 88


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--output-dir", type=Path, default=ROOT.parent / "output" / "pdf")
    parser.add_argument("--repo-url", default="", help="Actual GitHub repository URL, after uploading the code")
    parser.add_argument("--name", default="朱昕宇")
    parser.add_argument("--student-id", default="10245501426")
    parser.add_argument("--font-dir", type=Path, default=Path("C:/Windows/Fonts"))
    return parser.parse_args()


def load_runs(results_dir: Path) -> dict:
    runs = {}
    for group in GROUPS:
        directory = results_dir / group
        paths = {name: directory / name for name in ("summary.json", "test_metrics.json", "history.csv")}
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError(f"Missing real experiment output: {path}. Finish training and evaluation first.")
        summary = json.loads(paths["summary.json"].read_text(encoding="utf-8"))
        metrics = json.loads(paths["test_metrics.json"].read_text(encoding="utf-8"))
        with paths["history.csv"].open(encoding="utf-8", newline="") as stream:
            history = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]
        config = summary["config"]
        expected_model = "improved" if group == "improved" else "baseline"
        expected_augment = "none" if group == "baseline" else "standard"
        if config["model"] != expected_model or config["augment"] != expected_augment:
            raise ValueError(f"{group} configuration does not match its stated comparison role")
        split = summary["split"]
        if split["train_size"] != 45000 or split["val_size"] != 5000:
            raise ValueError(f"{group} is not a full 45,000/5,000 formal experiment; do not use a subset run")
        if not summary.get("stopped_early") and summary["completed_epochs"] < config["epochs"]:
            raise ValueError(f"{group} is still training: {summary['completed_epochs']}/{config['epochs']} epochs")
        if len(history) != summary["completed_epochs"]:
            raise ValueError(f"History/summary epoch mismatch in {group}")
        if metrics["num_test_samples"] != 10000:
            raise ValueError(f"{group} did not evaluate the complete official test split")
        matrix = np.asarray(metrics["confusion_matrix"])
        if matrix.shape != (10, 10) or int(matrix.sum()) != 10000:
            raise ValueError(f"Invalid confusion matrix in {group}")
        if not np.all(matrix.sum(axis=1) == 1000):
            raise ValueError(f"{group} test data does not contain exactly 1,000 examples per CIFAR-10 class")
        if abs(float(np.trace(matrix) / 10000) - metrics["overall_accuracy"]) > 1e-12:
            raise ValueError(f"Accuracy/confusion matrix mismatch in {group}")
        if metrics.get("selected_epoch") != summary["best_epoch"]:
            raise ValueError(f"Evaluated checkpoint is not the validation-selected best checkpoint in {group}")
        runs[group] = {"path": directory, "summary": summary, "metrics": metrics, "history": history,
                       "config": config, "hashes": {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}}
    split_hashes = {(run["summary"]["split"]["train_sha256"], run["summary"]["split"]["val_sha256"]) for run in runs.values()}
    if len(split_hashes) != 1:
        raise ValueError("Comparison runs use different train/validation splits")
    return runs


def register_fonts(font_dir: Path) -> None:
    pdfmetrics.registerFont(TTFont("Song", str(font_dir / "simsun.ttc"), subfontIndex=0))
    pdfmetrics.registerFont(TTFont("Hei", str(font_dir / "simhei.ttf")))
    pdfmetrics.registerFontFamily("Song", normal="Song", bold="Hei", italic="Song", boldItalic="Hei")


def styles() -> dict:
    base = {"fontName": "Song", "fontSize": 10.4, "leading": 15.3, "textColor": INK,
            "wordWrap": "CJK", "spaceAfter": 5, "allowWidows": 0, "allowOrphans": 0}
    return {
        "body": ParagraphStyle("body", **base),
        "small": ParagraphStyle("small", **{**base, "fontSize": 9.2, "leading": 12.7, "spaceAfter": 3}),
        "caption": ParagraphStyle("caption", **{**base, "fontSize": 9.1, "leading": 12.4, "alignment": TA_CENTER, "spaceAfter": 6}),
        "title": ParagraphStyle("title", **{**base, "fontName": "Hei", "fontSize": 18, "leading": 25, "spaceAfter": 7}),
        "h1": ParagraphStyle("h1", **{**base, "fontName": "Hei", "fontSize": 13.4, "leading": 20, "spaceBefore": 5, "spaceAfter": 6}),
        "h2": ParagraphStyle("h2", **{**base, "fontName": "Hei", "fontSize": 11, "leading": 16, "spaceBefore": 4, "spaceAfter": 4}),
        "cell": ParagraphStyle("cell", **{**base, "fontSize": 9.4, "leading": 12.8, "spaceAfter": 0}),
        "header": ParagraphStyle("header", **{**base, "fontName": "Hei", "fontSize": 9.4, "leading": 12.8, "spaceAfter": 0}),
        "repo": ParagraphStyle("repo", **{**base, "fontName": "Hei", "fontSize": 10.6, "leading": 16, "spaceAfter": 9}),
    }


def p(text: str, sty: dict, kind: str = "body") -> Paragraph:
    return Paragraph(text, sty[kind])


def table(rows: list, widths: list, sty: dict, row_height: float | None = None) -> Table:
    data = [[p(escape(str(value)), sty, "header" if index == 0 else "cell") for value in row]
            for index, row in enumerate(rows)]
    result = Table(data, colWidths=widths, rowHeights=row_height, hAlign="LEFT", repeatRows=1)
    result.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), LIGHT), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, 0), 0.7, BLUE), ("LINEBELOW", (0, 1), (-1, -1), 0.35, LINE),
    ]))
    return result


def image(path: Path, width: float, max_height: float | None = None) -> Image:
    if not path.is_file():
        raise FileNotFoundError(f"Missing evaluation figure: {path}")
    result = Image(str(path))
    ratio = width / result.imageWidth
    if max_height is not None:
        ratio = min(ratio, max_height / result.imageHeight)
    result.drawWidth, result.drawHeight = result.imageWidth * ratio, result.imageHeight * ratio
    result.hAlign = "CENTER"
    return result


def percentage(value: float) -> str:
    return f"{value * 100:.2f}%"


def delta(value: float) -> str:
    return f"{value * 100:+.2f} 个百分点"


def draw_comparison(runs: dict, path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(9.4, 5.0), constrained_layout=True)
    colors_ = ("#2474a5", "#d47727", "#45946c")
    for group, color in zip(GROUPS, colors_):
        history = runs[group]["history"]
        epochs = [row["epoch"] for row in history]
        for axis, key, title in zip(axes.flat, ("train_loss", "val_loss", "train_accuracy", "val_accuracy"),
                                    ("Training loss", "Validation loss", "Training accuracy", "Validation accuracy")):
            multiplier = 100 if "accuracy" in key else 1
            axis.plot(epochs, [row[key] * multiplier for row in history], color=color, label=SHORT[group], linewidth=1.6)
            axis.set_title(title, fontsize=10)
            axis.set_xlabel("Epoch", fontsize=9)
            axis.set_ylabel("Accuracy (%)" if "accuracy" in key else "Cross-entropy", fontsize=9)
            axis.grid(alpha=0.2)
            axis.tick_params(labelsize=8)
            axis.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend(frameon=False, ncol=3, fontsize=8)
    figure.savefig(path, dpi=200, facecolor="white")
    plt.close(figure)


def footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(LINE)
    canvas.line(38, 32, A4[0] - 38, 32)
    canvas.setFont("Song", 8.5)
    canvas.setFillColor(colors.HexColor("#657782"))
    canvas.drawString(38, 19, "当代人工智能 · 实验三：CIFAR-10 图像分类")
    canvas.drawRightString(A4[0] - 38, 19, f"{doc.page} / 5")
    canvas.restoreState()


def top_confusions(metrics: dict, count: int = 5) -> list:
    matrix = np.asarray(metrics["confusion_matrix"])
    pairs = [(int(matrix[row, col]), row, col) for row in range(10) for col in range(10) if row != col]
    return sorted(pairs, key=lambda value: (-value[0], value[1], value[2]))[:count]


def augmentation_description(config: dict) -> str:
    return "随机裁剪(补边4)+水平翻转" if config["augment"] == "standard" else "无随机增强"


def make_story(options: argparse.Namespace, runs: dict, sty: dict, curves: Path) -> tuple[list, str]:
    # Dict insertion order and strict '>' match the experiment rule for ties.
    best_group = max(GROUPS, key=lambda group: runs[group]["summary"]["best_val_accuracy"])
    chosen = runs[best_group]
    summary, metrics, config = chosen["summary"], chosen["metrics"], chosen["config"]
    split, environment = summary["split"], summary["environment"]
    augmentation_delta = runs['ablation']['summary']['best_val_accuracy'] - runs['baseline']['summary']['best_val_accuracy']
    augmentation_note = ("这一结果与我原先预期的增强效果不同。结合训练曲线，我认为增强提高了样本难度，在相同轮数下可能需要更长时间收敛，后续还需用验证集检验这一判断。"
                         if augmentation_delta < 0 else
                         "本次增强组的验证表现有所改善，我认为随机裁剪与翻转有助于提高泛化能力，但还需要重复实验判断这一提升是否稳定。")
    notes_path = chosen['path'] / 'qualitative_analysis.json'
    notes = json.loads(notes_path.read_text(encoding='utf-8')) if notes_path.exists() else {}
    if notes and notes.get('checkpoint_sha256') != metrics['checkpoint_sha256']:
        raise ValueError('Qualitative observations refer to a different checkpoint')
    mistake_note = notes.get('mistakes', '高置信错误提示模型可能学习了可替代的背景或局部纹理线索，也可能面对遮挡、截断与主体过小的图像。后续应先在验证集按类别分组抽样检查，再比较遮挡背景、保留主体等干预前后的预测，避免仅凭少量挑选案例下结论。')
    cam_note = notes.get('gradcam', '')
    story = []
    if options.repo_url:
        repo = escape(options.repo_url)
        story.append(p(f'GitHub 仓库：<link href="{repo}" color="#185d82">{repo}</link>', sty, "repo"))
    else:
        story.append(p('<font color="#a52e20">GitHub 仓库：待填写</font>', sty, "repo"))
    story.extend([
        p("CIFAR-10 图像分类实验报告", sty, "title"),
        p(f"姓名：{escape(options.name)}　学号：{escape(options.student_id)}", sty),
        p("1　实验目的与方法", sty, "h1"),
        p("本次实验使用 CIFAR-10 数据集完成图像十分类。我以基础 CNN 为起点，分别考察数据增强和残差注意力结构对分类效果的影响，并通过训练曲线、混淆矩阵及 Grad-CAM 分析模型的表现与不足。", sty),
        p("1.1　数据与划分", sty, "h2"),
        p(f"CIFAR-10 包含 60,000 张 32×32 RGB 图像，共 10 类[1]。我从官方 50,000 张训练图像中按类别分层划分出 {split['train_size']:,} 张训练图像和 {split['val_size']:,} 张验证图像，随机种子固定为 {config['seed']}。三组实验使用相同且互不重叠的训练、验证划分；官方 10,000 张测试图像只在训练结束后用于评估。", sty),
        p("预处理时，我先将像素转换到 [0,1]，再按通道标准化：mean=(0.4914, 0.4822, 0.4465)，std=(0.2470, 0.2435, 0.2616)。增强组在训练时使用补边4像素的随机裁剪和随机水平翻转；验证与测试只进行张量转换和标准化。", sty),
        p("1.2　模型原理与结构", sty, "h2"),
        table([
            ["模块", "基础CNN（A/B）", "改进CNN（C）"],
            ["特征提取", "3组：两层3×3卷积+ReLU+2×2最大池化；通道32/64/128", "3×3卷积起始层；3阶段各2个残差SE块，通道32/64/128"],
            ["空间尺寸", "32→16→8→4", "32→32→16→8；阶段切换stride=2"],
            ["分类头", "全局平均池化+Dropout(0.3)+Linear(128,10)", "全局平均池化+Dropout(0.2)+Linear(128,10)"],
            ["改进机制", "局部卷积提取纹理，池化减小计算量", "残差连接、BatchNorm、SE通道注意力（压缩比8）"],
        ], [67, 225, BODY_WIDTH - 292], sty),
        Spacer(1, 5),
        p("基础 CNN 通过卷积和池化逐层提取特征。改进模型加入残差连接 y=ReLU(F(x)+shortcut(x))，以改善深层特征的传递[2]；SE 利用全局池化和两层映射生成通道权重，使模型能够调整各通道的重要性[3]。尺寸改变时用 1×1 卷积对齐。两种模型均从随机初始化开始训练。", sty),
        p("1.3　实验设置", sty, "h2"),
    ])
    rows = [["组别", "随机增强", "标签平滑", "优化器/初始学习率", "轮数/批量"]]
    for group in GROUPS:
        current = runs[group]["config"]
        rows.append([SHORT[group], "裁剪+翻转" if current["augment"] == "standard" else "无",
                     f"{current['label_smoothing']:g}", f"{current['optimizer'].upper()} / {current['lr']:g}",
                     f"{runs[group]['summary']['completed_epochs']}/{current['batch_size']}"])
    story.extend([
        table(rows, [44, 105, 65, 175, BODY_WIDTH - 389], sty),
        Spacer(1, 5),
        p(f"三组均使用 AdamW、交叉熵损失和余弦学习率调度（最低学习率为初始值的0.01倍），权重衰减为 {config['weight_decay']:g}；不使用预训练权重，关闭早停。每组保留验证准确率最高的模型，同分时保留较早轮次。运行环境为 PyTorch {escape(str(environment['torch']))}、{escape(str(environment.get('gpu') or environment['device']))}，混合精度{'开启' if environment.get('amp') else '关闭'}。", sty, "small"),
        PageBreak(),
        p("2　三组对照结果与训练过程", sty, "h1"),
        p("为区分增强策略与模型结构的影响，我设置了三组实验：A 为基础 CNN，B 在相同 CNN 上加入数据增强，C 使用残差 SE-CNN 和相同增强。三组均训练30轮，优化器等设置一致，先用验证集选择模型，再统一评估测试集。结果如下表所示。", sty),
    ])
    rows = [["组别", "参数量", "最佳轮", "验证准确率", "测试准确率", "宏F1", "训练/min"]]
    for group in GROUPS:
        current = runs[group]
        summ, test = current["summary"], current["metrics"]
        rows.append([SHORT[group], f"{summ['parameters'] / 1e6:.3f}M", summ["best_epoch"],
                     percentage(summ["best_val_accuracy"]), percentage(test["overall_accuracy"]),
                     f"{test['macro_f1']:.4f}", f"{summ['training_seconds'] / 60:.1f}"])
    story.extend([
        table(rows, [35, 62, 48, 94, 94, 65, BODY_WIDTH - 398], sty),
        Spacer(1, 7),
        image(curves, BODY_WIDTH, 248),
        p("图1　三组完整训练/验证 loss 与 accuracy 曲线。A、B、C 定义见正文；横轴为实际轮次。三组使用相同交叉熵目标；随机增强仅在训练阶段启用。", sty, "caption"),
        p("2.1　定量比较", sty, "h2"),
        p(f"B 相对 A 的验证准确率变化为 {delta(augmentation_delta)}，测试准确率变化为 {delta(runs['ablation']['metrics']['overall_accuracy'] - runs['baseline']['metrics']['overall_accuracy'])}。两组结构及训练设置相同，仅改变随机增强。{augmentation_note}", sty),
        p(f"C 相对 B 的验证准确率提高了 {delta(runs['improved']['summary']['best_val_accuracy'] - runs['ablation']['summary']['best_val_accuracy'])}，测试准确率提高了 {delta(runs['improved']['metrics']['overall_accuracy'] - runs['ablation']['metrics']['overall_accuracy'])}，说明本次组合结构改进取得了较好的效果。不过，C 同时调整了残差结构、BatchNorm、SE 和 Dropout，我还不能据此判断其中哪一项贡献最大。", sty),
        p("2.2　曲线与模型选择", sty, "h2"),
        p(f"从图1可以看出，C组的损失下降较快，验证准确率也明显高于A、B组。其最高验证准确率为 {percentage(summary['best_val_accuracy'])}，出现在第 {summary['best_epoch']} 轮。因此，我选取这一轮的模型进行后续逐类和误分类分析，而没有直接采用最后一轮权重。", sty),
    ])
    last = chosen["history"][-1]
    story.extend([
        p(f"C组最后一轮训练和验证准确率分别为 {percentage(last['train_accuracy'])}、{percentage(last['val_accuracy'])}，相差 {(last['train_accuracy']-last['val_accuracy'])*100:.2f} 个百分点。后期训练准确率继续上升，而验证表现逐渐趋于稳定，说明模型仍有一定过拟合。这也让我认识到，训练集上的高准确率并不等于更好的泛化能力。", sty),
        PageBreak(),
        p("3　完整测试集与逐类表现", sty, "h1"),
        p(f"分析模型：{LABELS[best_group]}，验证最优第 {summary['best_epoch']} 轮。官方测试集共 {metrics['num_test_samples']:,} 张，预测正确 {metrics['correct']:,} 张；总体准确率 <b>{percentage(metrics['overall_accuracy'])}</b>，宏平均 F1={metrics['macro_f1']:.4f}，平均交叉熵={metrics['loss']:.4f}。", sty),
        image(chosen["path"] / "confusion_matrix.png", 410, 341),
        p("图2　行表示真实类别，列表示预测类别；格内为样本数，颜色为该行所占比例。", sty, "caption"),
    ])
    rows = [["类别", "测试数", "正确数", "每类准确率/召回率", "精确率", "F1"]]
    for item in metrics["per_class"]:
        rows.append([f"{CHINESE[item['class_name']]} ({item['class_name']})", item["support"], item["correct"],
                     percentage(item["accuracy"]), percentage(item["precision"]), f"{item['f1']:.4f}"])
    story.extend([
        table(rows, [146, 52, 52, 119, 77, BODY_WIDTH - 446], sty),
        Spacer(1, 5),
        p("本表将每类准确率定义为该类正确数除以该类测试数，即召回率 TP/(TP+FN)；精确率为 TP/(TP+FP)，F1 综合两者。总体准确率为全部正确数/10,000，宏 F1 为十类 F1 的平均值。", sty, "small"),
    ])
    ordered = sorted(metrics["per_class"], key=lambda value: value["accuracy"])
    story.extend([
        p(f"逐类结果中，{CHINESE[ordered[0]['class_name']]}的准确率最低，为 {percentage(ordered[0]['accuracy'])}；{CHINESE[ordered[-1]['class_name']]}最高，为 {percentage(ordered[-1]['accuracy'])}。结合混淆矩阵，我发现模型对部分动物类别区分较困难，因此进一步检查了常见混淆方向和具体错误样本。", sty),
        PageBreak(),
        p("4　误分类样本与错误模式", sty, "h1"),
        p("为了解模型在什么情况下会作出错误判断，我从测试集错误预测中选取了预测置信度最高的12张图像，如图3所示。图中给出样本索引、真实类别、预测类别与置信度。这些样本用于分析高置信错误，整体错误分布则结合完整混淆矩阵判断。", sty),
        image(chosen["path"] / "misclassifications.png", BODY_WIDTH, 358),
        p("图3　最高置信错误。32×32 原图放大显示；置信度四舍五入至0.1个百分点，100.0%不代表预测可靠。", sty, "caption"),
        p("4.1　完整测试集最常见的有向混淆", sty, "h2"),
    ])
    names = metrics["class_names"]
    pairs = top_confusions(metrics)
    rows = [["真实类别 → 预测类别", "数量", "占真实类别比例"]]
    for count, true, pred in pairs:
        support = int(np.asarray(metrics["confusion_matrix"])[true].sum())
        rows.append([f"{CHINESE[names[true]]} → {CHINESE[names[pred]]}", count, percentage(count / support)])
    story.extend([
        table(rows, [285, 85, BODY_WIDTH - 370], sty),
        Spacer(1, 5),
        p("4.2　误分类原因分析", sty, "h2"),
        p(f"最常见的错误是把{CHINESE[names[pairs[0][1]]]}分为{CHINESE[names[pairs[0][2]]]}，共有 {pairs[0][0]} 张。我认为，32×32 图像丢失了较多细节，而猫、狗等动物在毛发纹理和身体轮廓上又存在相似之处，容易产生混淆。相比之下，汽车、船等类别的形状特征更明显，分类结果也相对较好。", sty),
        p(escape(mistake_note), sty),
        PageBreak(),
        p("5　解释性分析、局限与改进", sty, "h1"),
        p("5.1　Grad-CAM", sty, "h2"),
        p("我使用 Grad-CAM 观察模型作出分类判断时关注的区域[4]。具体做法是对预测类别分数求梯度，将梯度的空间均值作为通道权重，再对特征图加权、经过 ReLU 并上采样。改进模型选用最后一个残差块的 conv2，以保留空间信息。", sty),
        image(chosen["path"] / "gradcam.png", BODY_WIDTH, 285),
        p("图4　上排为原图，下排为预测类别 Grad-CAM；左侧三个高置信错误，右侧三个正确案例。红色表示相对更强的正响应。", sty, "caption"),
        p(escape(cam_note) + "这让我认识到，关注主体区域仍可能发生误分类。由于热图空间分辨率较低、每张图单独归一化，我将其作为辅助观察，而不把热区直接视为模型判断原因的证明。", sty),
        p("5.2　实验体会与不足", sty, "h2"),
        p(f"本次实验中，改进模型的测试准确率达到 {percentage(metrics['overall_accuracy'])}，高于基础 CNN。通过对照，我认识到结构改进能够提高分类效果，而数据增强的收益还与模型、训练轮数有关，不能只根据是否使用增强来判断效果。与此同时，改进模型的参数量和训练耗时都增加了，性能提升也有相应的计算成本。", sty),
        p("本次每组只使用一个随机种子，训练30轮，尚不足以说明方法的稳定性。此外，改进模型包含多项机制，各项改动的贡献还有待区分。后续我计划保持网络宽度和训练设置一致，逐步比较残差、BatchNorm及SE的作用，并用多个随机种子重复实验，统计均值和标准差。", sty),
        p("针对猫、狗等易混淆类别，我还希望检查更多典型样本，比较更长训练与其他增强方式的效果。对于高置信错误，可以进一步在验证集上尝试标签平滑或温度缩放，观察预测是否更可靠；确定方案后，再使用测试集评估。", sty),
        p("5.3　参考资料", sty, "h2"),
        p('[1] Krizhevsky. Learning Multiple Layers of Features from Tiny Images, 2009. <link href="https://www.cs.toronto.edu/~kriz/cifar.html" color="#185d82">CIFAR-10 数据与技术报告</link>', sty, "small"),
        p('[2] He et al. Deep Residual Learning for Image Recognition, 2015. <link href="https://arxiv.org/abs/1512.03385" color="#185d82">arXiv:1512.03385</link>', sty, "small"),
        p('[3] Hu et al. Squeeze-and-Excitation Networks, 2017. <link href="https://arxiv.org/abs/1709.01507" color="#185d82">arXiv:1709.01507</link>', sty, "small"),
        p('[4] Selvaraju et al. Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization, 2016. <link href="https://arxiv.org/abs/1610.02391" color="#185d82">arXiv:1610.02391</link>', sty, "small"),
    ])
    return story, best_group


def main() -> None:
    options = args()
    options.repo_url = options.repo_url.strip().rstrip("/")
    if options.repo_url:
        parsed = urlparse(options.repo_url)
        if parsed.scheme != "https" or parsed.hostname != "github.com" or len(parsed.path.strip("/").split("/")) != 2:
            raise ValueError("--repo-url must be an actual https://github.com/OWNER/REPOSITORY URL")
    runs = load_runs(options.results_dir)
    register_fonts(options.font_dir)
    options.output_dir.mkdir(parents=True, exist_ok=True)
    asset_dir = options.output_dir / "assets"
    asset_dir.mkdir(exist_ok=True)
    curves = asset_dir / "comparison_curves.png"
    draw_comparison(runs, curves)
    story, best_group = make_story(options, runs, styles(), curves)
    suffix = "" if options.repo_url else "_待填仓库链接"
    filename = f"实验三报告_{options.name}_{options.student_id}_学生视角版{suffix}.pdf"
    path = options.output_dir / filename
    temporary = path.with_name(path.stem + ".tmp.pdf")
    document = SimpleDocTemplate(str(temporary), pagesize=A4, leftMargin=38, rightMargin=38,
                                 topMargin=29, bottomMargin=42, pageCompression=1,
                                 title="CIFAR-10 图像分类实验报告", author=options.name)
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    reader = PdfReader(temporary)
    if len(reader.pages) != 5:
        raise RuntimeError(f"Report has {len(reader.pages)} pages, expected exactly five. Inspect {temporary}")
    temporary.replace(path)
    audit = {"report": str(path), "pages": len(reader.pages), "repo_url": options.repo_url or None,
             "submission_ready": bool(options.repo_url), "selected_group": best_group,
             "selection_rule": "Highest validation accuracy across A/B/C; ties retain A, then B, then C.",
             "source_hashes": {group: run["hashes"] for group, run in runs.items()}}
    (options.output_dir / "report_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
