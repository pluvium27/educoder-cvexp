"""出图：训练曲线、混淆矩阵、样例、错误预览、预测对照、实验对照条形图。

固定用 Agg 后端，不依赖 notebook。图表标签一律英文；对照图的实验名是中文，
因此有 CJK 字体时用名字，没有时退化成 #序号（序号 = compare.md 的行序），避免出方块。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

CJK_FONTS = (
    "Noto Sans CJK SC", "Noto Sans SC", "Source Han Sans SC", "Source Han Sans CN",
    "WenQuanYi Zen Hei", "WenQuanYi Micro Hei", "Microsoft YaHei", "PingFang SC",
    "Heiti SC", "SimHei", "Arial Unicode MS",
)


def _enable_cjk_font() -> bool:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in CJK_FONTS:
        if name in available:
            matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return True
    return False


HAS_CJK_FONT = _enable_cjk_font()


def _save(fig, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_history(history: list[dict], path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    epochs = [row["epoch"] for row in history]
    for key, label in (("train", "train"), ("val", "validation")):
        axes[0].plot(epochs, [row[f"{key}_loss"] for row in history], label=label)
        axes[1].plot(epochs, [row[f"{key}_accuracy"] for row in history], label=label)
    for ax, title in zip(axes, ["Loss (lower is better)", "Accuracy"]):
        ax.set(xlabel="Epoch", title=title)
        ax.legend()
        ax.grid(alpha=0.2)
    axes[1].set_ylim(0, 1)
    fig.tight_layout()
    return _save(fig, path)


def plot_confusion(evaluation: dict, labels: list, path) -> Path:
    cm = np.asarray(evaluation["confusion"])
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, cmap="Blues")
    ticks = range(len(labels))
    ax.set(xticks=ticks, yticks=ticks,
           xticklabels=[x["en"] for x in labels], yticklabels=[x["en"] for x in labels],
           xlabel="Predicted", ylabel="True", title="Confusion counts")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max() * 0.55 else "black")
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    fig.tight_layout()
    return _save(fig, path)


def plot_samples(split, path, per_class: int = 4) -> Path:
    labels = split.labels
    fig, axes = plt.subplots(len(labels), per_class,
                             figsize=(per_class * 1.5, len(labels) * 1.25), squeeze=False)
    for index, label in enumerate(labels):
        rows = np.flatnonzero(split.y.numpy() == index)[:per_class]
        for j, ax in enumerate(axes[index]):
            ax.axis("off")
            if j < len(rows):
                ax.imshow(split.x[rows[j], 0], cmap="gray", vmin=0, vmax=1)
            if j == 0:
                ax.set_title(f"{index}: {label['en']}", fontsize=9)
    fig.tight_layout()
    return _save(fig, path)


def plot_errors(split, evaluation: dict, path, limit: int = 12) -> Path | None:
    rows = evaluation["error_indices"][:limit]
    if not rows:
        return None
    cols = 4
    height = int(np.ceil(len(rows) / cols))
    fig, axes = plt.subplots(height, cols, figsize=(8, 2 * height), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, index in zip(axes.flat, rows):
        truth = evaluation["true_labels"][index]
        predicted = evaluation["predictions"][index]
        ax.imshow(split.x[index, 0], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"row {index}: {split.labels[truth]['en']}\n-> {split.labels[predicted]['en']}",
                     fontsize=8)
    fig.tight_layout()
    return _save(fig, path)


def plot_prediction(prediction: dict, path) -> Path:
    prepared = prediction["prepared"]
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    axes[0].imshow(prepared["original"])
    axes[0].set_title("Original input")
    axes[1].imshow(prepared["processed"], cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Model sees: 28 x 28")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    return _save(fig, path)


def plot_compare(rows: list[dict], metric: str, path) -> Path:
    """对照实验条形图：横轴为实验，纵轴为主指标。"""
    names = [row["name"] for row in rows]
    values = [row[metric] for row in rows]
    labels = names if HAS_CJK_FONT else [f"#{i + 1}" for i in range(len(rows))]
    fig, ax = plt.subplots(figsize=(max(6, len(rows) * 0.9), 3.4))
    bars = ax.bar(range(len(rows)), values, color="#2f7f9e")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(metric)
    ax.set_title(f"Comparison by {metric}")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}",
                ha="center", va="bottom", fontsize=7)
    if not HAS_CJK_FONT:
        ax.set_xlabel("#n 对应 compare.md 第 n 行（本机缺中文字体，改用序号）", fontsize=8)
    fig.tight_layout()
    return _save(fig, path)
