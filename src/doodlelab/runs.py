"""实验目录：保存、读取、汇总、对照。

目录结构（runs/<时间>_<名字>/）：
    config.json       实验条件（含环境、参数量、耗时、fingerprint）
    history.json      逐轮 loss / accuracy
    evaluations.json  各划分的紧凑评价（不含逐图分数）
    model.pt          权重 + 配置 + 标签 + 预处理约定
    curve.png         训练曲线
    summary.md        可直接贴给 AI 或抄进报告的文字记录
"""
from __future__ import annotations

import json
import platform
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from . import metrics, models, plots
from .config import TrainConfig
from .preprocess import PREPROCESS

CODE_VERSION = "0.3.0"
SPLIT_TITLES = {
    "val_basic": "基础验证集",
    "challenge_original": "挑战：独立原图（配对参照）",
    "challenge_shift": "挑战：右移3、下移2像素",
    "challenge_thick": "挑战：3×3 邻域加粗",
    "challenge_occlusion": "挑战：中央 8×5 像素遮挡",
}


# --------------------------------------------------------------------- 数据类
@dataclass
class Run:
    path: Path
    record: dict
    history: list
    evaluations: dict
    curve: Path | None = None

    @property
    def name(self) -> str:
        return self.record.get("name") or self.path.name

    @property
    def config(self) -> TrainConfig:
        return TrainConfig.from_dict(self.record["config"])

    @property
    def created(self) -> str:
        return self.record.get("created_utc", "")

    @property
    def structure(self) -> str:
        args = self.record["config"]["model_args"]
        return ", ".join(f"{k}={v}" if not isinstance(v, list) else f"{k}={','.join(map(str, v))}"
                         for k, v in args.items())

    def metric(self, split: str = "val_basic", key: str = "accuracy") -> float:
        table = self.evaluations.get(split)
        if table is None:
            raise KeyError(f"{self.path.name} 没有 {split} 的评价，先运行 eval")
        return float(table[key])

    def final(self) -> dict:
        return self.history[-1]


# --------------------------------------------------------------------- 工具
def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _slug(name: str) -> str:
    cleaned = re.sub(r"[\\/:*?\"<>|\s]+", "_", name.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        raise ValueError("实验名不能为空")
    return cleaned[:40]


def new_dir(root: str | Path, name: str) -> Path:
    """按时间戳新建实验目录；重名不覆盖。"""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(root) / "runs" / f"{stamp}_{_slug(name)}"
    suffix = 1
    while path.exists():
        suffix += 1
        path = Path(root) / "runs" / f"{stamp}_{_slug(name)}_{suffix}"
    path.mkdir(parents=True)
    return path


def environment() -> dict:
    """全部转成纯 str/int：权重包里只存 JSON 友好类型，weights_only=True 才能读回。"""
    return dict(
        python=str(platform.python_version()),
        torch=str(torch.__version__),
        numpy=str(np.__version__),
        pillow=str(Image.__version__),
        platform=platform.platform(),
        threads=int(torch.get_num_threads()),
        device="cpu",
    )


# --------------------------------------------------------------------- 保存
def save(root: str | Path, name: str, model: torch.nn.Module, config: TrainConfig,
         history: list[dict], evaluations: dict[str, dict], seconds: float,
         parameter_count: int | None = None) -> Run:
    """写出一整套实验产物；evaluations 里是完整评价，落盘前会被 compact。"""
    path = new_dir(root, name)
    config = config.validate()
    parameter_count = parameter_count or sum(p.numel() for p in model.parameters())
    record = dict(
        name=name,
        tag=config.tag,
        created_utc=datetime.now(timezone.utc).isoformat(),
        code_version=CODE_VERSION,
        config=config.to_dict(),
        fingerprint=config.fingerprint(),
        parameter_count=int(parameter_count),
        seconds=float(seconds),
        environment=environment(),
        training_policy="last_epoch; validation_visible; no_independent_test",
        preprocess=PREPROCESS,
        data=dict(train=config.train_split, validation="val_basic"),
    )
    compacted = {split: metrics.compact(ev) for split, ev in evaluations.items()}
    _write_json(path / "config.json", record)
    _write_json(path / "history.json", history)
    _write_json(path / "evaluations.json", compacted)
    torch.save(
        dict(state_dict=model.state_dict(), config=record["config"], labels=model.doodle_labels,
             preprocess=record["preprocess"], code_version=CODE_VERSION, record=record),
        path / "model.pt",
    )
    plots.plot_history(history, path / "curve.png")
    run = Run(path=path, record=record, history=history, evaluations=compacted,
              curve=path / "curve.png")
    (path / "summary.md").write_text(summary_markdown(run), encoding="utf-8")
    return run


def save_plot_evaluations(run: Run, evaluations: dict[str, dict]) -> list[Path]:
    """把某个划分的混淆矩阵/错误样例落盘，供报告引用。"""
    outputs = []
    for split, ev in evaluations.items():
        if "confusion" in ev:
            outputs.append(plots.plot_confusion(ev, ev_labels(run), run.path / f"confusion_{split}.png"))
    return outputs


def ev_labels(run: Run) -> list:
    return run.record.get("labels") or load_checkpoint(run.path)["labels"]


# --------------------------------------------------------------------- 读取
def read(path: str | Path) -> Run:
    path = Path(path)
    if path.is_file() and path.name == "model.pt":
        path = path.parent
    config_file = path / "config.json"
    if not config_file.exists():
        raise ValueError(f"{path} 不是新版实验目录（缺 config.json）；旧 notebook 实验见 legacy/")
    record = json.loads(config_file.read_text(encoding="utf-8"))
    record.setdefault("labels", load_checkpoint(path)["labels"])
    history = json.loads((path / "history.json").read_text(encoding="utf-8"))
    evaluations = json.loads((path / "evaluations.json").read_text(encoding="utf-8"))
    curve = path / "curve.png"
    return Run(path=path, record=record, history=history, evaluations=evaluations,
               curve=curve if curve.exists() else None)


def list_runs(root: str | Path = ".") -> tuple[list[Run], list[Path]]:
    """返回 (可读实验, 被跳过的旧目录)，按创建时间排序。"""
    base = Path(root) / "runs"
    runs, skipped = [], []
    if not base.exists():
        return runs, skipped
    for entry in sorted(base.iterdir()):
        if not entry.is_dir():
            continue
        try:
            runs.append(read(entry))
        except (ValueError, FileNotFoundError, KeyError):
            skipped.append(entry)
    runs.sort(key=lambda run: (run.created, run.path.name))
    return runs, skipped


def resolve(root: str | Path, spec: str | Path) -> Run:
    """允许写完整路径、目录名、实验名，或目录名的唯一片段。"""
    candidate = Path(spec)
    if candidate.is_dir():
        return read(candidate)
    base = Path(root) / "runs"
    if (base / str(spec)).is_dir():
        return read(base / str(spec))
    if not base.exists():
        raise ValueError(f"找不到实验 {spec!r}（{base} 不存在）")
    entries = [p for p in sorted(base.iterdir()) if p.is_dir()]
    # 先按实验名精确匹配，再退回目录名片段，避免「基线」被「残差基线」抢走。
    exact = []
    for path in entries:
        try:
            if read(path).name == str(spec):
                exact.append(path)
        except (ValueError, FileNotFoundError, KeyError):
            continue
    if len(exact) == 1:
        return read(exact[0])
    matches = [p for p in entries if str(spec) in p.name] + [p for p in exact if p not in entries]
    matches = list(dict.fromkeys(matches))
    if len(matches) == 1:
        return read(matches[0])
    if not matches:
        raise ValueError(f"找不到实验 {spec!r}")
    raise ValueError(f"{spec!r} 匹配到多个实验：" + "、".join(p.name for p in matches))


def load_checkpoint(path: str | Path) -> dict:
    path = Path(path)
    file = path / "model.pt" if path.is_dir() else path
    return torch.load(file, map_location="cpu", weights_only=True)


def load_model(path: str | Path) -> tuple[torch.nn.Module, TrainConfig, list, dict]:
    """加载实验目录 → (eval 模式模型, 配置, 标签, 预处理约定)。"""
    checkpoint = load_checkpoint(path)
    config = TrainConfig.from_dict(checkpoint["config"])
    model = models.build(config.model, config.resolved_args(), len(checkpoint["labels"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.doodle_labels = checkpoint["labels"]
    model.doodle_preprocess = checkpoint["preprocess"]
    return model, config, checkpoint["labels"], checkpoint["preprocess"]


# --------------------------------------------------------------------- 文字
def summary_markdown(run: Run) -> str:
    record, config = run.record, run.config
    env = record.get("environment", {})
    last = run.final()
    lines = [
        f"# 实验：{run.name}",
        "",
        f"- 时间（UTC）：{record.get('created_utc', '未记录')}",
        f"- 备注：{record.get('tag') or '（无）'}",
        f"- 模型：{config.model}（{run.structure}）；参数量 {record['parameter_count']:,}",
        f"- 数据：{config.train_split} 训练，val_basic 验证",
        f"- 学习率 {config.lr}；轮数 {config.epochs}；批大小 {config.batch_size}；"
        f"weight_decay {config.weight_decay}；label_smoothing {config.label_smoothing}",
        f"- 优化器 {config.optimizer}" + (f"（momentum {config.momentum}）" if config.optimizer == "sgd" else ""),
        f"- dropout {config.resolved_args().get('dropout', 0)}；随机偏移 {config.augment_shift} 像素；种子 {config.seed}",
        f"- 环境：Python {env.get('python', '?')} / PyTorch {env.get('torch', '?')} / "
        f"NumPy {env.get('numpy', '?')} / 线程 {env.get('threads', '?')} / CPU",
        f"- 训练耗时：{record['seconds']:.2f} 秒；程序版本 {record['code_version']}",
        "",
        "## 逐轮结果",
        "",
        "| 轮次 | 训练 loss | 验证 loss | 训练准确率 | 验证准确率 |",
        "|---:|---:|---:|---:|---:|",
    ]
    lines += [f"| {row['epoch']} | {row['train_loss']:.4f} | {row['val_loss']:.4f} | "
              f"{row['train_accuracy']:.2%} | {row['val_accuracy']:.2%} |" for row in run.history]
    lines += ["", f"最后一轮：训练 {last['train_accuracy']:.2%} / 验证 {last['val_accuracy']:.2%}"]
    if run.evaluations:
        lines += ["", "## 各划分评价", ""]
        for split, ev in run.evaluations.items():
            lines.append(f"### {SPLIT_TITLES.get(split, split)}")
            lines.append("")
            lines.append(f"张数 {ev['count']}；准确率 {ev['accuracy']:.2%}；loss {ev['loss']:.4f}")
            lines.append("")
            lines.append("| 类别 | 正确 / 总数 | 准确率 |")
            lines.append("|---|---:|---:|")
            lines += [f"| {row['zh']}（{row['en']}） | {row['correct']} / {row['count']} | "
                      f"{(row['accuracy'] or 0):.1%} |" for row in ev["per_class"]]
            lines.append("")
            if ev.get("errors_preview"):
                labels = {x["id"]: x["zh"] for x in ev["per_class"]}
                lines.append("前几条错误：")
                lines += [f"- 行 {row['row']}：标注 {labels[row['true']]}，预测 "
                          f"{labels[row['predicted']]}，最高分数 {row['score']:.3f}"
                          for row in ev["errors_preview"]]
                lines.append("")
    lines += [
        "",
        "## 输入约定",
        "",
        f"28×28、黑底亮线、取值 0—1；图案最长边占 {PREPROCESS['fit_size']} 格；"
        f"找主体阈值 {PREPROCESS['threshold']}；外部处理版本 {PREPROCESS['version']}。",
        "",
        "结果来自公开验证集，不是独立测试成绩；保存最后一轮权重，不做早停挑选。",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------- 对照
def compare_rows(runs: list[Run], split: str = "val_basic") -> list[dict]:
    rows = []
    for run in runs:
        if split not in run.evaluations:
            continue
        config = run.config
        args = config.resolved_args()
        rows.append(dict(
            name=run.name,
            path=run.path.name,
            model=config.model,
            structure="; ".join(f"{k}={v}" for k, v in args.items()),
            lr=config.lr,
            epochs=config.epochs,
            params=run.record["parameter_count"],
            seconds=run.record["seconds"],
            accuracy=float(run.evaluations[split]["accuracy"]),
            loss=float(run.evaluations[split]["loss"]),
            tag=run.record.get("tag") or "",
        ))
    return rows


def with_deltas(rows: list[dict], baseline: str | None) -> list[dict]:
    """给每行补上与基线的差值（准确率绝对差、loss 差、参数量倍数）。"""
    if not rows:
        return rows
    base = rows[0]
    if baseline:
        matched = [row for row in rows if baseline in (row["name"], row["path"])]
        if not matched:
            raise ValueError(f"基线 {baseline!r} 不在候选实验里")
        base = matched[0]
    for row in rows:
        row["is_baseline"] = row is base
        row["d_accuracy"] = row["accuracy"] - base["accuracy"]
        row["d_loss"] = row["loss"] - base["loss"]
        row["param_ratio"] = row["params"] / base["params"] if base["params"] else None
    return rows


def write_compare(root: str | Path, rows: list[dict], split: str = "val_basic",
                  metric: str = "accuracy") -> dict[str, Path]:
    """写 runs/compare.md、compare.csv、compare.png。"""
    base = Path(root) / "runs"
    base.mkdir(parents=True, exist_ok=True)
    header = ["实验", "模型", "结构", "lr", "轮数", "参数量", "秒", f"{split}_准确率", "Δ准确率", "备注"]
    md = [f"# 实验对照（{split}）", "",
          "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        md.append("| " + " | ".join([
            row["name"] + ("（基线）" if row.get("is_baseline") else ""),
            row["model"], row["structure"], f"{row['lr']:g}", str(row["epochs"]),
            f"{row['params']:,}", f"{row['seconds']:.1f}",
            f"{row['accuracy']:.2%}", f"{row['d_accuracy']:+.2%}", row["tag"],
        ]) + " |")
    md += ["", "Δ准确率 = 相对基线的绝对差；负值表示更差。验证集参与分析，不是独立测试成绩。"]
    md_path = base / "compare.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    csv_path = base / "compare.csv"
    csv_header = ["name", "model", "structure", "lr", "epochs", "params", "seconds",
                  "accuracy", "loss", "d_accuracy", "param_ratio", "tag", "path"]
    csv_lines = [",".join(csv_header)]
    for row in rows:
        csv_lines.append(",".join([
            f'"{row["name"]}"', row["model"], f'"{row["structure"]}"', f"{row['lr']:g}",
            str(row["epochs"]), str(row["params"]), f"{row['seconds']:.2f}",
            f"{row['accuracy']:.6f}", f"{row['loss']:.6f}", f"{row['d_accuracy']:+.6f}",
            f"{row['param_ratio']:.3f}" if row["param_ratio"] else "",
            f'"{row["tag"]}"', row["path"],
        ]))
    csv_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    png_path = plots.plot_compare(rows, metric, base / "compare.png") if rows else None
    return dict(markdown=md_path, csv=csv_path, png=png_path)
