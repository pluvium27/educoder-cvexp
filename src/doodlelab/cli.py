"""命令行入口：`doodlelab <命令> --help` 查看用法。

命令：train / eval / runs / show / compare / predict / models / splits
输出面向人读，表格式对齐；机器可读的产物都落在 runs/ 下的 JSON/CSV。
"""
from __future__ import annotations

import argparse
import sys
import unicodedata
from pathlib import Path

from . import models, plots, runs
from .config import TrainConfig
from .data import DataStore
from .preprocess import predict_image
from .training import evaluate_run, train as run_training


# --------------------------------------------------------------------- 输出
def _width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def _pad(text: str, width: int, right: bool = False) -> str:
    fill = " " * max(0, width - _width(text))
    return fill + text if right else text + fill


def table(headers: list[str], rows: list[list[str]], right: set[int] | None = None) -> str:
    """等宽表（按显示宽度对齐中文）。right 指定右对齐的列号。"""
    right = right or set()
    cells = [[str(v) for v in row] for row in rows]
    widths = [max([_width(h), *(_width(r[i]) for r in cells)]) for i, h in enumerate(headers)]
    line = lambda values: "  ".join(  # noqa: E731
        _pad(v, widths[i], right=i in right) for i, v in enumerate(values)).rstrip()
    head = line(headers)
    body = [line(row) for row in cells]
    return "\n".join([head, "-" * _width(head), *body])


def _parse_pairs(items: list[str] | None, what: str) -> dict:
    out: dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"{what} 需要 键=值 形式，收到 {item!r}")
        key, value = item.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def _parse_crop(text: str | None):
    if not text:
        return None
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 4:
        raise ValueError("--crop 需要 左,上,右,下 四个整数")
    return tuple(int(p) for p in parts)


# --------------------------------------------------------------------- 命令
def cmd_train(args) -> int:
    config = TrainConfig(
        model=args.model,
        model_args=_parse_pairs(args.param, "--param"),
        train_split=args.train_split,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        augment_shift=args.augment_shift,
        label_smoothing=args.label_smoothing,
        optimizer=args.optimizer,
        momentum=args.momentum,
        seed=args.seed,
        threads=args.threads,
        tag=args.tag,
    ).validate()

    print(f"模型：{config.model}（{models.resolve_args(config.model, config.model_args)}）")
    print(f"训练划分：{config.train_split} · 学习率 {config.lr:g} · {config.epochs} 轮 · "
          f"批大小 {config.batch_size} · 线程 {config.threads}")
    if config.tag:
        print(f"备注：{config.tag}")
    print()

    def progress(row: dict, total: int) -> None:
        print(f"[{row['epoch']:>2}/{total}] 训练 {row['train_accuracy']:6.2%} "
              f"验证 {row['val_accuracy']:6.2%} | 验证 loss {row['val_loss']:.4f} "
              f"| {row['seconds']:.1f}s")

    extra = tuple(s.strip() for s in (args.eval or "").split(",") if s.strip())
    run = run_training(config, root=args.root, name=args.name, verify=not args.no_verify,
                        extra_splits=extra, on_epoch=progress)

    config = run.config
    print()
    print(f"✔ 完成：{run.path}")
    print(f"  参数量 {run.record['parameter_count']:,} · 训练 {run.record['seconds']:.1f}s · "
          f"基础验证准确率 {run.metric():.2%}")
    print(f"  保存/重载一致性差异 {run.record['reload_difference']:.2g}")
    print(f"  曲线 {run.curve}")
    print(f"  记录 {run.path / 'summary.md'}")
    print(f"下一句：doodlelab compare --baseline '{run.path.name}'")
    return 0


def cmd_eval(args) -> int:
    run = runs.resolve(args.root, args.run)
    splits = tuple(s.strip() for s in args.split if s.strip())
    results = evaluate_run(run, splits, root=args.root, verify=not args.no_verify)
    for split, ev in results.items():
        print(f"{split}：{ev['count']} 张，准确率 {ev['accuracy']:.2%}，loss {ev['loss']:.4f}")
        hardest = sorted(ev["per_class"], key=lambda row: row["accuracy"] or 0)[:2]
        print("  最难的类别：" + "、".join(
            f"{row['zh']} {row['correct']}/{row['count']}" for row in hardest))
    print(f"已写回 {run.path / 'evaluations.json'}")
    return 0


def cmd_runs(args) -> int:
    items, skipped = runs.list_runs(args.root)
    if args.json:
        for run in items:
            print(run.path.name)
        return 0
    if not items:
        print("还没有实验。先运行：doodlelab train 第一次训练")
    else:
        rows = []
        for run in items:
            val = run.evaluations.get("val_basic")
            rows.append([
                run.path.name, run.config.model, run.structure,
                f"{run.record['parameter_count']:,}",
                f"{val['accuracy']:.2%}" if val else "-",
                f"{run.record['seconds']:.0f}s",
                run.record.get("tag") or "",
            ])
        print(table(["实验目录", "模型", "结构", "参数", "验证准确率", "耗时", "备注"], rows,
                    right={3, 4, 5}))
    if skipped:
        print(f"\n跳过 {len(skipped)} 个旧格式目录（notebook 时代产物）："
              + "、".join(p.name for p in skipped))
    return 0


def cmd_show(args) -> int:
    run = runs.resolve(args.root, args.run)
    print((run.path / "summary.md").read_text(encoding="utf-8"))
    return 0


def cmd_compare(args) -> int:
    items, _ = runs.list_runs(args.root)
    if args.last:
        items = items[-args.last:]
    rows = runs.compare_rows(items, split=args.split)
    rows = [row for row in rows if row["name"] in {r.name for r in items}] or rows
    if not rows:
        print(f"没有实验含有 {args.split} 的评价。先用 doodlelab eval 补做，或换 --split。")
        return 1
    baseline = None
    if args.baseline:
        baseline = runs.resolve(args.root, args.baseline).path.name
    rows = runs.with_deltas(rows, baseline)
    order = {"accuracy": ("accuracy", True), "loss": ("loss", False),
             "seconds": ("seconds", False)}[args.metric]
    rows.sort(key=lambda row: row[order[0]], reverse=order[1])

    print(table(
        ["实验", "模型", "结构", "参数量", "准确率", "Δ准确率", "loss", "耗时", "备注"],
        [[row["name"] + ("*" if row.get("is_baseline") else ""), row["model"], row["structure"],
          f"{row['params']:,}", f"{row['accuracy']:.2%}", f"{row['d_accuracy']:+.2%}",
          f"{row['loss']:.4f}", f"{row['seconds']:.1f}s", row["tag"]] for row in rows],
        right={3, 4, 5, 6, 7},
    ))
    print("\n* = 基线；Δ准确率 为相对基线的绝对差。验证集参与分析，不是独立测试成绩。")
    if not args.no_write:
        written = runs.write_compare(args.root, rows, split=args.split, metric=args.metric)
        print(f"\n已写出：{written['markdown']}\n         {written['csv']}\n         {written['png']}")
    return 0


def cmd_predict(args) -> int:
    run = runs.resolve(args.root, args.run)
    model, _, labels, preprocess = runs.load_model(run.path)
    options = dict(mode=args.mode, polarity=args.polarity,
                   threshold=args.threshold, fit_size=args.fit_size)
    crop = _parse_crop(args.crop)
    if crop:
        options["crop"] = crop
    prediction = predict_image(model, args.image, labels=labels, top_k=args.top_k, **options)

    labels_zh = {row["id"]: row["zh"] for row in labels}
    info = prediction["prepared"]["info"]
    print(f"使用模型：{run.name}")
    print(f"处理：{info['mode']} · 墨色 {info['polarity']} · 边框去除 {info['border_trimmed']} "
          f"· 裁剪 {info['crop']} · 墨迹框 {info['ink_bbox']} · 阈值 {info['threshold']} "
          f"· 最长边 {info['fit_size']} 格")
    for rank, item in enumerate(prediction["top"], 1):
        print(f"  {rank}. {item['zh']}（{item['en']}） {item['score']:.3f}")
    print(f"注意：{prediction['notice']}")

    preview = Path(args.preview) if args.preview else (
        run.path / "predictions" / f"{Path(args.image).stem}_{_stamp()}.png")
    plots.plot_prediction(prediction, preview)
    print(f"对照图：{preview}")
    prediction_log = run.path / "predictions.md"
    with prediction_log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n## {_stamp()} {args.image}\n\n")
        handle.write(f"- 处理：{prediction['prepared']['info']}\n")
        handle.write("- 前三项：" + "；".join(
            f"{labels_zh[item['id']]} {item['score']:.3f}" for item in prediction["top"]) + "\n")
    print(f"记录追加到：{prediction_log}")
    return 0


def _stamp() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def cmd_models(args) -> int:
    if args.name:
        print(models.describe(args.name))
        return 0
    print(table(["模型", "说明"], [[spec.name, spec.summary] for spec in models.REGISTRY.values()]))
    print("\n查看参数：doodlelab models resnet")
    return 0


def cmd_splits(args) -> int:
    store = DataStore(args.root)
    rows = [[name, str(info["count"]), info.get("role") or "",
             info.get("per_class") and str(info["per_class"]) or ""]
            for name in store.splits for info in [store.info(name)]]
    print(f"数据版本 {store.manifest['data_version']} · 根目录 {store.root}")
    print(table(["划分", "张数", "用途", "每类"], rows, right={1, 3}))
    return 0


# --------------------------------------------------------------------- 解析
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="doodlelab", description="28×28 涂鸦识别实验工具链（CPU）")
    parser.add_argument("--root", default=None, help="项目根目录（含 data/manifest.json），默认向上自动查找")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="训练一个新模型并保存实验目录")
    p.add_argument("name", help="实验名，例如 基线 / resnet_宽度32")
    p.add_argument("--model", default="mlp", help=f"模型名：{'、'.join(sorted(models.REGISTRY))}")
    p.add_argument("--param", action="append", metavar="键=值",
                   help="模型参数，可重复：--param widths=16,32,64 --param blocks=1,1,1")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--optimizer", choices=("adam", "sgd"), default="adam")
    p.add_argument("--momentum", type=float, default=0.9, help="仅 sgd 使用")
    p.add_argument("--augment-shift", type=int, choices=(0, 1, 2, 3), default=0)
    p.add_argument("--train-split", default="train_basic", choices=("train_basic", "train_extended"))
    p.add_argument("--seed", type=int, default=20261008)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--tag", default="", help="一行备注，写进实验记录")
    p.add_argument("--eval", default="", help="训练后额外评价的划分，逗号分隔，如 challenge_shift,challenge_thick")
    p.add_argument("--no-verify", action="store_true", help="跳过数据 SHA256 校验（更快，慎用）")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("eval", help="对已有实验补做评价（可含挑战划分）")
    p.add_argument("run", help="实验目录名或路径")
    p.add_argument("--split", action="append", default=["val_basic"],
                   help="可重复；如 challenge_original/challenge_shift/challenge_thick/challenge_occlusion")
    p.add_argument("--no-verify", action="store_true")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("runs", help="列出所有实验")
    p.add_argument("--json", action="store_true", help="只输出目录名，便于脚本使用")
    p.set_defaults(func=cmd_runs)

    p = sub.add_parser("show", help="打印某个实验的记录（summary.md）")
    p.add_argument("run")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("compare", help="实验对照表，可指定基线")
    p.add_argument("--baseline", default=None, help="基线实验（名字/目录/唯一片段）；默认第一个")
    p.add_argument("--split", default="val_basic")
    p.add_argument("--metric", choices=("accuracy", "loss", "seconds"), default="accuracy")
    p.add_argument("--last", type=int, default=0, help="只看最近 N 个实验")
    p.add_argument("--no-write", action="store_true", help="只打印，不写 compare.md/csv/png")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("predict", help="用某个实验的模型识别一张图片")
    p.add_argument("run")
    p.add_argument("image", help="图片路径（PNG/JPG），可用 tools/board.html 手绘导出")
    p.add_argument("--mode", choices=("drawing", "photo"), default="drawing")
    p.add_argument("--polarity", choices=("auto", "dark", "light"), default="auto")
    p.add_argument("--crop", default=None, help="左,上,右,下（像素），拍照图常用")
    p.add_argument("--threshold", type=int, default=40)
    p.add_argument("--fit-size", type=int, default=26)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--preview", default=None, help="对照图输出路径")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("models", help="列出模型与参数")
    p.add_argument("name", nargs="?", default=None)
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("splits", help="列出数据划分")
    p.set_defaults(func=cmd_splits)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError, KeyError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
