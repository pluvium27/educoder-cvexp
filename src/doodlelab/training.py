"""训练循环：一次性、从头训练、保存最后一轮。

刻意不做的几件事：不在验证集上挑最优轮次、不断点续训、不把验证集混进训练。
这样每次实验的结果都由 config 唯一决定，对照实验才有意义。
"""
from __future__ import annotations

import copy
import json
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from . import metrics, models, runs
from .config import TrainConfig
from .data import DataStore
from .preprocess import PREPROCESS


def set_seed(seed: int, threads: int = 2) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def shift_batch(x: torch.Tensor, amount: int) -> torch.Tensor:
    """训练时随机平移（数据增强）；逐张独立取值，幅度 ±amount 像素。"""
    if not amount:
        return x
    out = torch.zeros_like(x)
    shifts = torch.randint(-amount, amount + 1, (len(x), 2))
    for i, (dy, dx) in enumerate(shifts.tolist()):
        sy0, sy1 = max(0, -dy), min(28, 28 - dy)
        sx0, sx1 = max(0, -dx), min(28, 28 - dx)
        out[i, :, sy0 + dy:sy1 + dy, sx0 + dx:sx1 + dx] = x[i, :, sy0:sy1, sx0:sx1]
    return out


def _optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    if config.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    return torch.optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum,
                           weight_decay=config.weight_decay)


def _overlap(train, val) -> int:
    return len(set(train.sample_hash.tolist()) & set(val.sample_hash.tolist()))


def train(config: TrainConfig, root=".", name: str = "实验", verify: bool = True,
          extra_splits: tuple[str, ...] = (), on_epoch=None) -> runs.Run:
    """按配置训练一个模型，保存实验目录并返回 Run。"""
    config = config.validate()
    store = DataStore(root)
    train_split = store.load(config.train_split, verify=verify)
    val_split = store.load("val_basic", verify=verify)
    if train_split.labels != val_split.labels:
        raise ValueError("训练与验证的标签映射不一致")
    if _overlap(train_split, val_split):
        raise ValueError("训练集与验证集存在重复图像，数据文件可能被改动")

    set_seed(config.seed, config.threads)
    model = models.build(config.model, config.resolved_args(), len(val_split.labels))
    model.doodle_labels = val_split.labels
    model.doodle_preprocess = copy.deepcopy(PREPROCESS)

    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(TensorDataset(train_split.x, train_split.y),
                        batch_size=config.batch_size, shuffle=True,
                        generator=generator, num_workers=0)
    optimizer = _optimizer(model, config)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    history: list[dict] = []
    started = time.perf_counter()
    cpu_started = time.process_time()
    for epoch in range(1, config.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        grad_seen = False
        for x, y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(shift_batch(x, config.augment_shift)), y)
            if not torch.isfinite(loss):
                raise ValueError("loss 非有限：检查学习率或输入")
            loss.backward()
            for param in model.parameters():
                if param.grad is not None:
                    if not torch.isfinite(param.grad).all():
                        raise ValueError("梯度非有限：调小学习率")
                    grad_seen = grad_seen or bool(torch.count_nonzero(param.grad))
            optimizer.step()
        if not grad_seen:
            raise ValueError("所有梯度为零：检查网络结构与训练设置")

        tr = metrics.quick(model, train_split, config.batch_size)
        va = metrics.quick(model, val_split, config.batch_size)
        row = dict(epoch=epoch, train_loss=tr["loss"], train_accuracy=tr["accuracy"],
                   val_loss=va["loss"], val_accuracy=va["accuracy"],
                   seconds=time.perf_counter() - epoch_started)
        history.append(row)
        if on_epoch:
            on_epoch(row, config.epochs)
    model.eval()
    seconds = time.perf_counter() - started

    evaluations = {"val_basic": metrics.evaluate(model, val_split)}
    for split in extra_splits:
        if split in ("val_basic", config.train_split):
            continue
        evaluations[split] = metrics.evaluate(model, store.load(split, verify=verify))

    run = runs.save(root, name, model, config, history, evaluations, seconds,
                    parameter_count=sum(p.numel() for p in model.parameters()))

    # 保存 → 重载 → 再评价：确认权重与配置真的可复现，差异不为零说明保存链路有问题。
    reloaded, _, _, _ = runs.load_model(run.path)
    repeat = metrics.evaluate(reloaded, val_split)
    difference = float(np.max(np.abs(
        np.asarray(repeat["scores"]) - np.asarray(evaluations["val_basic"]["scores"]))))
    run.record["reload_difference"] = difference
    run.record["cpu_seconds"] = time.process_time() - cpu_started
    if difference > 1e-6:
        raise RuntimeError(f"保存/重载后预测分数差 {difference:g}，请检查权重保存链路")
    (run.path / "summary.md").write_text(runs.summary_markdown(run), encoding="utf-8")
    return run


def evaluate_run(run: runs.Run, splits: tuple[str, ...], root=".", verify: bool = True,
                 batch_size: int = 256) -> dict[str, dict]:
    """对已有实验补做评价（如挑战划分），结果写回 evaluations.json。"""
    store = DataStore(root)
    model, config, _, _ = runs.load_model(run.path)
    results = {}
    for split in splits:
        ev = metrics.evaluate(model, store.load(split, verify=verify), batch_size=batch_size)
        results[split] = ev
        run.evaluations[split] = metrics.compact(ev)
        if "confusion" in ev:
            from . import plots
            plots.plot_confusion(ev, store.labels, run.path / f"confusion_{split}.png")
    (run.path / "evaluations.json").write_text(
        json.dumps(run.evaluations, ensure_ascii=False, indent=2), encoding="utf-8")
    (run.path / "summary.md").write_text(runs.summary_markdown(run), encoding="utf-8")
    return results
