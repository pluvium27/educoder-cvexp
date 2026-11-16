"""评价指标：准确率、混淆矩阵、每类表现、错误样例预览。

说明：分数是 softmax 输出，不是保证正确的概率；类别外输入同样会被归到八类之一。
"""
from __future__ import annotations

from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch


def evaluate(model: nn.Module, split, batch_size: int = 256) -> dict:
    """模型 + Split → 完整评价字典（内存中使用，落盘前先 compact）。"""
    was_training = model.training
    model.eval()
    scores: list[np.ndarray] = []
    total_loss = 0.0
    try:
        with torch.no_grad():
            loader = DataLoader(TensorDataset(split.x, split.y), batch_size=batch_size,
                                shuffle=False, num_workers=0)
            for x, y in loader:
                logits = model(x)
                if tuple(logits.shape) != (len(y), len(split.labels)):
                    raise ValueError(f"模型输出应为 [B,{len(split.labels)}]，收到 {tuple(logits.shape)}")
                if not torch.isfinite(logits).all():
                    raise ValueError("模型出现非有限输出")
                total_loss += nn.functional.cross_entropy(logits, y, reduction="sum").item()
                scores.append(torch.softmax(logits, dim=1).numpy())
    finally:
        model.train(was_training)

    probs = np.concatenate(scores)
    predicted = probs.argmax(1)
    truth = split.y.numpy()
    n_classes = len(split.labels)
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(confusion, (truth, predicted), 1)
    per_class = []
    for index, label in enumerate(split.labels):
        count = int(confusion[index].sum())
        correct = int(confusion[index, index])
        per_class.append(dict(**label, count=count, correct=correct,
                              accuracy=correct / count if count else None))
    return dict(
        accuracy=float((predicted == truth).mean()),
        loss=total_loss / len(truth),
        count=int(len(truth)),
        per_class=per_class,
        confusion=confusion.tolist(),
        predictions=predicted.tolist(),
        true_labels=truth.tolist(),
        scores=probs.tolist(),
        error_indices=np.flatnonzero(predicted != truth).tolist(),
        dataset=split.meta,
    )


def quick(model: nn.Module, split, batch_size: int = 256) -> dict:
    """逐轮监测用：只算 loss 与准确率，不保留分数/混淆矩阵（比 evaluate 省一半开销）。"""
    was_training = model.training
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    try:
        with torch.no_grad():
            loader = DataLoader(TensorDataset(split.x, split.y), batch_size=batch_size,
                                shuffle=False, num_workers=0)
            for x, y in loader:
                logits = model(x)
                total_loss += nn.functional.cross_entropy(logits, y, reduction="sum").item()
                correct += int((logits.argmax(1) == y).sum())
                seen += len(y)
    finally:
        model.train(was_training)
    return dict(loss=total_loss / seen, accuracy=correct / seen, count=seen)


def compact(ev: dict, errors: int = 8) -> dict:
    """完整评价 → 落盘用小记录：不保存逐图分数，只留前 errors 条错误。"""
    keep = ("accuracy", "loss", "count", "per_class", "confusion", "dataset")
    result = {key: ev[key] for key in keep if key in ev}
    result["errors_preview"] = [
        dict(row=index, true=ev["true_labels"][index], predicted=ev["predictions"][index],
             score=float(max(ev["scores"][index])))
        for index in ev.get("error_indices", [])[:errors]
    ]
    return result
