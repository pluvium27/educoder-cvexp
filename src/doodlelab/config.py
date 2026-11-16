"""实验配置：纯数据 + 校验。不依赖 notebook、UI 或全局状态。"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field

from . import models


@dataclass(frozen=True)
class TrainConfig:
    model: str = "mlp"
    model_args: dict = field(default_factory=dict)
    train_split: str = "train_basic"
    lr: float = 1e-3
    epochs: int = 12
    batch_size: int = 64
    weight_decay: float = 0.0
    augment_shift: int = 0
    label_smoothing: float = 0.0
    optimizer: str = "adam"
    momentum: float = 0.9
    seed: int = 20261008
    threads: int = 2
    tag: str = ""

    # ------------------------------------------------------------------ 校验
    def validate(self) -> "TrainConfig":
        """任何一项不合法都抛 ValueError，信息直接可读。"""
        if self.model not in models.REGISTRY:
            raise ValueError(f"未知模型 {self.model!r}；可选：{'、'.join(sorted(models.REGISTRY))}")
        models.resolve_args(self.model, self.model_args)          # 参数名/类型/范围
        for key in ("epochs", "batch_size", "threads"):
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{key} 必须为正整数，收到 {value!r}")
        if not math.isfinite(self.lr) or self.lr <= 0:
            raise ValueError(f"lr 必须为有限正数，收到 {self.lr!r}")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay 不能为负")
        if not math.isfinite(self.label_smoothing) or not 0 <= self.label_smoothing < 1:
            raise ValueError("label_smoothing 需在 [0,1) 内")
        if not math.isfinite(self.momentum) or not 0 <= self.momentum < 1:
            raise ValueError("momentum 需在 [0,1) 内")
        if self.optimizer not in ("adam", "sgd"):
            raise ValueError("optimizer 选择 adam 或 sgd")
        if self.augment_shift not in (0, 1, 2, 3):
            raise ValueError("augment_shift 为 0—3 的整数像素")
        if not isinstance(self.seed, int) or not 0 <= self.seed < 2 ** 32:
            raise ValueError("seed 需 0—2³²−1 整数")
        if self.tag and not isinstance(self.tag, str):
            raise ValueError("tag 需字符串")
        return self

    # ------------------------------------------------------------ 序列化
    def resolved_args(self) -> dict:
        return models.resolve_args(self.model, self.model_args)

    def to_dict(self) -> dict:
        data = copy.deepcopy(self.__dict__)
        data["model_args"] = self.resolved_args()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "TrainConfig":
        known = set(cls.__dataclass_fields__)
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"配置里有未知字段 {sorted(unknown)}")
        payload = copy.deepcopy(data)
        payload["model_args"] = payload.get("model_args") or {}
        return cls(**payload).validate()

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def fingerprint(self) -> str:
        """结构 + 训练条件的稳定摘要，用于判断两次实验差异来自哪里。"""
        payload = self.to_dict()
        payload.pop("tag", None)
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)
