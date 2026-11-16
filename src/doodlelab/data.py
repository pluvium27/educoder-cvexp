"""数据读取与校验。所有数据来自本地 data/，不联网。

约定：x 为 float32 [N,1,28,28]、[0,1] 黑底亮线；y 为 int64 [N]；标签顺序由 manifest.json 固定。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

MANIFEST_NAME = "data/manifest.json"


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(2 ** 20), b""):
            digest.update(block)
    return digest.hexdigest()


def find_root(start: str | Path | None = None) -> Path:
    """从 start（默认当前目录）向上找含 data/manifest.json 的目录。"""
    current = Path(start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / MANIFEST_NAME).is_file():
            return candidate
    raise FileNotFoundError(f"未找到 {MANIFEST_NAME}：请在项目根目录运行，或用 --root 指定")


@dataclass(frozen=True)
class Split:
    """一个数据划分。meta 内含数据版本、SHA256 与预处理说明，便于写进实验记录。"""
    name: str
    x: torch.Tensor
    y: torch.Tensor
    labels: list
    meta: dict
    source_index: np.ndarray
    sample_hash: np.ndarray

    def __len__(self) -> int:
        return int(self.y.numel())


class DataStore:
    """按需加载并缓存划分；每次首次加载都会做 SHA256 校验（可关）。"""

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = find_root(root)
        manifest_path = self.root / MANIFEST_NAME
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.labels = self.manifest["labels"]
        self.data_dir = manifest_path.parent
        self.manifest_sha256 = sha256(manifest_path)
        self._cache: dict[str, Split] = {}

    # ------------------------------------------------------------------ 查询
    @property
    def splits(self) -> tuple[str, ...]:
        return tuple(self.manifest["splits"])

    def info(self, name: str) -> dict:
        try:
            return dict(self.manifest["splits"][name], name=name)
        except KeyError as error:
            raise ValueError(f"未知划分 {name!r}；可用：{'、'.join(self.splits)}") from error

    # ------------------------------------------------------------------ 加载
    def load(self, name: str = "train_basic", verify: bool = True) -> Split:
        if name in self._cache:
            return self._cache[name]
        info = self.info(name)
        path = self.data_dir / info["file"]
        if not path.exists():
            raise FileNotFoundError(f"{path} 不存在；请重新完整解压数据包")
        if verify and sha256(path) != info["sha256"]:
            raise ValueError(f"{path.name} 校验失败，请重新解压课程发布包")
        with np.load(path, allow_pickle=False) as handle:
            images = handle["images"]
            targets = handle["labels"]
            source_index = handle["source_index"].copy()
            sample_hash = handle["sample_hash"].copy()
        n_classes = len(self.labels)
        if images.dtype != np.uint8 or images.shape != (len(targets), 28, 28):
            raise ValueError(f"{path.name} 图像类型或形状错误：{images.dtype} {images.shape}")
        if len(targets) == 0 or targets.min() < 0 or targets.max() >= n_classes:
            raise ValueError(f"{path.name} 标签越界")
        if targets.dtype != np.int64:
            raise ValueError(f"{path.name} 标签必须为 int64")
        split = Split(
            name=name,
            x=torch.from_numpy(images.copy()).unsqueeze(1).float() / 255.0,
            y=torch.from_numpy(targets.copy()),
            labels=self.labels,
            meta=dict(
                split=name, file=info["file"], count=int(len(targets)), role=info.get("role"),
                data_version=self.manifest["data_version"],
                sha256=info["sha256"], manifest_sha256=self.manifest_sha256,
                preprocessing=self.manifest.get("preprocessing"),
            ),
            source_index=source_index,
            sample_hash=sample_hash,
        )
        self._cache[name] = split
        return split

    def counts(self) -> dict[str, int]:
        return {name: int(self.manifest["splits"][name]["count"]) for name in self.splits}
