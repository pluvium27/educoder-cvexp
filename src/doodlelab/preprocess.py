"""外部图片预处理：把手绘/拍照图变成模型能吃的 28×28 黑底亮线。

处理链：透明合成白底 → (可选裁剪) → 灰度 → (拍照模式自动对比度) → 判断墨色方向
→ 按阈值找墨迹外接框 → 等比缩放到最长边 fit_size → 居中贴到 28×28。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps
import torch

MIN_VISIBLE_INK = 180.0   # 缩小保底灰阶：保证 1 像素可见，又不把笔画糊成纯白
PREPROCESS = {
    "version": "external-v1",
    "polarity": "bright_ink_dark_background",
    "shape": [1, 28, 28],
    "range": [0, 1],
    "fit_size": 26,
    "threshold": 40,
    "centering": "bounding_box",
    "photo_autocontrast": True,
}


def preprocess_image(image, mode: str = "drawing", polarity: str = "auto",
                     crop: tuple[int, int, int, int] | list | None = None,
                     threshold: int = 40, fit_size: int = 26) -> dict:
    """路径 / PIL.Image / uint8 数组 → dict(original, processed, tensor, info)。"""
    if mode not in ("drawing", "photo"):
        raise ValueError("mode 选择 drawing 或 photo")
    if polarity not in ("auto", "dark", "light"):
        raise ValueError("polarity 选择 auto/dark/light")
    if not 0 <= threshold < 255 or not 8 <= fit_size <= 28:
        raise ValueError("threshold 需在 [0,255)、fit_size 需在 [8,28]")

    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            im = ImageOps.exif_transpose(opened).convert("RGBA")
    elif isinstance(image, Image.Image):
        im = ImageOps.exif_transpose(image).convert("RGBA")
    elif isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            raise TypeError("数组需 uint8[0,255]；浮点输入请先明确转换")
        im = Image.fromarray(image).convert("RGBA")
    else:
        raise TypeError("image 需路径、PIL 图像或 uint8 数组")

    # 透明区域合成到白底，否则转灰度会把透明背景当成黑色。
    im = Image.alpha_composite(Image.new("RGBA", im.size, "white"), im).convert("RGB")
    original = im.copy()
    if crop:
        left, top, right, bottom = (int(v) for v in crop)
        if not (0 <= left < right <= im.width and 0 <= top < bottom <= im.height):
            raise ValueError("crop 超出图像范围或为空")
        im = im.crop((left, top, right, bottom))

    gray = ImageOps.grayscale(im)
    if mode == "photo":
        gray = ImageOps.autocontrast(gray, cutoff=1)
    # 3×3 中值滤波：去掉纸纹、压缩噪点和扫描杂点；线条本身有数像素宽，几乎不受影响。
    gray = gray.filter(ImageFilter.MedianFilter(3))
    array = np.asarray(gray)
    array, trimmed = trim_uniform_border(array)
    chosen = polarity if polarity != "auto" else auto_polarity(array, threshold)
    ink = 255 - array if chosen == "dark" else array.copy()
    background = float(np.median(ink))
    if background > 6:
        # “纸不是黑的”：把底色调回 0、笔画拉到 255，使输入亮度与训练数据一致。
        ink = np.clip((ink - background) * 255.0 / max(1.0, 255.0 - background), 0, 255)
    cutoff = threshold
    if mode == "photo":
        # 拍照图还会残留暗角与阴影，按“笔画只占少数像素”的常识再抬高一次阈值。
        cutoff = max(threshold, ink_cutoff(ink, threshold))
        ink = np.where(ink >= cutoff, ink, 0)
    mask = ink > threshold
    if not mask.any():
        raise ValueError("未找到线条：图片为空、阈值过高，或黑白方向判断错误")

    # 用“整行/整列墨迹够多”来定外接框：单个噪点、边角残留不应把框撑到整张图。
    min_count = max(2, int(0.01 * max(ink.shape)))
    ys = np.flatnonzero(mask.sum(axis=1) >= min_count)
    xs = np.flatnonzero(mask.sum(axis=0) >= min_count)
    if not len(xs) or not len(ys):
        raise ValueError("未找到成片线条：图案太淡或被拆成碎片，可放宽 --threshold")
    bbox = (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1))
    patch = ink[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    scale = fit_size / max(patch.shape)
    size = (max(1, round(patch.shape[1] * scale)), max(1, round(patch.shape[0] * scale)))
    factor = max(patch.shape) / max(size)
    resized = np.asarray(Image.fromarray(patch).resize(size, Image.Resampling.LANCZOS)).astype(np.float32)
    if factor >= 3:
        # 512 画布缩到 26 格时线条只占零点几像素，会被插值抹成近乎全黑。
        # 只补“本来会消失”的块：块内有墨迹但灰度已被压低，就给一个清楚可见的灰阶；
        # 本来就清晰的笔画不动，避免凭空变粗。
        blocks = any_ink_blocks(patch > threshold, size)
        resized = np.where(blocks, np.maximum(resized, MIN_VISIBLE_INK), resized)
    canvas = np.zeros((28, 28), dtype=np.float32)
    top, left = (28 - size[1]) // 2, (28 - size[0]) // 2
    canvas[top:top + size[1], left:left + size[0]] = resized
    canvas = np.clip(canvas, 0.0, 255.0)   # LANCZOS 会轻微过冲，越界值必须夹回
    processed = Image.fromarray(canvas.astype(np.uint8))
    tensor = torch.from_numpy(canvas[None, None] / 255.0)
    return dict(
        original=original,
        processed=processed,
        tensor=tensor,
        info=dict(version=PREPROCESS["version"], mode=mode, polarity=chosen,
                  crop=list(crop) if crop else None, border_trimmed=list(trimmed),
                  ink_bbox=list(bbox), threshold=threshold, fit_size=fit_size,
                  centering="bounding_box", downscale=round(factor, 1),
                  photo_cutoff=cutoff if mode == "photo" else None),
    )


def any_ink_blocks(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """把 mask 按 size=(宽,高) 分块，返回每块内是否含墨迹（用积分图，近似最大池化）。"""
    height, width = mask.shape
    target_w, target_h = size
    ys = np.linspace(0, height, target_h + 1).astype(int)
    xs = np.linspace(0, width, target_w + 1).astype(int)
    integral = np.zeros((height + 1, width + 1), dtype=np.int32)
    integral[1:, 1:] = mask.astype(np.int32).cumsum(0).cumsum(1)
    y1 = np.maximum(ys[1:], ys[:-1] + 1)
    x1 = np.maximum(xs[1:], xs[:-1] + 1)
    y1 = np.minimum(y1, height)
    x1 = np.minimum(x1, width)
    total = (integral[np.ix_(y1, x1)] - integral[np.ix_(ys[:-1], x1)]
             - integral[np.ix_(y1, xs[:-1])] + integral[np.ix_(ys[:-1], xs[:-1])])
    return total > 0


def trim_uniform_border(array: np.ndarray, tolerance: int = 12,
                        max_fraction: float = 0.4) -> tuple[np.ndarray, tuple[int, int]]:
    """去掉四周与边缘同色的整齐外框（拍照时的桌面、纸边、截图留白）。

    只裁“整行/整列都接近边缘主色”的部分，最多裁掉每边 max_fraction，
    因此对正常白底手绘几乎无影响，但能救掉“纸面图外面一圈灰背景”。
    """
    top = bottom = left = right = 0
    limit_y, limit_x = int(array.shape[0] * max_fraction), int(array.shape[1] * max_fraction)

    def uniform(line: np.ndarray, color: float) -> bool:
        # 用最大偏差而非中位数：一行里只要有几像素线条，中位数仍等于背景色，
        # 会把“穿过图案”的行误判成空白而裁掉图案。
        return bool(np.abs(line.astype(np.int16) - color).max() <= tolerance)

    color = float(np.median(np.concatenate([array[0], array[-1], array[:, 0], array[:, -1]])))
    while top < limit_y and uniform(array[top, left:array.shape[1] - right], color):
        top += 1
    while bottom < limit_y and uniform(array[array.shape[0] - 1 - bottom, left:array.shape[1] - right], color):
        bottom += 1
    while left < limit_x and uniform(array[top:array.shape[0] - bottom, left], color):
        left += 1
    while right < limit_x and uniform(array[top:array.shape[0] - bottom, array.shape[1] - 1 - right], color):
        right += 1
    if top + bottom + left + right == 0:
        return array, (0, 0)
    trimmed = array[top:array.shape[0] - bottom or None, left:array.shape[1] - right or None]
    if trimmed.size == 0:
        return array, (0, 0)
    return trimmed, (top, left)


def ink_cutoff(ink: np.ndarray, threshold: int = 40, max_fraction: float = 0.15) -> int:
    """拍照模式：把阈值抬到“墨迹不超过画面 max_fraction”的位置，压掉纸面灰雾与暗角。

    依据仍是“手绘线条只占少数像素”。阈值只会被抬高，不会被降低。
    """
    total = ink.size
    for level in range(int(threshold), 256, 4):
        if float((ink >= level).sum()) <= max_fraction * total:
            return level
    return 254


def auto_polarity(array: np.ndarray, threshold: int = 40) -> str:
    """判断墨色方向：手绘线条只占少数像素，因此取“墨迹占比更小”的那个方向。

    仅当两个方向都像“大半是墨”（对比度异常）时，才退回“边框亮则墨深”的老规则。
    """
    dark = float((255 - array > threshold).mean())
    light = float((array > threshold).mean())
    if min(dark, light) > 0.5:
        border = float(np.median(np.concatenate([array[0], array[-1], array[:, 0], array[:, -1]])))
        return "dark" if border > 127 else "light"
    return "dark" if dark < light else "light"


def predict_image(model, image, labels=None, top_k: int = 3, **options) -> dict:
    """模型 + 图片 → 预测 dict（含预处理结果，便于核对模型实际看到了什么）。"""
    labels = labels or getattr(model, "doodle_labels", None)
    if labels is None:
        raise ValueError("需要 labels：先 load(\"runs/...\")，或显式传入 labels")
    if not 1 <= top_k <= len(labels):
        raise ValueError(f"top_k 需在 [1,{len(labels)}]")
    prepared = image if isinstance(image, dict) and "tensor" in image else preprocess_image(image, **options)
    x = prepared["tensor"]
    if tuple(x.shape) != (1, 1, 28, 28) or not torch.isfinite(x).all() or x.min() < 0 or x.max() > 1:
        raise ValueError("预测输入应为 [1,1,28,28] 且取值在 [0,1]")
    was_training = model.training
    model.eval()
    with torch.no_grad():
        scores = torch.softmax(model(x), 1)[0].numpy()
    model.train(was_training)
    order = np.argsort(-scores)[:top_k]
    return dict(
        top=[dict(**labels[int(i)], score=float(scores[i])) for i in order],
        scores=scores.tolist(),
        prepared=prepared,
        notice="分数不构成正确性保证；类别外输入仍会被归入八类之一。",
    )
