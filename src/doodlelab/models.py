"""模型注册表：mlp / cnn / resnet 平级，新增结构只需注册一个 ModelSpec。

约定（所有模型共享）：输入 float32 [B,1,28,28]、范围 [0,1]；输出 [B,8] 未 softmax 的 logits。
做 ResNet 改进时，推荐直接注册新模型（如 resnet_wide、resnet_se），
不要改已有模型的默认参数，否则旧实验无法按 config 复现。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Param:
    """一个可调参数的声明：类型、默认值、取值范围。CLI 与校验都读它。"""
    name: str
    kind: str                     # int / float / int_list
    default: Any
    minimum: float | None = None
    maximum: float | None = None
    help: str = ""


@dataclass(frozen=True)
class ModelSpec:
    name: str
    summary: str
    params: tuple[Param, ...]
    build: Callable[[dict[str, Any], int], nn.Module]


REGISTRY: dict[str, ModelSpec] = {}


def register(spec: ModelSpec) -> ModelSpec:
    if spec.name in REGISTRY:
        raise ValueError(f"模型 {spec.name} 已注册")
    REGISTRY[spec.name] = spec
    return spec


def _coerce(param: Param, value: Any) -> Any:
    """把 CLI 字符串/列表统一成声明类型，并检查范围。"""
    try:
        if param.kind == "int_list":
            items = value if isinstance(value, (list, tuple)) else str(value).split(",")
            out = [int(str(v).strip()) for v in items if str(v).strip() != ""]
            if not out:
                raise ValueError
        elif param.kind == "int":
            out = int(value)
        elif param.kind == "float":
            out = float(value)
        else:  # pragma: no cover - 注册表自身出错
            raise TypeError(f"未知参数类型 {param.kind}")
    except (TypeError, ValueError) as error:
        raise ValueError(f"参数 {param.name} 需要 {param.kind}，收到 {value!r}") from error
    values = out if isinstance(out, list) else [out]
    for item in values:
        if param.minimum is not None and item < param.minimum:
            raise ValueError(f"参数 {param.name} 不能小于 {param.minimum}")
        if param.maximum is not None and item > param.maximum:
            raise ValueError(f"参数 {param.name} 不能大于 {param.maximum}")
    return out


def defaults(model: str) -> dict[str, Any]:
    """模型未指定参数时使用的默认值。"""
    spec = REGISTRY.get(model)
    if spec is None:
        raise ValueError(f"未知模型 {model!r}；可选：{'、'.join(sorted(REGISTRY))}")
    return {p.name: list(p.default) if isinstance(p.default, list) else p.default for p in spec.params}


def resolve_args(model: str, args: dict[str, Any] | None) -> dict[str, Any]:
    """补默认值 + 拒绝拼错的参数名。返回可直接 JSON 保存的新字典。"""
    spec = REGISTRY.get(model)
    if spec is None:
        raise ValueError(f"未知模型 {model!r}；可选：{'、'.join(sorted(REGISTRY))}")
    given = dict(args or {})
    known = {p.name for p in spec.params}
    unknown = set(given) - known
    if unknown:
        raise ValueError(f"模型 {model} 不接受参数 {sorted(unknown)}；可用：{sorted(known)}")
    resolved = defaults(model)
    for param in spec.params:
        if param.name in given:
            resolved[param.name] = _coerce(param, given[param.name])
    return resolved


def build(model: str, args: dict[str, Any] | None = None, num_classes: int = 8) -> nn.Module:
    """按名字与参数构建模型，并做一次前向形状/有限性检查。"""
    spec = REGISTRY.get(model)
    if spec is None:
        raise ValueError(f"未知模型 {model!r}；可选：{'、'.join(sorted(REGISTRY))}")
    resolved = resolve_args(model, args)
    module = spec.build(resolved, num_classes)
    if not isinstance(module, nn.Module):
        raise TypeError(f"模型 {model} 的构造函数必须返回 torch.nn.Module")
    module.eval()
    with torch.no_grad():
        out = module(torch.zeros(2, 1, 28, 28))
    if tuple(out.shape) != (2, num_classes) or not torch.isfinite(out).all():
        raise ValueError(f"模型 {model} 应接受 [B,1,28,28] 并返回有限的 [B,{num_classes}] logits")
    return module


# --------------------------------------------------------------------------- mlp

def _build_mlp(args: dict[str, Any], num_classes: int) -> nn.Module:
    layers: list[nn.Module] = [nn.Flatten()]
    width = 28 * 28
    for hidden in args["hidden_sizes"]:
        layers += [nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(args["dropout"])]
        width = hidden
    layers.append(nn.Linear(width, num_classes))
    return nn.Sequential(*layers)


register(ModelSpec(
    name="mlp",
    summary="两层隐藏层全连接；作业默认基线，最快",
    params=(
        Param("hidden_sizes", "int_list", [64, 32], 1, 2048, "各隐藏层宽度"),
        Param("dropout", "float", 0.0, 0.0, 0.9, "全连接层后的 dropout"),
    ),
    build=_build_mlp,
))


# --------------------------------------------------------------------------- cnn

def _build_cnn(args: dict[str, Any], num_classes: int) -> nn.Module:
    a, b = args["channels"]
    return nn.Sequential(
        nn.Conv2d(1, a, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(a, b, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(),
        nn.Linear(b * 7 * 7, 32), nn.ReLU(), nn.Dropout(args["dropout"]),
        nn.Linear(32, num_classes),
    )


register(ModelSpec(
    name="cnn",
    summary="两层 3×3 卷积 + 池化；参数少、比 MLP 更懂局部笔画",
    params=(
        Param("channels", "int_list", [8, 16], 1, 256, "两个卷积层的通道数"),
        Param("dropout", "float", 0.0, 0.0, 0.9, "全连接层前的 dropout"),
    ),
    build=_build_cnn,
))


# ------------------------------------------------------------------------ resnet

class BasicBlock(nn.Module):
    """3×3-3×3 残差块；下采样/升通道时用 1×1 卷积对齐形状。"""

    def __init__(self, cin: int, cout: int, stride: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.drop = nn.Dropout(dropout)
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(self.drop(out)))
        skip = x if self.short is None else self.short(x)
        return F.relu(out + skip)


class TinyResNet(nn.Module):
    """面 28×28 裁剪过的 ResNet：3×3 stem 不降采样，首 stage stride=1。"""

    def __init__(self, widths: list[int], blocks: list[int], stem: int,
                 num_classes: int, dropout: float) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, stem, 3, 1, 1, bias=False), nn.BatchNorm2d(stem), nn.ReLU(inplace=True))
        stages: list[nn.Module] = []
        cin = stem
        for index, (width, count) in enumerate(zip(widths, blocks)):
            for j in range(count):
                stride = (2 if index > 0 else 1) if j == 0 else 1
                stages.append(BasicBlock(cin, width, stride, dropout))
                cin = width
        self.stages = nn.Sequential(*stages)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(cin, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.stages(self.stem(x)))


def _build_resnet(args: dict[str, Any], num_classes: int) -> nn.Module:
    widths, blocks = args["widths"], args["blocks"]
    if len(widths) != len(blocks):
        raise ValueError(f"widths 与 blocks 长度需一致：{widths} vs {blocks}")
    return TinyResNet(widths, blocks, args["stem"], num_classes, args["dropout"])


register(ModelSpec(
    name="resnet",
    summary="BasicBlock 残差网络；改宽度/深度/宽度组是主要改进方向",
    params=(
        Param("widths", "int_list", [16, 32, 64], 1, 512, "各 stage 的输出通道"),
        Param("blocks", "int_list", [1, 1, 1], 1, 6, "各 stage 的残差块数"),
        Param("stem", "int", 16, 1, 512, "stem 卷积通道数"),
        Param("dropout", "float", 0.0, 0.0, 0.9, "残差块内 dropout"),
    ),
    build=_build_resnet,
))


def describe(model: str) -> str:
    spec = REGISTRY[model]
    lines = [f"{spec.name}：{spec.summary}"]
    for param in spec.params:
        lines.append(f"  --param {param.name}={param.default}  {param.help}")
    return "\n".join(lines)
