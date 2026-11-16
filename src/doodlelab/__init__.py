"""doodlelab：28×28 涂鸦识别的命令行实验工具链（CPU）。

典型用法:
    lab = DataStore()                      # 读 data/manifest.json
    run = train(TrainConfig(model="resnet"), name="基线")
    evaluate(run.model, store.load("challenge_shift"))
命令行等价形式见 `doodlelab --help`。
"""
from .config import TrainConfig
from .data import DataStore, Split, find_root
from .metrics import compact, evaluate, quick
from .models import REGISTRY, build, describe, register, ModelSpec, Param
from .preprocess import PREPROCESS, predict_image, preprocess_image
from .runs import (Run, compare_rows, list_runs, load_model, read, resolve,
                   with_deltas, write_compare)
from .training import evaluate_run, set_seed, train

__version__ = "0.3.0"

__all__ = [
    "PREPROCESS", "REGISTRY", "DataStore", "ModelSpec", "Param", "Run", "Split", "TrainConfig",
    "build", "compact", "compare_rows", "describe", "evaluate", "evaluate_run", "find_root",
    "list_runs", "load_model", "predict_image", "preprocess_image", "quick", "read", "register",
    "resolve", "set_seed", "train", "with_deltas", "write_compare", "__version__",
]
