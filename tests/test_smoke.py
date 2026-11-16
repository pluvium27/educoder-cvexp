"""冒烟测试：数据能读、模型能建、一次真训练能跑通并且可复现。

运行：uv run pytest
"""
from __future__ import annotations

import pytest
import torch

from doodlelab import DataStore, TrainConfig, build, evaluate, list_runs, predict_image, train
from doodlelab.data import find_root
from doodlelab.models import REGISTRY, resolve_args

ROOT = find_root()


@pytest.mark.parametrize("model", sorted(REGISTRY))
def test_build_all_models(model):
    module = build(model, None)
    out = module(torch.zeros(3, 1, 28, 28))
    assert tuple(out.shape) == (3, 8)
    assert torch.isfinite(out).all()
    assert sum(p.numel() for p in module.parameters()) > 0


def test_model_args_are_validated():
    with pytest.raises(ValueError, match="不接受参数"):
        resolve_args("mlp", {"hidden": [64]})
    with pytest.raises(ValueError, match="长度需一致"):
        build("resnet", {"widths": [16, 32], "blocks": [1, 1, 1]})
    assert resolve_args("resnet", {"widths": "8,16"})["widths"] == [8, 16]


def test_config_rejects_bad_values():
    with pytest.raises(ValueError):
        TrainConfig(lr=0).validate()
    with pytest.raises(ValueError):
        TrainConfig(optimizer="lamb").validate()
    with pytest.raises(ValueError):
        TrainConfig(tag="x").from_dict({"tag": "x", "typo": 1})


def test_data_splits_load_and_verify():
    store = DataStore(ROOT)
    train_split = store.load("train_basic")
    val_split = store.load("val_basic")
    assert len(train_split) == 2400 and len(val_split) == 1600
    assert train_split.x.dtype == torch.float32 and tuple(train_split.x.shape[1:]) == (1, 28, 28)
    assert float(train_split.x.max()) <= 1.0 and float(train_split.x.min()) >= 0.0
    assert not set(train_split.sample_hash.tolist()) & set(val_split.sample_hash.tolist())


@pytest.fixture(scope="module")
def smoke_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("项目")
    (root / "data").symlink_to(ROOT / "data")
    config = TrainConfig(model="cnn", model_args={"channels": [4, 8]}, epochs=1,
                         batch_size=512, threads=2)
    return train(config, root=root, name="冒烟", extra_splits=("challenge_shift",))


def test_train_saves_reproducible_run(smoke_run):
    assert smoke_run.metric() > 0.2
    assert smoke_run.record["reload_difference"] <= 1e-6
    assert {"config.json", "history.json", "evaluations.json", "model.pt",
            "curve.png", "summary.md"} <= {p.name for p in smoke_run.path.iterdir()}
    assert "challenge_shift" in smoke_run.evaluations


def test_runs_are_listed_and_compared(smoke_run):
    found, skipped = list_runs(smoke_run.path.parent.parent)
    assert [run.path.name for run in found] == [smoke_run.path.name]
    assert skipped == []


def test_predict_on_bundled_examples(smoke_run):
    from doodlelab import load_model
    model, _, labels, _ = load_model(smoke_run.path)
    prediction = predict_image(model, ROOT / "examples/synthetic_paper_apple.png", labels=labels)
    assert len(prediction["top"]) == 3
    assert prediction["prepared"]["tensor"].shape == (1, 1, 28, 28)
    assert abs(sum(prediction["scores"]) - 1) < 1e-5


def test_evaluate_matches_history(smoke_run):
    from doodlelab import load_model
    model, _, _, _ = load_model(smoke_run.path)
    store = DataStore(ROOT)
    ev = evaluate(model, store.load("val_basic", verify=False))
    assert ev["accuracy"] == pytest.approx(smoke_run.evaluations["val_basic"]["accuracy"], abs=1e-9)
