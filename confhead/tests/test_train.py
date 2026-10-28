import json

import numpy as np
import pytest

from confhead import HeadInputs, LabelConfig, load_head
from confhead.cache.config import CacheConfig
from confhead.cache.store import build_index, open_for_build, write_entry
from confhead.train import TrainConfig, Training, dev_scenes, load_train_config
from test_cache import make_entry

SCENES = {"scene-a": "day", "scene-b": "day", "scene-c": "day", "scene-d": "day", "scene-e": "night", "scene-f": "night"}


def make_cache(root):
    """Six train scenes of two images each, with the 64 feature channels of the default tap."""
    cfg = CacheConfig()
    cfg.output.root = str(root)
    cfg.splits = {"train": sorted(SCENES)}
    root = open_for_build(cfg, overwrite=False)
    rng = np.random.default_rng(0)
    for scene, condition in SCENES.items():
        for i in range(2):
            arrays, meta = make_entry(scene=scene, key=f"{scene}_{i:03d}_CAM_FRONT")
            arrays["features"] = rng.normal(size=(64, 4, 8)).astype(np.float16)
            arrays["lidar"][0, 5] = 0.5  # predicted 20 m against 10 m: one cell above tau
            meta.update(condition=condition, index=i, fingerprint=cfg.fingerprint())
            write_entry(root, scene, meta["key"], arrays, meta)
    build_index(root, cfg.splits)
    return root


def small_config(cache, runs, **train):
    cfg = TrainConfig()
    cfg.data.cache, cfg.output.runs, cfg.output.name = str(cache), str(runs), "test"
    cfg.head.widths = [4]
    cfg.run.device, cfg.run.num_workers = "cpu", 0
    cfg.train.epochs, cfg.train.batch_size, cfg.train.dev_fraction = 2, 2, 0.25
    for key, value in train.items():
        setattr(cfg.train, key, value)
    return cfg


def test_config_takes_the_file_then_nested_overrides(tmp_path):
    path = tmp_path / "train.toml"
    path.write_text('[data]\ncache = "cache"\n[inputs.labels]\ntau = 0.2\n[train]\nepochs = 3\n')
    cfg = load_train_config(path, ["inputs.radar=false", "inputs.labels.align=radar", "train.lr=0.01"])
    assert cfg.data.cache == str(tmp_path / "cache")
    assert cfg.inputs == HeadInputs(radar=False, labels=LabelConfig(tau=0.2, align="radar"))
    assert (cfg.train.epochs, cfg.train.lr, cfg.train.batch_size) == (3, 0.01, 8)


@pytest.mark.parametrize("text, message", [
    ("[data]\ncache = 'c'\n[model]\nx = 1\n", r"unknown section \[model\]"),
    ("[data]\ncache = 'c'\n[train]\nepoch = 1\n", "unknown setting train.epoch"),
    ("[data]\ncache = 'c'\n[train]\nepochs = 'many'\n", "train.epochs must be an integer"),
    ("[data]\ncache = 'c'\n[inputs.labels]\nalign = 'gps'\n", "inputs.labels.align 'gps'"),
    ("[train]\nepochs = 2\n", "data.cache is empty"),
])
def test_bad_settings_name_the_setting(tmp_path, text, message):
    path = tmp_path / "train.toml"
    path.write_text(text)
    with pytest.raises(ValueError, match=message):
        load_train_config(path)


def test_dev_takes_a_share_of_each_condition_and_weather():
    rows = [{"scene": s, "condition": c, "weather": "clear"} for s, c in SCENES.items()]
    rows.append({"scene": "scene-g", "condition": "night", "weather": "rain"})
    dev = dev_scenes(rows, 0.25, seed=0)
    assert dev == dev_scenes(rows, 0.25, seed=0)
    assert len([s for s in dev if SCENES.get(s) == "day"]) == 1
    assert len([s for s in dev if SCENES.get(s) == "night"]) == 1
    assert "scene-g" not in dev  # a group of one scene stays in train


def losses(history):
    return [(h["epoch"], h["train_loss"], h["dev_loss"], h["lr"]) for h in history]


def test_a_run_writes_its_checkpoints(tmp_path):
    cache = make_cache(tmp_path / "cache")
    training = Training(small_config(cache, tmp_path / "runs"))
    assert [s["scenes"] for s in training.splits()] == [4, 2]
    history = training.fit()
    run = tmp_path / "runs/test"
    assert [h["epoch"] for h in history] == [1, 2]
    assert {p.name for p in run.iterdir()} == {"config.json", "history.json", "best.pt", "last.pt"}
    config = json.loads((run / "config.json").read_text())
    assert config["dev_scenes"] == training.dev_scenes and config["settings"]["train"]["epochs"] == 2

    head, inputs, info = load_head(run / "best.pt")
    assert inputs == training.cfg.inputs and info["channels"] == training.channels
    assert info["cache"]["fingerprint"] == training.cache.header["fingerprint"]
    assert info["history"][-1]["dev_loss"] == min(h["dev_loss"] for h in history)
    assert losses(Training(small_config(cache, tmp_path / "runs")).fit()) == losses(history)  # nothing left to train


def test_a_run_step_by_step_trains_as_one_fit(tmp_path):
    cache = make_cache(tmp_path / "cache")
    whole = small_config(cache, tmp_path / "runs")
    whole.output.name = "whole"
    expected = Training(whole).fit()
    steps = Training(small_config(cache, tmp_path / "runs"))
    steps.input_stats()
    steps.new_head()
    steps.speed(batches=2, steps=1)
    assert losses(steps.fit()) == pytest.approx(losses(expected))


def test_a_stopped_run_continues_as_if_it_never_stopped(tmp_path, monkeypatch):
    cache = make_cache(tmp_path / "cache")
    whole = small_config(cache, tmp_path / "runs", epochs=3)
    whole.output.name = "whole"
    expected = Training(whole).fit()

    calls = []
    dev_loss = Training.dev_loss

    def stop_in_epoch_3(self, *args):
        calls.append(1)
        if len(calls) == 3:
            raise KeyboardInterrupt
        return dev_loss(self, *args)

    monkeypatch.setattr(Training, "dev_loss", stop_in_epoch_3)
    with pytest.raises(KeyboardInterrupt):
        Training(small_config(cache, tmp_path / "runs", epochs=3)).fit()
    monkeypatch.undo()
    continued = Training(small_config(cache, tmp_path / "runs", epochs=3)).fit()
    assert losses(continued) == pytest.approx(losses(expected))

    with pytest.raises(ValueError, match="train.lr 0.001 -> 0.01"):
        Training(small_config(cache, tmp_path / "runs", epochs=3, lr=0.01)).fit()
