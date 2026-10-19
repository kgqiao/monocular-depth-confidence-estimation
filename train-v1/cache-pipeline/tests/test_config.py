import tomllib

import pytest

from confcache.config import Config, load_config, to_toml


def write(tmp_path, text):
    p = tmp_path / "c.toml"
    p.write_text(text)
    return p


def test_defaults_load_without_a_file(monkeypatch):
    monkeypatch.delenv("NUSCENES_DATAROOT", raising=False)
    cfg = load_config()
    assert cfg.model.tap == "output_conv1"
    assert cfg.temporal.offsets == [-2, -1, 1, 2]


def test_file_then_set_and_paths_resolve_against_the_file(tmp_path):
    p = write(tmp_path, '[model]\nshort_side = 280\n[output]\nroot = "out"\n[source]\ndataroot = "data"\n')
    cfg = load_config(p, ["temporal.offsets=[-1, 1]", "run.precision=fp16", "points.max_depth=60"])
    assert cfg.model.short_side == 280
    assert cfg.temporal.offsets == [-1, 1]
    assert cfg.run.precision == "fp16"
    assert cfg.points.max_depth == 60.0 and isinstance(cfg.points.max_depth, float)
    assert cfg.output.root == str((tmp_path / "out").resolve())
    assert cfg.source.dataroot == str((tmp_path / "data").resolve())


def test_environment_fills_empty_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("NUSCENES_DATAROOT", str(tmp_path))
    monkeypatch.setenv("DEPTH_ANYTHING_CHECKPOINT", str(tmp_path / "m.pth"))
    cfg = load_config()
    assert cfg.source.dataroot == str(tmp_path.resolve())
    assert cfg.model.checkpoint == str((tmp_path / "m.pth").resolve())


@pytest.mark.parametrize("text, message", [
    ("[model]\ntapp = 1\n", "unknown setting model.tapp"),
    ("[modell]\ntap = 1\n", "unknown section"),
    ("[model]\nshort_side = \"big\"\n", "must be an integer"),
    ("[model]\nshort_side = 500\n", "multiple of the 14 px patch"),
    ("[temporal]\noffsets = [0, 1]\n", "must not hold 0"),
    ("[temporal]\nenabled = 1\n", "true or false"),
    ("[splits]\ntrain = [\"a\"]\nval = [\"a\"]\n", "more than one tag"),
])
def test_bad_settings_name_the_setting(tmp_path, text, message):
    with pytest.raises(ValueError, match=message):
        load_config(write(tmp_path, text))


def test_set_needs_section_and_key():
    with pytest.raises(ValueError, match="section.key=value"):
        load_config(None, ["tap=refinenet1"])


def test_fingerprint_follows_content_only():
    a, b = Config(), Config()
    b.run.batch_size = 99
    b.source.scenes = ["scene-0061"]
    b.output.root = "/elsewhere"
    b.splits = {"train": ["scene-0061"]}
    assert a.fingerprint() == b.fingerprint()
    b.model.tap = "refinenet1"
    assert a.fingerprint() != b.fingerprint()
    c = Config()
    c.run.precision = "fp16"
    assert a.fingerprint() != c.fingerprint()


def test_toml_round_trip(tmp_path):
    cfg = load_config(None, ["temporal.offsets=[-3, 3]", "model.tap=refinenet2"])
    cfg.splits = {"train": ["scene-0061"], "val": ["scene-0103"]}
    text = to_toml(cfg)
    tomllib.loads(text)
    again = load_config(write(tmp_path, text))
    assert again.fingerprint() == cfg.fingerprint()
    assert again.splits == cfg.splits
