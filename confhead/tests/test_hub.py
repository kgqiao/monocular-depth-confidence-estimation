import json
from types import SimpleNamespace

import httpx
import pytest
from huggingface_hub.errors import HfHubHTTPError

from confhead import hub
from confhead.hub import RunRecord, model_card, upload_run
from confhead.tracking import run_config, run_points
from confhead.train import Training
from test_train import make_cache, small_config


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """A finished run folder of 2 epochs."""
    tmp = tmp_path_factory.mktemp("hub")
    cfg = small_config(make_cache(tmp / "cache"), tmp / "runs")
    cfg.output.group = "baseline"
    Training(cfg).fit()
    return tmp / "runs/test"


def test_the_same_runs_give_the_same_card(run):
    a = {"test": RunRecord.read(run), "other": RunRecord.read(run)}
    b = dict(reversed(a.items()))
    assert model_card("u/r", a) == model_card("u/r", b)
    card = model_card("u/r", a)
    assert card.index("### other") < card.index("### test")
    assert "| [test](#test) | baseline |" in card and "| 2 of 2 |" in card
    assert "hf download u/r <run>/best.pt" in card


def test_a_partial_run_shows_its_epochs(run):
    record = RunRecord.read(run)
    record.history = record.history[:1]
    assert "| 1 of 2 |" in model_card("u/r", {"test": record})


def test_an_unfinished_run_needs_partial(run, tmp_path):
    partial = tmp_path / "partial"
    partial.mkdir()
    for f in run.iterdir():
        (partial / f.name).write_bytes(f.read_bytes())
    history = json.loads((run / "history.json").read_text())
    (partial / "history.json").write_text(json.dumps(history[:1]))
    with pytest.raises(ValueError, match="holds 1 of 2 epochs"):
        upload_run(partial, "u/r", api=SimpleNamespace())
    (partial / "best.pt").unlink()
    with pytest.raises(ValueError, match="has no best.pt"):
        upload_run(partial, "u/r", partial=True, api=SimpleNamespace())


class FakeApi:
    """A repository whose first `conflicts` commits lose to another upload."""

    def __init__(self, conflicts: int):
        self.conflicts, self.commits = conflicts, []

    def create_repo(self, repo_id, exist_ok):
        pass

    def repo_info(self, repo_id):
        return SimpleNamespace(sha=f"{len(self.commits):040x}")

    def create_commit(self, repo_id, operations, commit_message, parent_commit):
        self.commits.append(([op.path_in_repo for op in operations], parent_commit))
        if len(self.commits) <= self.conflicts:
            response = httpx.Response(412, request=httpx.Request("POST", "https://huggingface.co"))
            raise HfHubHTTPError("412 Precondition Failed", response=response)
        return SimpleNamespace(commit_url="url")


def test_an_upload_retries_after_another_upload_gets_ahead(run, monkeypatch):
    monkeypatch.setattr(hub, "download_runs", lambda *a, **k: {})
    api = FakeApi(conflicts=2)
    assert upload_run(run, "u/r", api=api).commit_url == "url"
    paths, parent = api.commits[-1]
    assert len(api.commits) == 3 and parent == f"{2:040x}"  # each try names the state it read
    assert sorted(paths) == ["README.md", "test/best.pt", "test/config.json", "test/history.json",
                             "test/last.pt", "test/steps.json"]
    with pytest.raises(HfHubHTTPError):
        upload_run(run, "u/r", api=FakeApi(conflicts=9), attempts=3)


def test_trackio_points_hold_the_steps_and_each_epoch_end(run):
    record = RunRecord.read(run)
    points = run_points(record)
    assert [step for step, _ in points] == list(range(1, 9))  # 4 steps per epoch
    assert set(points[0][1]) == {"train/loss"}
    assert set(points[3][1]) == {"train/loss", "epoch", "train/epoch_loss", "dev/loss", "lr", "minutes"}
    assert run_config(record)["output.group"] == "baseline"

    record.steps = None  # a run from before steps.json
    assert [(step, m["epoch"]) for step, m in run_points(record)] == [(4, 1), (8, 2)]
