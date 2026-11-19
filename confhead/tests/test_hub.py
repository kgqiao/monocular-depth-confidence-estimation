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


def other_model(run):
    record = RunRecord.read(run)
    record.config["train_images"] += 100
    return record


def test_an_upload_replaces_another_model_only_on_purpose(run, monkeypatch):
    monkeypatch.setattr(hub, "download_runs", lambda *a, **k: {"test": other_model(run)})
    api = FakeApi(conflicts=0)
    with pytest.raises(ValueError, match="holds a run test with another fingerprint"):
        upload_run(run, "u/r", api=api)
    assert not api.commits
    assert upload_run(run, "u/r", api=api, replace=True).commit_url == "url"

    monkeypatch.setattr(hub, "download_runs", lambda *a, **k: {"test": RunRecord.read(run)})
    assert upload_run(run, "u/r", api=api).commit_url == "url"  # the same model: an epoch more, or the same files


def test_deleting_runs_takes_them_off_the_card(run, monkeypatch):
    monkeypatch.setattr(hub, "download_runs", lambda *a, **k: {"trial-a": RunRecord.read(run), "keep": RunRecord.read(run)})
    cards = []
    monkeypatch.setattr(hub, "model_card", lambda repo, runs: cards.append(sorted(runs)) or "card")
    api = FakeApi(conflicts=0)
    hub.delete_runs("u/r", ["trial-a"], api=api)
    assert cards == [["keep"]] and api.commits[0][0] == ["trial-a/", "README.md"]


class AccessApi:
    def __init__(self, writes: bool):
        self.writes, self.branches = writes, []

    def whoami(self):
        return {"name": "me", "orgs": [{"name": "team", "roleInOrg": "read"}], "auth": {"accessToken": {"role": "write"}}}

    def repo_exists(self, repo_id):
        return repo_id == "team/r"

    def create_branch(self, repo_id, branch, exist_ok):
        if not self.writes:
            response = httpx.Response(403, request=httpx.Request("POST", "https://huggingface.co"))
            raise HfHubHTTPError("403 Forbidden", response=response)
        self.branches.append(branch)

    def delete_branch(self, repo_id, branch):
        self.branches.remove(branch)


def test_write_access_is_proved_by_a_write(run):
    api = AccessApi(writes=True)
    assert hub.check_write_access("team/r", api) == "me" and not api.branches
    with pytest.raises(ValueError, match=r"cannot write to team/r \(HTTP 403\). Role in team: read; token role: write"):
        hub.check_write_access("team/r", AccessApi(writes=False))
    with pytest.raises(ValueError, match="does not exist"):
        hub.check_write_access("team/other", api)


def test_a_tag_keeps_the_state_before_a_replace():
    tags = []
    api = SimpleNamespace(repo_info=lambda repo: SimpleNamespace(sha="abc"),
                          list_repo_refs=lambda repo: SimpleNamespace(tags=tags),
                          create_tag=lambda repo, tag, revision: tags.append(SimpleNamespace(name=tag, target_commit=revision)))
    assert hub.tag_state("u/r", "before", api) == "before"
    assert hub.tag_state("u/r", "again", api) is None and len(tags) == 1


def test_trackio_points_hold_the_steps_and_each_epoch_end(run):
    record = RunRecord.read(run)
    points = run_points(record)
    assert [step for step, _ in points] == list(range(1, 9))  # 4 steps per epoch
    assert set(points[0][1]) == {"train/loss"}
    assert set(points[3][1]) == {"train/loss", "epoch", "train/epoch_loss", "dev/loss", "lr", "minutes"}
    assert run_config(record)["output.group"] == "baseline"

    record.steps = None  # a run from before steps.json
    assert [(step, m["epoch"]) for step, m in run_points(record)] == [(4, 1), (8, 2)]
