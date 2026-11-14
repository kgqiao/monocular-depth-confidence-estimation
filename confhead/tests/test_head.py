import pytest
import torch

from confhead import HeadInputs, LabelConfig, RadarConfig
from confhead.head import ConfidenceHead, load_head, masked_bce, save_head


def test_head_gives_one_logit_per_cell():
    head = ConfidenceHead(6, widths=(8, 4))
    assert head(torch.randn(2, 6, 10, 12)).shape == (2, 1, 10, 12)


def test_input_stats_standardize_x():
    torch.manual_seed(0)
    head = ConfidenceHead(3, widths=(4,))
    x = torch.randn(1, 3, 5, 5) * 7 + 3
    mean, std = torch.tensor([3.0, 1.0, -2.0]), torch.tensor([7.0, 2.0, 0.0])
    plain = head(((x - mean.view(-1, 1, 1)) / torch.tensor([7.0, 2.0, 1.0]).view(-1, 1, 1)))
    head.set_input_stats(mean, std)
    assert torch.allclose(head(x), plain, atol=1e-6)
    assert {"mean", "std"} <= head.state_dict().keys()


def test_masked_bce_counts_only_labeled_cells():
    logits = torch.tensor([[[[2.0, -1.0, 9.0]]]])
    target = torch.tensor([[[[1.0, 0.0, 0.0]]]])
    mask = torch.tensor([[[[1.0, 1.0, 0.0]]]])
    expected = (torch.nn.functional.softplus(torch.tensor(-2.0)) + torch.nn.functional.softplus(torch.tensor(-1.0))) / 2
    assert masked_bce(logits, target, mask).item() == pytest.approx(expected.item())
    weighted = (3 * torch.nn.functional.softplus(torch.tensor(-2.0)) + torch.nn.functional.softplus(torch.tensor(-1.0))) / 2
    assert masked_bce(logits, target, mask, pos_weight=3.0).item() == pytest.approx(weighted.item())
    assert masked_bce(logits, target, torch.zeros_like(mask)).item() == 0.0


def test_a_saved_head_loads_with_its_inputs(tmp_path):
    torch.manual_seed(0)
    head = ConfidenceHead(5, widths=(4, 3))
    head.set_input_stats(torch.arange(5.0), torch.full((5,), 2.0))
    inputs = HeadInputs(radar=False, labels=LabelConfig(tau=0.2), radar_cfg=RadarConfig(radius=2))
    channels = ["a", "b", "c", "d", "e"]
    save_head(tmp_path / "head.pt", head, inputs, channels, epoch=3, history=[{"dev_loss": 0.5}])

    loaded, loaded_inputs, fields = load_head(tmp_path / "head.pt")
    x = torch.randn(2, 5, 6, 7)
    assert torch.equal(loaded(x), head(x))
    assert not loaded.training
    assert loaded_inputs == inputs
    assert fields == {"channels": channels, "epoch": 3, "history": [{"dev_loss": 0.5}]}
