import itertools

import numpy as np
import pytest
import torch

from confhead.head import ConfidenceHead, masked_bce
from confhead.metrics import auroc, ause_aurg, sparsification


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


def test_auroc_matches_pair_counting():
    rng = np.random.default_rng(0)
    scores = rng.integers(0, 5, 60).astype(float)  # many ties
    labels = rng.random(60) < 0.4
    pairs = [1.0 if p > n else 0.5 if p == n else 0.0
             for p, n in itertools.product(scores[labels], scores[~labels])]
    assert auroc(scores, labels) == pytest.approx(np.mean(pairs))
    assert auroc([1, 2, 3, 4], [0, 0, 1, 1]) == 1.0
    assert auroc([4, 3, 2, 1], [0, 0, 1, 1]) == 0.0
    assert np.isnan(auroc([1, 2], [1, 1]))


def test_sparsification_of_the_oracle():
    errors = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(sparsification(errors, errors, steps=4), [2.5, 2.0, 1.5, 1.0])
    ause, aurg = ause_aurg(errors, errors, steps=4)
    assert ause == pytest.approx(0.0)
    assert aurg == pytest.approx(2.5 - 1.75)


def test_a_reversed_ranking_is_worse_than_random():
    errors = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(sparsification(errors, -errors, steps=4), [2.5, 3.0, 3.5, 4.0])
    ause, aurg = ause_aurg(errors, -errors, steps=4)
    assert ause == pytest.approx(3.25 - 1.75)
    assert aurg == pytest.approx(2.5 - 3.25)
