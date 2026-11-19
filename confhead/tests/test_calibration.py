"""
Build test metrics for the following cases: AUROC, ECE, Brier, reliability bins

"""

import numpy as np
from evals.metrics import calculate_rejection_auc, compute_ece, compute_brier_score, compute_reliability_bins

# ---------- AUROC: does higher confidence go to the correct cells? ----------

def test_auroc_perfect_separation_gives_one(): #Perfect separation results in AUROC = 1.0
    is_correct = np.array([1, 1, 1, 0, 0, 0]) # first 3 cells correct, last 3 wrong
    confidence = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1]) # every correct cell scores above every wrong one
    result = calculate_rejection_auc(is_correct, confidence) # call the function
    assert np.isclose(result, 1.0) # perfect ranking must give exactly 1.0

def test_auroc_reversed_gives_zero(): # Reversed result gives AUROC = 0.0. Result that risks passing p_error where confidence belongs
    is_correct = np.array([1, 1, 1, 0, 0, 0]) # Present 6 cells with its correctness value and confidence level (3 correct with low confidence, 3 incorrect with high confidence)
    confidence = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9]) # flipped: wrong cells get the high scores
    result = calculate_rejection_auc(is_correct, confidence)
    assert np.isclose(result, 0.0) 
    
def test_auroc_random_gives_about_half(): # Random separation results in AUROC = ~0.5
    rng = np.random.default_rng(0) # Initializes fixed seed
    is_correct = rng.integers(0, 2, size = 10_000) # Gives random 0/1 labels
    confidence = rng.random(10_000) # Gives random scores, unrelated to the labels
    result = calculate_rejection_auc(is_correct, confidence)
    assert abs(result - 0.5) < 0.02

def test_auroc_one_class_gives_nan():
    is_correct = np.ones(5, dtype = int) # Where every cell is correct, thus no wrong cells to compare with
    confidence = np.array([0.9, 0.8, 0.7, 0.6, 0.5]) # Varying levels of confidence
    result = calculate_rejection_auc(is_correct, confidence)
    assert np.isnan(result)  # AUROC is undefined here; must be NaN


# ---------- ECE: do the confidence values match how often cells are correct? ----------

def test_ece_calibrated_scores_give_about_zero():
    rng = np.random.default_rng(1) # Initializes fixed seed
    confidence = rng.random(100_000) # scores spread over 0 to 1
    is_correct = (rng.random(100_000) < confidence).astype(int) # a cell with score 0.7 is correct 70% of the time: calibrated by construction
    result = compute_ece(is_correct, confidence)
    assert result < 0.01  # near 0, not exactly 0, because the data is random


def test_ece_hand_example():
    is_correct = np.array([0, 0, 1, 0]) # 4 cells, small enough to work out on paper
    confidence = np.array([0.2, 0.2, 0.8, 0.8]) # two cells in the 0.2 bin, two in the 0.8 bin
    # 0.2 bin: accuracy 0/2 = 0.0, confidence 0.2, gap 0.2, weight 2/4
    # 0.8 bin: accuracy 1/2 = 0.5, confidence 0.8, gap 0.3, weight 2/4
    # ECE = 0.5 * 0.2 + 0.5 * 0.3 = 0.25
    result = compute_ece(is_correct, confidence)
    assert np.isclose(result, 0.25) # must match the hand calculation


def test_ece_counts_a_score_of_one():
    is_correct = np.zeros(4, dtype=int)  # every cell wrong
    confidence = np.ones(4)  # every score exactly 1.0, the top edge of the last bin
    result = compute_ece(is_correct, confidence)
    assert np.isclose(result, 1.0)  # gap is |0 - 1| = 1; a 0.0 here means the last bin dropped the 1.0s


# ---------- Brier score: how far is each cell's confidence from what actually happened? ----------

def test_brier_perfect_gives_zero():
    is_correct = np.array([1, 1, 0, 0])
    confidence = np.array([1.0, 1.0, 0.0, 0.0]) # fully sure and right every time
    assert np.isclose(compute_brier_score(is_correct, confidence), 0.0)


def test_brier_confidently_wrong_gives_one():
    is_correct = np.array([1, 1, 0, 0])
    confidence = np.array([0.0, 0.0, 1.0, 1.0])  # fully sure and wrong every time
    assert np.isclose(compute_brier_score(is_correct, confidence), 1.0)


def test_brier_always_half_gives_quarter():
    is_correct = np.array([1, 0, 1, 0])
    confidence = np.full(4, 0.5)  # every gap is 0.5, and 0.5 squared is 0.25
    assert np.isclose(compute_brier_score(is_correct, confidence), 0.25)



# ---------- Reliability diagram: in each confidence bin, how often are the cells actually correct? ----------

def test_reliability_hand_example():
    is_correct = np.array([0, 0, 1, 0]) # same 4 cells as the ECE hand example
    confidence = np.array([0.2, 0.2, 0.8, 0.8])
    bin_conf, bin_acc, bin_count = compute_reliability_bins(is_correct, confidence)
    assert bin_count.tolist() == [0, 0, 2, 0, 0, 0, 0, 0, 2, 0] # two cells in bin 2, two in bin 8
    assert np.isclose(bin_conf[2], 0.2) and np.isclose(bin_acc[2], 0.0)
    assert np.isclose(bin_conf[8], 0.8) and np.isclose(bin_acc[8], 0.5)
    assert np.isnan(bin_acc[0])  # empty bins are NaN, not 0


def test_reliability_counts_every_cell_once():
    confidence = np.array([0.0, 0.1, 0.35, 0.5, 0.99, 1.0])  # includes both ends, 0.0 and 1.0, and an edge, 0.1
    is_correct = np.array([0, 1, 0, 1, 1, 1])
    _, _, bin_count = compute_reliability_bins(is_correct, confidence)
    assert bin_count.sum() == len(confidence)  # no cell dropped, no cell counted twice


def test_reliability_bins_add_up_to_ece():
    rng = np.random.default_rng(2)
    confidence = rng.random(5_000)
    is_correct = rng.integers(0, 2, size=5_000)
    bin_conf, bin_acc, bin_count = compute_reliability_bins(is_correct, confidence)
    filled = bin_count > 0
    weights = bin_count[filled] / bin_count.sum() # each bin's share of the cells
    from_bins = np.sum(weights * np.abs(bin_acc[filled] - bin_conf[filled]))
    assert np.isclose(from_bins, compute_ece(is_correct, confidence))  # the diagram and the ECE must tell the same story

