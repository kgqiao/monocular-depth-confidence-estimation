"""
**Core Error Metrics**: Metrics for the confidence head. All take labeled cells only (apply the mask first).

Inputs point two ways:
- Sparsification, AUSE, AURG take the true error (abs_rel) and an uncertainty score (p_error): higher = worse.
- AUROC, ECE (and Brier) take is_correct (1 - target) and a confidence (1 - p_error): higher = better.


**Ranking metrics**: Does the head flag the worst cells first?

1. Sparsification curve: drop the least-trusted cells a few at a time and track the mean error of what's
left. Three curves: the head's order, the oracle's (true error order, the best possible), and random
(flat at the mean error).

2. AUSE (area between the head's curve and the oracle's): how far the head is from flagging perfectly.
0 means it always drops the truly worst cells first. The lower, the better.

3. AURG (area between the random curve and the head's): how much better the head is than dropping cells
at random. 0 means the head adds nothing; below 0 means worse than random. Higher is better. Its units
are roughly "average error saved" across all cutoffs.

4. Rejection ROC-AUC: the chance that a correct cell (AbsRel <= tau) gets higher confidence than a wrong
one. 0.5 = guessing, 1.0 = every wrong cell scored below every correct one. Depends on tau, unlike AUSE
and AURG, which use the continuous error. AKA tests to see if higher confidence goes to the right cellss.


**Calibration metrics**: Do the confidence values match reality?

2. Reliability diagram: uses the same 10 confidence bins as ECE and plots each bin's accuracy (y) against
its mean confidence (x). The y = x diagonal is perfect calibration. Points below the diagonal mean the
head is overconfident (it says 0.8, and 0.6 are correct); points above mean underconfident. ECE is the
weighted average vertical gap between these points and the diagonal, so the diagram shows where the
ECE comes from.

3. Brier score: the mean squared gap between each cell's confidence and its 0/1 outcome (1 = correct).
0 = perfect, 0.25 = always saying 0.5, 1 = confidently wrong every time. Lower is better. Unlike ECE,
it rewards both calibration and ranking, so a head that says 0.5 for everything can have ECE near 0
but still scores 0.25 here. Depends on tau, like ROC-AUC and ECE.

Returns the per-bin values; plotting lives in ___
"""

import numpy as np


def compute_sparsification_curves(true_errors: np.ndarray, uncertainty_scores: np.ndarray, num_steps = 50):
    """
    Finds the gap between the head (sparsification curve) and the baseline (oracle curve) = aka the head's mistakes
    Gap between the random and head is what the head gained. Computes the sparsification curve, oracle curve, and random curve.
    
    Args:
        true_errors: 1D array of absolute errors per cell (e.g., abs_rel).
        uncertainty_scores: 1D array of predicted uncertainty scores per cell 
                            (higher means less trustworthy).
        num_steps = 50: practical default that evaluates curve every ~2% of 
                        removed data, balances smooth plotting and fast execution
                            
    Returns:
        fractions: 1D array of num_steps fractions from 0.0 to 0.99.
        method_curve: Mean error remaining at each fraction using the uncertainty scores.
        oracle_curve: Mean error remaining at each fraction if sorted perfectly by true error.
        random_curve: The baseline flat curve at the global mean error.
    """

    #Convert to arrays true errors and uncertainty scores, validate that arrays exist
    true_errors = np.asarray(true_errors)
    uncertainty_scores = np.asarray(uncertainty_scores)
    n = len(true_errors)
    if n == 0:
        raise ValueError("Input arrays cannot be empty.")

    # Sort indices from highest uncertainty error to lowest
    uncertainty_order = np.argsort(uncertainty_scores)[::-1]
    oracle_order = np.argsort(true_errors)[::-1]

    sorted_errors_uncertainty = true_errors[uncertainty_order] # Re Order true errors in order of descending uncertainty scores
    sorted_errors_oracle = true_errors[oracle_order] # Reorder true errors in order of descending oracle order (ideal baseline)

    # Define the fractions of data to remove (avoid 1.0 to prevent dividing by 0)
    fractions = np.linspace(0, 0.99, num_steps)

    # Initilaize sparsification curve and oracle curve
    sparsification_curve = []
    oracle_curve = []

    for frac in fractions: # for every fraction
        cutoff = int(np.round(frac * n))
        # Calculate mean error on the remaining (most confident) samples, add to each curve
        sparsification_curve.append(np.mean(sorted_errors_uncertainty[cutoff:])) 
        oracle_curve.append(np.mean(sorted_errors_oracle[cutoff:]))

    # Create Random Baseline Curve (Flat line at the overall mean error)
    global_mean_error = np.mean(true_errors)
    random_curve = np.full_like(fractions, global_mean_error)

    # Convert sparsification curve and oracle curve into arrays
    sparsification_curve = np.array(sparsification_curve)
    oracle_curve = np.array(oracle_curve)

    return fractions, sparsification_curve, oracle_curve, random_curve



def calculate_ause_and_aurg(fractions: np.ndarray, method_curve: np.ndarray, 
                            oracle_curve: np.ndarray, random_curve: np.ndarray):
    """
    Calculates AUSE (Area Under the Sparsification Error) and 
    AURG (Area Under the Gain Curve) using trapezoidal integration.
    """
    # AUSE: Area between method's curve and oracle curve (lower is better, 0 is perfect)
    ause = np.trapezoid(method_curve - oracle_curve, x=fractions)
    
    # AURG: Area between random curve and method's curve (higher is better, 0 is random)
    aurg = np.trapezoid(random_curve - method_curve, x=fractions)
    
    return float(ause), float(aurg)



def compute_ece(is_correct: np.ndarray, confidence_scores: np.ndarray, n_bins: int = 10) -> float:
    """Calculates Expected Calibration Error (ECE).
    
    1. Groups cellss into n_bins by confidence. 
    2. In each bin, compare the average confidence with the share that's actually correct. 
    3. Average those gaps weighted by how many cellss are in the bins (i.e. 0 = perfect)

    Args:     
        is_correct: 1 if cells's depth is correct, else 0 (is_correct = 1 - target)
        confidence_scores: higher = more trusted (confidence = 1 - p_error)
        n_bins: number of equal-width bins. Each bin includes its left edge and excludes its right edge. 
    
    - Pass only the cells that are labeled.
    - Train weighted wrong cellss by pos_weight = 1.66, so that p_error runs high and ECE is shifted to account for this.
    """
    bins = np.linspace(0, 1, n_bins + 1) # bin edges: 0, 0.1,..., 1.0
    ece = 0.0
    n_samples = len(confidence_scores)

    for i in range(n_bins):
        if i == n_bins - 1:
            upper = confidence_scores <= bins[i + 1] # last bin includes 1.0 (11 edges results in 10 bins)
        else:
            upper = confidence_scores < bins[i + 1] # last bin does NOT include 1.0

        mask = (confidence_scores >= bins[i]) & upper # which cellss fall in this bin
        
        bin_size = np.sum(mask)
        if bin_size > 0:
            # Calculate cellss in the bin that actually happened (bin_acc) vs what the model claimed (bin_conf)
            bin_acc = np.mean(is_correct[mask]) # Share actually correct cellss
            bin_conf = np.mean(confidence_scores[mask]) # What model said

            # Calculate addition of ECE, wherein (bin_size / n_samples)  means bins with more cellss count more
            # Empty bins are skipped as to avoid division by zero
            ece += (bin_size / n_samples) * np.abs(bin_acc - bin_conf)
            
    return float(ece)


#Fill in variables of is_correct, confidence scores
def calculate_rejection_auc(is_correct: np.ndarray, confidence_scores: np.ndarray) -> float:
    """Calculates Area Under the Rejection ROC Curve (AUROC)

    The share of (correct cells, wrong cells) pairs where the correct cells's confidence is higher.
    0.5 = random
    1.0 = every wrong cells scored below every correct one
    < 0.5 = inputs flipped

    Args:
        is_correct: 1 if cells's depth is correct, else 0 (is_correct = 1 - target)
        confidence_scores: higher = more trusted (confidence = 1 - p_error)

    Pass only the cellss that are labeled.
    If every cells is correct or every cells is wrong (aka no pairs to conmpare), return NaN

    Treats each labeled cells as yes/no: is the depth correct?
    """
    from sklearn.metrics import roc_auc_score

    if len(np.unique(is_correct)) < 2: #Guardrail incase only one class is present. 
        return float("nan")

    rejection_auc = float(roc_auc_score(is_correct, confidence_scores))
    return rejection_auc



def compute_brier_score(is_correct: np.ndarray, confidence_scores: np.ndarray) -> float:
    """
    Takes each cell’s gap between confidence and the 0/1 outcome, squares it, and averages

    0 = perfect. 0.25 = always saying 0.5. 1 = confidently wrong every time. Lower is better

    Comparing runs with a single number that punishes confident mistakes most
    
    """
    gaps = confidence_scores - is_correct  # per cell: how far the confidence is from the 0/1 outcome
    return float(np.mean(gaps ** 2)) # square each gap, then average over all cells


def compute_reliability_bins(is_correct: np.ndarray, confidence_scores: np.ndarray, n_bins: int = 10):
    """
    Groups cells into 10 confidence bins and plots what the head said (x) against how often it was correct (y)
    
    On the diagonal = calibrated. Below = overconfident. Above = underconfident

    Seeing where the head is off: which confidence range, and in which direction

    The Reliability Diagram ECE unpacked. ECE is the weighted average distance between 
    the diagram’s points and the diagonal. 
    ECE tells how miscalibrated the head is; the diagram tells where and which way.
    
    """
    bins = np.linspace(0, 1, n_bins + 1)  # same 11 edges as compute_ece: 0, 0.1, ..., 1.0
    bin_conf = np.full(n_bins, np.nan)  # mean confidence per bin; NaN marks an empty bin
    bin_acc = np.full(n_bins, np.nan)  # share correct per bin; NaN marks an empty bin
    bin_count = np.zeros(n_bins, dtype=int) # number of cells per bin

    for i in range(n_bins):
        if i == n_bins - 1:
            upper = confidence_scores <= bins[i + 1] # last bin includes 1.0
        else:
            upper = confidence_scores < bins[i + 1] # other bins exclude their right edge
        mask = (confidence_scores >= bins[i]) & upper # cells that fall in this bin

        bin_count[i] = np.sum(mask) # how many cells landed here
        if bin_count[i] > 0: # skip empty bins: the mean of nothing is undefined
            bin_conf[i] = np.mean(confidence_scores[mask]) # what the head said, on average
            bin_acc[i] = np.mean(is_correct[mask]) # how often it was actually correct

    return bin_conf, bin_acc, bin_count