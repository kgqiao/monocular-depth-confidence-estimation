"""Core math functions listed here.

1. Sparsification Curve 

2. AUSE (area between the head’s curve and the oracle’s): how far the head is from flagging perfectly. 
0 means it always drops the truly worst cells first.
3. AURG (area between the random curve and the head’s): how much better the head is than dropping cells at random. 
0 means the head adds nothing. Its units are roughly “average error saved” across all cutoffs.


**Calibration Metrics**: Do the scores match reality?

1. Expected Calibration Error (ECE): bins predictions by confidence level and calculates weighted average
difference between accuracy and confidence aacross all bins. (Lower is better)

2. Maximum Calibration Error (MCE): Focuses on the worst-case scenario by identifying the maximum deviation 
between confidence and accuracy in any single bin.

3. Reliability Diagrams: A visual chart plotting predicted confidence (x-axis) against actual accuracy 
(y-axis). A perfectly calibrated confidence head forms a perfect y = x diagonal line.

4. Brier Score: Measures the mean squared error between the predicted confidence score and the actual 
binary outcome (1 for correct, 0 for incorrect).


**Selective Classification & Rejection Metrics**: Can it filter bad predictions?

(These metrics would measure which examples are too low confidence to predict or handoff as
reliable results.)

1. Risk-Coverage Curve (RC Curve): Plots the model's error rate (Risk) against the percentage of data kept 
(Coverage) as you vary the confidence threshold.
    - EXPECTED OUTPUT: Risk should drop sharply as coverage decreases.

2. Area Under the Risk-Coverage Curve (AURC): Quantifies the entire RC curve into a single metric. A lower 
AURC means the confidence head is highly effective at filtering out failures early.

3. Rejection ROC-AUC: Treat the evaluation as a binary classification problem where the goal is to predict 
whether the main model's output is correct (1) or incorrect (0).
    - Calculate the Area Under the ROC curve using the confidence score as the predictor. 
    - AUC of 0.5 means confidence head is guessing randomly;
    - AUC of 1.0 means confidence head perfectly flags every error.


**Distributional & Out-of-Distribution (OOD) Robustness**: Can the conf head flag correctly on unseen test data?

1. OOD Detection Score (FPR at TPR 95%): Evaluate the confidence head on a completely separate out-of-
distribution dataset. Calculated from the measurements: 
    - False Positive Rate (how many OOD samples get high confidence scores) when the True Positive Rate 
    for in-distribution data is held at 95%.

2. Confidence Drift under Covariate Shift: Evaluate the head on data with synthetic corruptions 
(e.g., adding noise, blur, or changing lighting). The average confidence score should degrade 
proportionally to the drop in actual accuracy.

"""

import numpy as np

#Fill in variables of is_correct, confidence scores
def calculate_rejection_auc(is_correct: np.ndarray, confidence_scores: np.ndarray) -> float:
    """Calculates Area Under the Rejection ROC Curve (AUROC)

    The share of (correct pixel, wrong pixel) pairs where the correct pixel's confidence is higher.
    0.5 = random
    1.0 = every wrong pixel scored below every correct one
    < 0.5 = inputs flipped

    Args:
        is_correct: 1 if pixel's depth is correct, else 0 (is_correct = 1 - target)
        confidence_scores: higher = more trusted (confidence = 1 - p_error)

    Pass only the pixels that are labeled.
    If every pixel is correct or every pixel is wrong (aka no pairs to conmpare), return NaN

    Treats each labeled pixel as yes/no: is the depth correct?
    """
    from sklearn.metrics import roc_auc_score

    rejection_auc = float(roc_auc_score(is_correct, confidence_scores))
    return rejection_auc


def compute_ece(is_correct: np.ndarray, confidence_scores: np.ndarray, n_bins: int = 10) -> float:
    """Calculates Expected Calibration Error (ECE).
    
    1. Groups pixels into n_bins by confidence. 
    2. In each bin, compare the average confidence with the share that's actually correct. 
    3. Average those gaps weighted by how many pixels are in the bins (i.e. 0 = perfect)

    Args:     
        is_correct: 1 if pixel's depth is correct, else 0 (is_correct = 1 - target)
        confidence_scores: higher = more trusted (confidence = 1 - p_error)
        n_bins: number of equal-width bins. Each bin includes its left edge and excludes its right edge. 
    
    - Pass only the pixels that are labeled.
    - Train weighted wrong pixels by pos_weight = 1.66, so that p_error runs high and ECE is shifted to account for this.
    """
    bins = np.linspace(0, 1, n_bins + 1) # bin edges: 0, 0.1,..., 1.0
    ece = 0.0
    n_samples = len(confidence_scores)

    for i in range(n_bins):
        if i == n_bins - 1:
            upper = confidence_scores <= bins[i + 1] # last bin includes 1.0 (11 edges results in 10 bins)
        else:
            upper = confidence_scores < bins[i + 1] # last bin does NOT include 1.0

        mask = (confidence_scores > bins[i]) & upper # which pixels fall in this bin
        
        bin_size = np.sum(mask)
        if bin_size > 0:
            # Calculate pixels in the bin that actually happened (bin_acc) vs what the model claimed (bin_conf)
            bin_acc = np.mean(is_correct[mask]) # Share actually correct pixels
            bin_conf = np.mean(confidence_scores[mask]) # What model said

            # Calculate addition of ECE, wherein (bin_size / n_samples)  means bins with more pixels count more
            # Empty bins are skipped as to avoid division by zero
            ece += (bin_size / n_samples) * np.abs(bin_acc - bin_conf)
            
    return float(ece)


def compute_sparsification_curves(true_errors: np.ndarray, uncertainty_scores: np.ndarray, num_steps = 50):
    """
    Finds the gap between the head (sparsification curve) and the baseline (oracle curve) = aka the head's mistakes
    Gap between the random and head is what the head gained. 

    Computes the sparsification curve, oracle curve, and random curve.
    
    Args:
        true_errors: 1D array of absolute errors per cell (e.g., abs_rel).
        uncertainty_scores: 1D array of predicted uncertainty scores per cell 
                            (higher means less trustworthy).
        num_steps = 50: practical default that evaluates curve every ~2% of 
                        removed data, balances smooth plotting and fast execution
                            
    Returns:
        fractions: 1D array of 100 fractions from 0.0 to 0.99.
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


