"""
1. Imports frozen DepthAnythingV2, confidence head
2. Loads dataset (NuScenes inside src)
3. Runs inference
4. Calls metrics (/metrics.py)

"""


import numpy as np
from evals.metrics import calculate_rejection_auc, compute_ece
from confhead.nuscenes import NuScenes

def main():
    print("Running Custom Confidence Head Evaluations...")

    #1. Load model, run over validation/test split


    #2. Collect ground truth depth, predicted depth, confidenc escore



    #3. For Monocular Depth: determine "correctness" based on a threshold
    #(e.g., relative error < 1.25 delta threshold, or absolute error below a limit)
    absolute_errors = 0.0
    is_correct = (absolute_errors < 0.1).astype(int)  # <---- Variables to add in later
    is_correct = np.array([1, 1, 0, 1, 0]) # <---- Variables to add in later
    confidence_scores = np.array([0.9, 0.85, 0.3, 0.95, 0.4]) # <---- Variables to add in later



    #4. Calculate metrics using metric functions in /metrics.py
    auc = calculate_rejection_auc(is_correct, confidence_scores)\
    ece = compute_ece(is_correct, confidence_scores)

    print(f"--- Results ---")
    print(f"Rejection ROC-AUC: {auc:.4f}")
    print(f"Expected Calibration Error (ECE): {ece:.4f}")


    # NOTE: add random tie breaking



    # Load data


    # Add Per condition rpeporting and bootstrampping







if __name__ == "__main__":
    main()