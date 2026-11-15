"""
1. Imports frozen DepthAnythingV2, confidence head
2. Loads dataset (NuScenes inside src)
3. Runs inference
4. Calls metrics (/metrics.py)

"""


import numpy as np
from evals.metrics import calculate_rejection_auc, compute_ece
from confhead.nuscenes import NuScenes
from evals.bevcar_splits import BEVCAR_CONDITION
from collections import defaultdict

def main():
    print("Running Custom Confidence Head Evaluations...")

    #1. Load model, run over validation/test split


    #2. Collect ground truth depth, predicted depth, confidenc escore

    # Wire the bevCar condition into loop 


    results = defaultdict(lambda: {"errors": [], "confs": []})

    # for row in cache.rows(tag="val"):
    #      scene_name = row["scene"]
    #      bev_group = BEVCAR_CONDITION.get(scene_name, "unknown")

    #      # run inference, get abs_rel and confidence 

    #      #Accumulate per group 

    #      results[bev_group]["errors"].append(abs_rel_flat)
    #      results[bev_group]["confs"].append(conf_flat)

    #      # After loop: compute all metrics per group
    # for group in ("day", "rain", "night", "all"):
    #      errors = np.concatenate(results[group]["errors"])
    #      confs  = np.concatenate(results[group]["confs"])
    #      is_correct = (errors < 0.10).astype(int)
              
        # ... ECE, AUC, AUSE ...

   





    #3. For Monocular Depth: determine "correctness" based on a threshold
    #(e.g., relative error < 1.25 delta threshold, or absolute error below a limit)
    absolute_errors = 0.0
    # is_correct = (absolute_errors < 0.1).astype(int)  # <---- Variables to add in later
    is_correct = np.array([1, 1, 0, 1, 0]) # <---- Variables to add in later
    confidence_scores = np.array([0.9, 0.85, 0.3, 0.95, 0.4]) # <---- Variables to add in later



    #4. Calculate metrics using metric functions in /metrics.py
    auc = calculate_rejection_auc(is_correct, confidence_scores)
    ece = compute_ece(is_correct, confidence_scores)

    print(f"--- Results ---")
    print(f"Rejection ROC-AUC: {auc:.4f}")
    print(f"Expected Calibration Error (ECE): {ece:.4f}")


    # NOTE: add random tie breaking



    # Load data


    # Add Per condition rpeporting and bootstrampping







if __name__ == "__main__":
    main()