"""
Build test metrics for the following cases:

pytest - 
- Passing true_error as uncertainty score returns AUSE = 0.0



"""

import numpy as np
import pytest
from evals.metrics import compute_sparsification_curves, calculate_ause_and_aurg



def test_sparsification_metrics_synthetic():
    np.random.seed(42)
    n_samples = 10000
    
    # Create fake true errors
    true_errors = np.random.exponential(scale=1.0, size=n_samples)
    
    # --- TEST 1: A perfect score gives AUSE = 0 ---



    # --- TEST 2: A random score gives AURG near 0 ---



    # --- TEST 3: A reversed score gives negative AURG ---



    # --- TEST 4: Constant error gives flat curves ---