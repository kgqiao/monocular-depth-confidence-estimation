"""
Build test metrics for the following cases:

pytest - 
- Passing true_error as uncertainty score returns AUSE = 0.0



"""

import numpy as np
import pytest
from evals.metrics import compute_sparsification_curves, calculate_ause_and_aurg

@pytest.fixture # Hands same setup to every test tha tnames it as an arugment. No test can change another test's data
def true_errors():
    """
    10,000 fake per cell errors, same every run (fixed seed).
    """
    rng = np.random.default_rng(42)
    return rng.exponential(scale = 1.0, size = 10_000)


def score(true_errors, uncertainty_scores):
    '''
    Run both functions and return everything a test might check."""
    
    '''
    # Run both functions
    fractions, sparsification, oracle, random_curve = compute_sparsification_curves(true_errors, uncertainty_scores)
    ause, aurg = calculate_ause_and_aurg(fractions, sparsification, oracle, random_curve)

    return sparsification, oracle, random_curve, ause, aurg

def test_perfect_scores_gives_zero_ause(true_errors): # Perfect Uncertainty, AUSE = 0
    '''
    Where AUSE (how far the head is from flagging perfectly) is 0, thus a perfect head is the oracle. 
    Sparsification curve is identical to oracle curve. 

    Where score = true error
    '''
    #Aka errors of predicted and real are identical
    sparsification, oracle, _, ause, _ = score(true_errors, true_errors.copy())
    assert np.allclose(sparsification, oracle) # Return true if two arrays are element-wise equal
    assert np.isclose(ause, 0.0)

def test_random_score_gives_aurg_near_zero(true_errors): # AURG == 0 
    rng = np.random.default_rng(7)
    random_scores = rng.random(len(true_errors)) # Test in which predicted scores are nothing like the real errors
    _, _, _, _, aurg = score(true_errors, random_scores)
    assert abs(aurg) < 0.05


def test_reversed_score_gives_negative_aurg(true_errors):
    '''
    
    In which model removes its most accurate instead of least accuract predictions first.
    Causes remaining error to skyrocket, producing negative AURG
    '''
    # Set uncertainties = -errors
    _, _, _, ause, aurg = score(true_errors, -true_errors)
    assert aurg < 0
    assert ause > 0


def test_constant_error_gives_flat_curves():













def test_sparsification_metrics_synthetic():
    np.random.seed(42)
    n_samples = 10000
    
    # Create fake true errors
    true_errors = np.random.exponential(scale=1.0, size=n_samples)
    
    # --- TEST 1: A perfect score gives AUSE = 0 ---



    # --- TEST 2: A random score gives AURG near 0 ---



    # --- TEST 3: A reversed score gives negative AURG ---



    # --- TEST 4: Constant error gives flat curves ---