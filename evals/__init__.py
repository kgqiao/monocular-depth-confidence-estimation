"""
Signals to Python that the evals folder is a package.

Expose metrics directly to have metrics directly importable from top level of the evals package
"""
from evals.metrics import compute_ece, calculate_rejection_auc

__all__ = ["compute_ece", "calculate_rejection_auc"]
