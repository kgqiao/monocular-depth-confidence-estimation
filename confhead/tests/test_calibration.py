"""
Build test metrics for the following cases:

1. AUROC (Area under ROC)
- Confidence high on every correct pixel, low on every wrong one -> 1.0
- The same scores reversed -> 0.0
- Random scores -> about 0.5
- All pixels correct (one class) -> NaN instead of a crash


2. ECE
- Scores that match reality -> about 0
- Hand example with two bins -> 0.25
- A score of exactly 1.0 is counted (last bin includes 1.0)


"""


# Test AUROC





# Test compute_ece() function