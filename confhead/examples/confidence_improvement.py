"""
Task 3: Does confidence improve depth? The one number.

AURG (Area Under Rejection Gain) measures how much depth error drops
when using model confidence to reject low-confidence predictions vs
random rejection. AURG = 0 means confidence is useless; AURG > 0 means
keeping high-confidence predictions is better than keeping random ones.

Compares flip_baseline_v2 vs best trained head (m3-tau02).
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from evals.bootstrap import scene_bootstrap_ci

RESULTS_DIR = Path("confhead/results")
CONDITIONS = ["day", "rain", "night"]
COMPARE = ["flip_baseline_v2", "m3-tau02"]

def load_models():
    skip = {"lidar_pixels_per_condition.csv", "results_table.csv"}
    models = {}
    for csv_path in sorted(RESULTS_DIR.glob("*/*.csv")):
        if csv_path.name in skip:
            continue
        models[csv_path.parent.name] = pd.read_csv(csv_path)
    return models

def main():
    models = load_models()

    print("Task 3: Confidence Improves Depth\n")
    print("AURG = how much error drops when filtering by confidence vs random")
    print("Baseline error rate = wrong_cells / total_cells (no filtering)\n")

    rows = []
    for name in COMPARE:
        df = models.get(name)
        if df is None:
            print(f"  WARNING: {name} not found")
            continue
        for cond in CONDITIONS:
            sub = df[df["condition"] == cond]
            if sub.empty:
                continue
            baseline_err = sub["n_wrong"].sum() / sub["n_labeled"].sum()
            mean_aurg, lo, hi = scene_bootstrap_ci(sub["aurg"], sub["scene"])
            rows.append({
                "model": name, "condition": cond,
                "baseline_error": baseline_err,
                "aurg": mean_aurg, "aurg_lo": lo, "aurg_hi": hi,
            })

    results = pd.DataFrame(rows)

    for cond in CONDITIONS:
        sub = results[results["condition"] == cond]
        print(f"=== {cond.upper()} ===")
        print(f"{'Model':<25} {'Baseline Err':>13} {'AURG (↑ better)':>22}")
        print("-" * 62)
        for _, r in sub.iterrows():
            aurg_str = f"{r['aurg']:.4f} [{r['aurg_lo']:.4f}, {r['aurg_hi']:.4f}]"
            print(f"{r['model']:<25} {r['baseline_error']:>13.4f} {aurg_str:>22}")
        print()

    # The one number: improvement % of m3-tau02 over flip baseline
    print("=== Key Finding ===")
    for cond in CONDITIONS:
        flip = results[(results["model"]=="flip_baseline_v2") & (results["condition"]==cond)]
        best = results[(results["model"]=="m3-tau02") & (results["condition"]==cond)]
        if flip.empty or best.empty:
            continue
        gain = (best["aurg"].values[0] - flip["aurg"].values[0])
        pct  = gain / flip["aurg"].values[0] * 100
        print(f"  {cond}: m3-tau02 AURG = {best['aurg'].values[0]:.4f} vs flip {flip['aurg'].values[0]:.4f}  → +{pct:.0f}% better confidence utility")

    out = RESULTS_DIR / "task3_confidence_improvement.csv"
    results.to_csv(out, index=False)
    print(f"\nSaved: {out}")

if __name__ == "__main__":
    main()