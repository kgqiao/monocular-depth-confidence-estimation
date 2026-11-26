"""
Build day/rain/night results table from all model CSVs.
Auto-discovers CSVs in confhead/results/*/
Add a new model: just drop its CSV in a new subfolder and re-run.

Usage: python3 confhead/examples/build_results_table.py
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add confhead/src to path so we can import evals.bootstrap
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from evals.bootstrap import scene_bootstrap_ci

RESULTS_DIR = Path("confhead/results")
N_BOOTSTRAP = 2000
CI_LEVEL = 0.95
METRICS = ["auroc", "ece", "ause", "aurg", "brier"]
CONDITIONS = ["day", "rain", "night"]


def discover_models():
    """Find all CSVs under results/*/  — one subfolder = one model."""
    models = {}
    skip = {"lidar_pixels_per_condition.csv", "results_table.csv"}
    for csv_path in sorted(RESULTS_DIR.glob("*/*.csv")):
        if csv_path.name in skip:
            continue
        model_name = csv_path.parent.name
        df = pd.read_csv(csv_path)
        models[model_name] = df
        print(f"  Loaded '{model_name}': {len(df):,} rows")
    return models


def condition_stats(df, condition):
    """Mean ± scene-bootstrap CI for each metric, filtered to one condition."""
    sub = df[df["condition"] == condition].copy()
    if sub.empty:
        return None
    out = {
        "n_images": len(sub),
        "n_labeled": int(sub["n_labeled"].sum()),
    }
    for m in METRICS:
        if m not in sub.columns:
            continue
        # scene_bootstrap_ci handles NaN dropping internally
        try:
            mean, lo, hi = scene_bootstrap_ci(sub[m], sub["scene"])
        except Exception:
            continue
        out[m] = mean
        out[f"{m}_lo"] = lo
        out[f"{m}_hi"] = hi
    return out


def fmt_cell(r, metric):
    """Format mean [lo, hi] for one metric."""
    if metric not in r:
        return "—"
    lo_key = f"{metric}_lo"
    if lo_key in r:
        return f"{r[metric]:.4f} [{r[lo_key]:.4f}, {r[f'{metric}_hi']:.4f}]"
    return f"{r[metric]:.4f}"


def print_table(results, models):
    W = 100
    for metric in METRICS:
        print(f"\n{'='*W}")
        print(f"  {metric.upper()}  (↑ = higher better for auroc/aurg; ↓ = lower better for ece/ause/brier)")
        print(f"  Scene-bootstrap {int(CI_LEVEL*100)}% CI  (n={N_BOOTSTRAP})")
        print(f"{'='*W}")
        print(f"{'Model':<25} {'Day':>30} {'Rain':>30} {'Night':>30}")
        print(f"{'-'*W}")
        for model in models:
            row_parts = [f"{model:<25}"]
            for cond in CONDITIONS:
                sub = results[
                    (results["model"] == model) & (results["condition"] == cond)
                ]
                cell = fmt_cell(sub.iloc[0].to_dict(), metric) if not sub.empty else "—"
                row_parts.append(f"{cell:>30}")
            print("".join(row_parts))


def save_outputs(results):
    # Full CSV with CIs
    csv_out = RESULTS_DIR / "results_table.csv"
    results.to_csv(csv_out, index=False)
    print(f"\nSaved: {csv_out}")

    # Markdown table (easy to paste into paper/report)
    md_lines = ["# Model Results by Condition\n",
                f"Scene-bootstrap {int(CI_LEVEL*100)}% CI, n={N_BOOTSTRAP}\n",
                f"Format: `mean [lo, hi]`\n"]
    for metric in METRICS:
        md_lines += [f"\n## {metric.upper()}\n",
                     "| Model | Day | Rain | Night |",
                     "|-------|-----|------|-------|"]
        for model in results["model"].unique():
            cells = [f"| **{model}**"]
            for cond in CONDITIONS:
                sub = results[
                    (results["model"] == model) & (results["condition"] == cond)
                ]
                cells.append(fmt_cell(sub.iloc[0].to_dict(), metric) if not sub.empty else "—")
            md_lines.append(" | ".join(cells) + " |")

    md_out = RESULTS_DIR / "results_table.md"
    md_out.write_text("\n".join(md_lines))
    print(f"Saved: {md_out}")


def main():
    print("Discovering model CSVs...\n")
    models = discover_models()

    if not models:
        print("No CSVs found in confhead/results/*/")
        return

    print(f"\n{len(models)} model(s) found: {list(models.keys())}\n")
    print(f"Computing scene-bootstrap CIs (n={N_BOOTSTRAP})...")

    rows = []
    for model_name, df in models.items():
        for cond in CONDITIONS:
            stats = condition_stats(df, cond)
            if stats:
                rows.append({"model": model_name, "condition": cond, **stats})

    results = pd.DataFrame(rows)
    print_table(results, models)
    save_outputs(results)


if __name__ == "__main__":
    main()