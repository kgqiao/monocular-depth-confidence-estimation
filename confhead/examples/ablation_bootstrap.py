import pandas as pd
from evals.bootstrap import paired_scene_bootstrap_ci

HEAD = "run-v1-radar"
RUNS = ["run-v1-radar-repeat", "run-v0-no-radar", "m1-no-features",
        "m1-features-only", "m1-no-temporal", "m1-seed1"]

def load(run):
    df = pd.read_csv(f"results/{run}/{run}.csv")
    return df.sort_values("key").reset_index(drop=True)

head = load(HEAD)
print("run | cond | run - head AUSE [95% CI] | clear of 0?")
for run in RUNS:
    df = load(run)
    assert (df["key"] == head["key"]).all()
    assert (df["scene"] == head["scene"]).all()
    for cond in ["all", "day", "rain", "night"]:
        m = (head["condition"] == cond) if cond != "all" else head["condition"].notna()
        mean, lo, hi = paired_scene_bootstrap_ci(
            df.loc[m, "ause"].to_numpy(), head.loc[m, "ause"].to_numpy(),
            head.loc[m, "scene"].to_numpy())
        clear = "YES" if (lo > 0 or hi < 0) else "no (tie)"
        print(f"{run:22s} {cond:6s} {mean:+.5f} [{lo:+.5f}, {hi:+.5f}] {clear}")
