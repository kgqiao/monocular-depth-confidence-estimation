"""Draw the train and dev loss of every confidence-head run for the midterm report.

The data is the history.json of each run folder on Hugging Face, the same records
the Trackio Space (nathancbao/confhead-charts) draws. Copies are in histories/.

To refresh the copies and redraw (from Docs/Midterm-Report):

    uv run --with matplotlib figures-src/plot_run_losses.py --download

Output: latex/figures/run-losses.pdf
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
HISTORIES = HERE / "histories"
OUT = HERE.parent / "latex" / "figures" / "run-losses.pdf"

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MAGENTA, GREEN, VIOLET, RED = "#e87ba4", "#008300", "#6250d6", "#e34948"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

# (run, repository, legend label, color, line style). The color names the input
# set; the three runs that read all inputs share one color and differ by style.
RUNS = [
    ("run1", "nadayusc/confhead", "all inputs (run1)", BLUE, "-"),
    ("run-v1-radar", "nathancbao/confhead", "all inputs, repeat (run-v1-radar)", BLUE, "--"),
    ("m1-seed1", "nathancbao/confhead", "all inputs, seed 1 (other dev scenes)", BLUE, ":"),
    ("run-v0-no-radar", "nathancbao/confhead", "no radar", ORANGE, "-"),
    ("m1-no-depth", "nathancbao/confhead", "no depth", AQUA, "-"),
    ("m1-no-temporal", "nathancbao/confhead", "no temporal spread", YELLOW, "-"),
    ("m1-features-only", "nathancbao/confhead", "features only", MAGENTA, "-"),
    ("m1-no-features", "nathancbao/confhead", "no features", GREEN, "-"),
    ("m1-depth-only", "nathancbao/confhead", "depth only", VIOLET, "-"),
    ("m1-temporal-only", "nathancbao/confhead", "temporal spread only", RED, "-"),
]

# Runs drawn in the zoomed panel, with the short label written at the line end.
ZOOM = {
    "run1": None,
    "run-v1-radar": None,
    "run-v0-no-radar": "all inputs (x2), no radar",
    "m1-no-depth": "no depth",
    "m1-no-temporal": "no temporal",
    "m1-features-only": "features only",
}


def download() -> None:
    HISTORIES.mkdir(exist_ok=True)
    for run, repo, *_ in RUNS:
        url = f"https://huggingface.co/{repo}/resolve/main/{run}/history.json"
        with urllib.request.urlopen(url) as response:
            (HISTORIES / f"{run}.json").write_bytes(response.read())


def load(run: str) -> list[dict]:
    return json.loads((HISTORIES / f"{run}.json").read_text())


def style(ax, title: str) -> None:
    ax.set_title(title, fontsize=8.5, color=INK, loc="left", pad=4)
    ax.set_xlabel("Epoch", fontsize=8, color=MUTED)
    ax.set_xticks(range(1, 9))
    ax.tick_params(labelsize=7.5, colors=MUTED, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="fetch history.json of every run first")
    if parser.parse_args().download:
        download()

    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                         "pdf.fonttype": 42})
    fig, (ax_train, ax_dev, ax_zoom) = plt.subplots(
        1, 3, figsize=(6.3, 2.75), gridspec_kw={"width_ratios": [1, 1, 1.05]})

    ends = []
    for run, _, _, color, dash in RUNS:
        history = load(run)
        epochs = [row["epoch"] for row in history]
        kwargs = dict(color=color, linestyle=dash, linewidth=1.3, solid_capstyle="round")
        ax_train.plot(epochs, [row["train_loss"] for row in history], **kwargs)
        ax_dev.plot(epochs, [row["dev_loss"] for row in history], **kwargs)
        if run in ZOOM:
            dev = [row["dev_loss"] for row in history]
            ax_zoom.plot(epochs, dev, **kwargs)
            if ZOOM[run]:
                ends.append((dev[-1], ZOOM[run]))

    style(ax_train, "(a) Train loss")
    style(ax_dev, "(b) Dev loss")
    style(ax_zoom, "(c) Dev loss, runs with decoder features")
    ax_train.set_ylabel("Masked BCE", fontsize=8, color=MUTED)
    for ax in (ax_train, ax_dev):
        ax.set_ylim(0.54, 0.72)
    ax_zoom.set_ylim(0.554, 0.596)
    ax_zoom.set_xlim(1, 11.6)

    # Direct labels at the line ends of the zoomed panel, pushed apart to 0.0034.
    gap, y_last = 0.0034, None
    for value, text in sorted(ends):
        y = value if y_last is None else max(value, y_last + gap)
        ax_zoom.annotate(f"{text}  {value:.4f}", xy=(8, value), xytext=(8.25, y), fontsize=6.3,
                         color=INK, va="center", arrowprops=dict(arrowstyle="-", color=GRID, linewidth=0.5))
        y_last = y

    handles = [Line2D([0], [0], color=color, linestyle=dash, linewidth=1.5) for _, _, _, color, dash in RUNS]
    fig.legend(handles, [label for _, _, label, _, _ in RUNS], loc="lower center", ncol=4, fontsize=7,
               frameon=False, handlelength=2.4, columnspacing=1.4, labelcolor=INK)
    fig.tight_layout(rect=(0, 0.17, 1, 1), w_pad=1.2)
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
