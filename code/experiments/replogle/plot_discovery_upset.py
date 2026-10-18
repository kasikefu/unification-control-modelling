"""Plot exclusive discovery intersections from the saved Replogle results."""

from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd


METHODS = ("iv", "sv", "gmm")
# Identical order in both panels; each pair belongs to exactly one column.
REGIONS = ((1, 0, 0), (0, 1, 0), (0, 0, 1),
           (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1))
SELECTIONS = ("wald_gmm", "local_gmm", "wilcoxon")
SELECTION_LABELS = ("GMM (non-robust)", "GMM (robust)", "Wilcoxon")
SELECTION_COLORS = ("#0173b2", "#de8f05", "#cc78bc")


def overlap_counts(call_table):
    bits = call_table[list(METHODS)].astype(int).to_numpy()
    return {key: int(np.all(bits == np.array(key), axis=1).sum())
            for key in REGIONS}


def count_tick(value, _position):
    return "0" if value == 0 else f"{value / 1000:g}k"


def draw_upset(fig, slot, counts, title, *, ymax, ylabel=False):
    """Draw seven exclusive intersections and their membership matrix."""
    grid = slot.subgridspec(2, 1, height_ratios=(3.3, 1), hspace=.08)
    ax = fig.add_subplot(grid[0])
    matrix = fig.add_subplot(grid[1], sharex=ax)
    values = np.array([counts[key] for key in REGIONS])
    x = np.arange(len(REGIONS))
    bars = ax.bar(x, values, width=.66, color="#424242", linewidth=0)
    for bar, value in zip(bars, values):
        ax.annotate(f"{value:,}", (bar.get_x() + bar.get_width()/2, value),
                    xytext=(0, 5), textcoords="offset points", ha="left",
                    va="bottom", rotation=60, fontsize=16,
                    rotation_mode="anchor")
    ax.set_ylim(0, ymax)
    ax.set_yticks(np.arange(0, ymax, 20000))
    ax.yaxis.set_major_formatter(FuncFormatter(count_tick))
    ax.tick_params(axis="y", labelsize=16)
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    ax.set_title(title, loc="left", fontsize=19, pad=12)
    if ylabel:
        ax.set_ylabel("Significant pairs", fontsize=17)
    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.spines["left"].set_color(".6")
    ax.grid(axis="y", alpha=.16)
    ax.set_axisbelow(True)

    # Filled dots are included methods; pale dots are excluded methods.
    for row in range(3):
        if row % 2 == 0:
            matrix.axhspan(row-.48, row+.48, color="#f3f3f3", zorder=0)
    for col, region in enumerate(REGIONS):
        active = np.flatnonzero(region)
        matrix.scatter(np.full(3, col), np.arange(3), s=55,
                       color="#d8d8d8", edgecolors="none", zorder=2)
        if len(active) > 1:
            matrix.plot([col, col], [active.min(), active.max()],
                        color="#424242", lw=2, zorder=3)
        matrix.scatter(np.full(len(active), col), active, s=55,
                       color="#424242", edgecolors="none", zorder=4)
    matrix.set(xlim=(-.65, 6.65), ylim=(2.55, -.55))
    matrix.set_yticks(range(3), ("IV", "SV", "GMM"), fontsize=16)
    matrix.tick_params(axis="both", length=0, labelbottom=False)
    matrix.spines[:].set_visible(False)
    ax._intersection_values = values
    matrix._membership = np.array(REGIONS)
    return ax, matrix


def draw_comparison(overlap_by_inference):
    fig = plt.figure(figsize=(11.4, 4.7))
    grid = fig.add_gridspec(1, 2, left=.10, right=.97, bottom=.07,
                           top=.86, wspace=.29)
    ymax = max(max(c.values()) for c in overlap_by_inference.values()) * 1.28
    for i, (mode, counts) in enumerate(overlap_by_inference.items()):
        draw_upset(fig, grid[0, i], counts, f"({chr(97+i)}) {mode}",
                   ymax=ymax, ylabel=(i == 0))
    return fig


def draw_summary(overlap_by_inference, go_overview, go_recurrent):
    fig = plt.figure(figsize=(11.8, 10.2))
    top = fig.add_gridspec(1, 3, left=.10, right=.98, bottom=.62,
                          top=.94, width_ratios=(1.35, 1.35, .68), wspace=.35)
    ymax = max(max(c.values()) for c in overlap_by_inference.values()) * 1.28
    for i, (mode, counts) in enumerate(overlap_by_inference.items()):
        draw_upset(fig, top[0, i], counts, f"({chr(97+i)}) {mode}",
                   ymax=ymax, ylabel=(i == 0))

    ax_count = fig.add_subplot(top[0, 2])
    # Reserve space below this narrow panel for its rotated category labels.
    position = ax_count.get_position()
    ax_count.set_position((position.x0, .65, position.width, position.y1-.65))
    counts = go_overview.set_index("selection").reindex(SELECTIONS)["selected_pairs"].astype(int)
    bars = ax_count.bar(range(3), counts, width=.65, color=SELECTION_COLORS)
    for bar, value in zip(bars, counts):
        ax_count.annotate(f"{value:,}", (bar.get_x()+bar.get_width()/2, value),
                          xytext=(0, 5), textcoords="offset points", fontsize=16,
                          ha="left", va="bottom", rotation=60, rotation_mode="anchor")
    ax_count.set_xticks(range(3), ("Non-robust", "Robust", "Wilcoxon"),
                        fontsize=15, rotation=50, ha="right", rotation_mode="anchor")
    ax_count.set_ylim(0, max(counts)*1.28)
    ax_count.set_yticks((0, 100000, 200000))
    ax_count.yaxis.set_major_formatter(FuncFormatter(count_tick))
    ax_count.tick_params(axis="x", length=0)
    ax_count.tick_params(axis="y", labelsize=16)
    ax_count.set_title("(c)", loc="left", fontsize=19, pad=12)
    ax_count.grid(axis="y", alpha=.16)
    ax_count.set_axisbelow(True)
    ax_count.spines[["top", "right"]].set_visible(False)

    pivot = go_recurrent[go_recurrent.selection.isin(SELECTIONS)].pivot_table(
        index=["go_id", "go_name"], columns="selection", values="n_perturbations", fill_value=0)
    pivot = pivot.reindex(columns=SELECTIONS, fill_value=0)
    pivot["total"] = pivot[list(SELECTIONS)].sum(axis=1)
    pivot = pivot.sort_values(["total", "go_name"], ascending=[False, True]).head(12).iloc[::-1]
    if pivot.empty:
        raise ValueError("No significant GO:BP terms to plot.")
    bottom = fig.add_gridspec(1, 1, left=.43, right=.98, bottom=.075, top=.52)
    ax_go = fig.add_subplot(bottom[0, 0])
    y = np.arange(len(pivot))
    height = .23
    for j, (selection, label, color) in enumerate(zip(SELECTIONS, SELECTION_LABELS, SELECTION_COLORS)):
        ax_go.barh(y + (j-1)*height, pivot[selection], height=height,
                   color=color, label=label)
    ax_go.set_yticks(y, [fill(name, width=38) for name in pivot.index.get_level_values("go_name")],
                    fontsize=16)
    for text in ax_go.get_yticklabels():
        text.set_linespacing(.88)
    ax_go.set_xlim(0, 66)
    ax_go.set_xticks((0, 20, 40, 60))
    ax_go.set_xlabel("Perturbations with GO enrichment", fontsize=17)
    ax_go.tick_params(axis="y", length=0, pad=8)
    ax_go.tick_params(axis="x", labelsize=16)
    ax_go.margins(y=.025)
    ax_go.legend(loc="upper center", bbox_to_anchor=(.43, -.17),
                 ncol=3, frameon=False, fontsize=14)
    ax_go.grid(axis="x", alpha=.16)
    ax_go.set_axisbelow(True)
    ax_go.spines[["top", "right"]].set_visible(False)
    fig.text(.02, .545, "(d)", fontsize=19)
    return fig


def read_discoveries(root):
    frame = pd.read_csv(root / "results/replogle/iv_sv_gmm_results.csv.gz")
    qcols = [f"q_{m}" for m in METHODS] + [f"q_robust_{m}" for m in METHODS]
    complete = frame.loc[frame[qcols].notna().all(axis=1)]
    if complete.duplicated(["perturbation", "y_gene"]).any():
        raise ValueError("Duplicate perturbation-outcome pairs.")
    counts = {}
    for mode, prefix in (("Non-robust", "q_"), ("Robust", "q_robust_")):
        calls = pd.DataFrame({m: complete[f"{prefix}{m}"].le(.05) for m in METHODS})
        counts[mode] = overlap_counts(calls)
        for index, method in enumerate(METHODS):
            assert sum(value for region, value in counts[mode].items() if region[index]) == calls[method].sum()
    return complete, counts


def main():
    root = Path(__file__).resolve().parents[3]
    plt.rcParams.update({"pdf.fonttype": 42, "figure.facecolor": "white",
                         "axes.spines.top": False, "axes.spines.right": False})
    complete, counts = read_discoveries(root)
    out = root / "results/replogle"
    figures = root / "figures"
    summary = draw_summary(counts, pd.read_csv(out / "go_bp_overview.csv"),
                           pd.read_csv(out / "go_bp_recurrent_terms.csv"))
    comparison = draw_comparison(counts)
    for fig, stem in ((summary, "realdata_compact_summary"),
                      (comparison, "realdata_upset_comparison")):
        fig.savefig(figures / f"{stem}.pdf", dpi=180,
                    bbox_inches="tight", pad_inches=.15, facecolor="white")
        plt.close(fig)
    pd.DataFrame([{"inference": mode, "region": "".join(map(str, region)), "count": value}
                  for mode, regions in counts.items() for region, value in regions.items()]).to_csv(
                      out / "upset_intersection_counts.csv", index=False)
    print(f"Plotted {len(complete):,} pairs across {complete.perturbation.nunique()} perturbations.")
    print(counts)


if __name__ == "__main__":
    main()
