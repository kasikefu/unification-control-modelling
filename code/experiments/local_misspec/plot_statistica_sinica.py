"""Create one-row, three-panel local-misspecification figures in journal style."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import ndtr


METHODS = ("ols_unadjusted", "iv", "sv", "equal", "local_risk")
METHOD_LABELS = {
    "ols_unadjusted": "Unadjusted OLS",
    "iv": "IV only",
    "sv": "SV only",
    "equal": "Equal GMM",
    "local_risk": "Local-risk GMM",
}
REGIME_LABELS = {
    "iv": "IV channel may deviate",
    "sv": "SV channel may deviate",
    "both": "Both channels may deviate",
}
FILE_LABELS = {
    "iv": "iv_deviates",
    "sv": "sv_deviates",
    "both": "both_deviate",
}
PALETTE = sns.color_palette("colorblind", 8)
COLORS = dict(zip(METHODS, (PALETTE[i] for i in (7, 0, 1, 2, 4))))
STYLES = {
    "ols_unadjusted": dict(color=COLORS["ols_unadjusted"], linestyle="-", marker="X"),
    "iv": dict(color=COLORS["iv"], linestyle="-", marker="o"),
    "sv": dict(color=COLORS["sv"], linestyle="-", marker="s"),
    "equal": dict(color=COLORS["equal"], linestyle="-", marker="^"),
    "local_risk": dict(color=COLORS["local_risk"], linestyle="-", marker="D"),
}


def inference_summary(result_dir):
    raw_columns = [
        "regime", "kappa", "n", "tau_z", "tau_w", "method", "beta_hat", "sigma"
    ]
    raw = pd.read_csv(
        result_dir / "local_misspec_replications.csv", usecols=raw_columns
    )
    raw = raw[(raw["kappa"] == 1.0) & raw["regime"].isin(REGIME_LABELS)].copy()

    critical_value = 1.959963984540054
    raw["null_z"] = np.sqrt(raw["n"]) * (raw["beta_hat"] - 1.0) / raw["sigma"]
    raw["local_z_d2"] = raw["null_z"] + 2.0 / raw["sigma"]
    raw["reject_null"] = raw["null_z"].abs() > critical_value
    raw["reject_local_d2"] = raw["local_z_d2"].abs() > critical_value

    boundary = raw.groupby(
        ["regime", "kappa", "n", "tau_z", "tau_w", "method"], as_index=False
    ).agg(
        type_i_error=("reject_null", "mean"),
        local_power_d2=("reject_local_d2", "mean"),
    )
    inference = boundary.groupby(
        ["regime", "kappa", "n", "method"], as_index=False
    ).agg(
        worst_type_i_error=("type_i_error", "max"),
        minimum_local_power_d2=("local_power_d2", "min"),
    )
    inference.to_csv(result_dir / "local_misspec_inference_summary.csv", index=False)
    return inference


def _style_axis(ax):
    ax.grid(False)
    ax.tick_params(direction="out", length=3.5, width=0.7, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(0.7)


def make_figures(result_dir):
    """Show MSE, robust Type I error, and minimum local power over every n."""
    result_dir = Path(result_dir)
    worst = pd.read_csv(result_dir / "local_misspec_worst_case.csv")
    raw = pd.read_csv(
        result_dir / "local_misspec_replications.csv",
        usecols=["regime", "kappa", "n", "tau_z", "tau_w", "method",
                 "beta_hat", "sigma", "bias_bound"],
    )
    raw = raw[raw["kappa"].eq(1.0) & raw["regime"].isin(REGIME_LABELS)].copy()
    t = np.where(raw["method"].eq("ols_unadjusted"), 0.0,
                 (raw["bias_bound"] / raw["sigma"]).to_numpy())
    lower = np.zeros_like(t)
    upper = t + 1.959963984540054
    for _ in range(45):
        midpoint = (lower + upper) / 2
        below = ndtr(midpoint - t) - ndtr(-midpoint - t) < 0.95
        lower = np.where(below, midpoint, lower)
        upper = np.where(below, upper, midpoint)
    critical = (lower + upper) / 2
    null_z = np.sqrt(raw["n"].to_numpy()) * (
        raw["beta_hat"].to_numpy() - 1.0
    ) / raw["sigma"].to_numpy()
    raw["local_power"] = np.abs(null_z + 2.0 / raw["sigma"].to_numpy()) > critical
    boundary = raw.groupby(
        ["regime", "n", "tau_z", "tau_w", "method"], as_index=False
    )["local_power"].mean()
    min_power = boundary.groupby(
        ["regime", "n", "method"], as_index=False
    )["local_power"].min()
    del raw

    for regime in REGIME_LABELS:
        panel = worst[worst["kappa"].eq(1.0) & worst["regime"].eq(regime)].copy()
        panel["type_i_error"] = np.where(
            panel["method"].eq("ols_unadjusted"),
            1 - panel["min_wald_coverage"],
            1 - panel["min_robust_coverage"],
        )
        power_panel = min_power[min_power["regime"].eq(regime)]
        n_values = sorted(panel["n"].unique())
        fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.8))
        for method in METHODS:
            frame = panel[panel["method"].eq(method)].sort_values("n")
            style = {**STYLES[method], "linewidth": 1.2, "markersize": 4.3,
                     "markerfacecolor": "white", "markeredgewidth": 0.9,
                     "label": METHOD_LABELS[method]}
            axes[0].plot(frame["n"], frame["worst_mse"], **style)
        x = np.arange(len(n_values))
        bar_width = 0.14
        type_i_values = panel.pivot(index="n", columns="method", values="type_i_error").reindex(
            index=n_values, columns=list(METHODS)
        )
        power_values = power_panel.pivot(index="n", columns="method", values="local_power").reindex(
            index=n_values, columns=list(METHODS)
        )
        for axis, values in ((axes[1], type_i_values), (axes[2], power_values)):
            for index, method in enumerate(METHODS):
                axis.bar(x + (index - 2) * bar_width, values[method], bar_width,
                         color=STYLES[method]["color"], edgecolor="black", linewidth=0.35)
        axes[0].set(title="(a) Worst-case mean squared error", ylabel="MSE", yscale="log")
        axes[1].set(title="(b) Type I error", ylabel="Worst-case rejection probability",
                    ylim=(0, 1.04))
        axes[2].set(title="(c) Minimum local power: beta = 1 + 2/sqrt(n)",
                    ylabel="Minimum rejection probability", ylim=(0, 1.04))
        axes[1].axhline(0.05, color="0.45", linestyle=":", linewidth=0.8)
        for ax in axes:
            _style_axis(ax)
            ax.set_xlabel("Sample size n")
            ax.grid(axis="y", color="0.88", linewidth=0.5)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_xticks(n_values)
        for ax in axes[1:]:
            ax.set_xticks(x, n_values)
        line_handles, line_labels = axes[0].get_legend_handles_labels()
        axes[0].legend(handles=line_handles, labels=line_labels, title="Method",
                       loc="upper center", bbox_to_anchor=(0.5, -0.11),
                       ncol=2, frameon=False, columnspacing=0.8, handlelength=1.7)
        method_handles = [
            Patch(facecolor=STYLES[method]["color"], edgecolor="black", linewidth=0.4,
                  label=METHOD_LABELS[method]) for method in METHODS
        ]
        for ax in axes[1:]:
            ax.legend(handles=method_handles, title="Method", loc="upper center",
                      bbox_to_anchor=(0.5, -0.11), ncol=2, frameon=False,
                      columnspacing=0.8, handlelength=1.3)
        fig.suptitle(f"{REGIME_LABELS[regime]} (κ = 1)", y=0.99)
        fig.text(0.5, 0.03, "OLS: Wald inference; IV/SV/GMM: local-robust inference.",
                 ha="center", fontsize=8.5)
        fig.subplots_adjust(left=0.055, right=0.99, bottom=0.39,
                            top=0.85, wspace=0.26)
        stem = result_dir / f"local_misspec_{FILE_LABELS[regime]}_combined"
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {stem}.pdf")
    return min_power


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path(__file__).resolve().parents[3] / "results" / "local_misspec",
    )
    args = parser.parse_args()
    make_figures(args.result_dir)


if __name__ == "__main__":
    main()

