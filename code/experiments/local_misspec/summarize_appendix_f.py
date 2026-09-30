"""Choose a table or line plot according to stability across sample sizes."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STABILITY_THRESHOLD = 0.10
METHODS = ("iv", "sv", "equal", "local_risk")
METHOD_LABELS = {
    "iv": "IV only",
    "sv": "SV only",
    "equal": "Equal GMM",
    "local_risk": "Appendix-F optimal",
}
REGIME_LABELS = {
    "none": "Exact",
    "iv": "IV deviates",
    "sv": "SV deviates",
    "both": "Both deviate",
}


def trend_change(frame):
    """Fitted end-to-end change in scaled MSE, relative to its mean."""
    x = frame["n"].to_numpy(dtype=float)
    y = frame["worst_scaled_mse"].to_numpy(dtype=float)
    slope = np.polyfit(x, y, 1)[0]
    return abs(slope * (x.max() - x.min())) / abs(y.mean())


def main():
    result_dir = Path(__file__).resolve().parents[3] / "results" / "local_misspec"
    results = pd.read_csv(result_dir / "local_misspec_worst_case.csv")
    grouped = results.groupby(["regime", "kappa", "method"], sort=False)

    stability = grouped.apply(
        lambda frame: pd.Series({
            "relative_fitted_change": trend_change(frame),
            "mean_scaled_mse": frame["worst_scaled_mse"].mean(),
            "sd_scaled_mse_across_n": frame["worst_scaled_mse"].std(ddof=1),
            "theoretical_risk": frame["theoretical_worst_risk"].iloc[0],
            "mean_iv_share": frame["mean_iv_share"].mean(),
            "min_robust_coverage": frame["min_robust_coverage"].mean(),
        }),
        include_groups=False,
    ).reset_index()
    stability.to_csv(result_dir / "appendix_f_n_stability.csv", index=False)

    if stability["relative_fitted_change"].max() >= STABILITY_THRESHOLD:
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
        for ax, method in zip(axes.flat, METHODS):
            for (regime, kappa), frame in results[results["method"].eq(method)].groupby(
                ["regime", "kappa"], sort=False
            ):
                label = f"{REGIME_LABELS[regime]}, kappa={kappa:g}"
                ax.plot(frame["n"], frame["worst_scaled_mse"], marker="o", label=label)
            ax.set_title(METHOD_LABELS[method])
            ax.set_ylabel("Worst-case scaled MSE")
            ax.grid(alpha=0.25)
        for ax in axes[-1]:
            ax.set_xlabel("n")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        stem = result_dir / "appendix_f_risk_by_n"
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
        print(f"Saved line plot to {stem}.pdf/.png")
        return

    rows = []
    order = (
        ("none", 0.0),
        ("iv", 0.5), ("sv", 0.5), ("both", 0.5),
        ("iv", 1.0), ("sv", 1.0), ("both", 1.0),
    )
    for regime, kappa in order:
        row = {"Design": REGIME_LABELS[regime], "kappa": kappa}
        subset = stability[
            stability["regime"].eq(regime) & stability["kappa"].eq(kappa)
        ].set_index("method")
        for method in METHODS:
            row[f"{method}_simulated"] = subset.loc[method, "mean_scaled_mse"]
            row[f"{method}_theory"] = subset.loc[method, "theoretical_risk"]
        rows.append(row)
    table_data = pd.DataFrame(rows)
    table_data.to_csv(result_dir / "appendix_f_risk_table.csv", index=False)

    display = table_data[["Design", "kappa"]].copy()
    display["kappa"] = display["kappa"].map(lambda value: f"{value:g}")
    for method in METHODS:
        display[METHOD_LABELS[method]] = [
            f"{simulated:.3f} ({theory:.3f})"
            for simulated, theory in zip(
                table_data[f"{method}_simulated"], table_data[f"{method}_theory"]
            )
        ]

    fig, ax = plt.subplots(figsize=(11.5, 3.6))
    ax.axis("off")
    table = ax.table(
        cellText=display.to_numpy(),
        colLabels=display.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    for column in range(len(display.columns)):
        table[(0, column)].set_facecolor("#D9EAF7")
        table[(0, column)].set_text_props(weight="bold")
    ax.set_title(
        "Worst-case scaled MSE averaged over n; theoretical risk in parentheses",
        pad=14,
    )
    fig.tight_layout()
    stem = result_dir / "appendix_f_risk_table"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(
        "Results are stable across n "
        f"(max fitted relative change={stability['relative_fitted_change'].max():.3f}); "
        f"saved table to {stem}.pdf/.png"
    )


if __name__ == "__main__":
    main()

