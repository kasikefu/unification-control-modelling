"""Monte Carlo experiment for Section 5.1 of the paper."""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ucm.dgp import generate_data
from ucm.estimators import combined, efficient_gmm
from ucm.inference import gmm_variance, standard_error, wald_ci, wald_test
from ucm.projected import projected_scores, sample_projections
from ucm.weighting import oracle_projected_weight


A_VALUES = (1.0, 2.0, 3.0)
N_VALUES = tuple(range(200, 2001, 200))
DEFAULT_SEED = 20260919


def estimate(data):
    x, y = data["x"], data["y"]
    z, w = data["z"], data["w"]
    h_iv, h_sv = sample_projections(x, z, w)
    h = np.column_stack((h_iv, h_sv))
    beta_equal = combined(x, y, h_iv, h_sv)
    beta_iv, iv_weight = efficient_gmm(
        x, y, z, w, h_iv, h_sv, return_weight=True, channel="iv"
    )
    beta_sv, sv_weight = efficient_gmm(
        x, y, z, w, h_iv, h_sv, return_weight=True, channel="sv"
    )
    beta_feasible, feasible_weight = efficient_gmm(
        x, y, z, w, h_iv, h_sv, return_weight=True
    )
    weights = {
        "iv": (h_iv, iv_weight, beta_iv),
        "sv": (h_sv, sv_weight, beta_sv),
        "equal": (h, None, beta_equal),
        "feasible_efficient": (h, feasible_weight, beta_feasible),
        "oracle_efficient": (h, oracle_projected_weight(data["a"]), None),
    }

    results = {}
    for method, (instruments, weight, beta_hat) in weights.items():
        if beta_hat is None:
            beta_hat = combined(x, y, h_iv, h_sv, weight)
        beta_hat = float(np.asarray(beta_hat).item())
        scores = projected_scores(x, y, z, w, beta_hat, h_iv, h_sv)
        if method == "iv":
            scores = scores[:, :1]
        elif method == "sv":
            scores = scores[:, 1:]
        variance = gmm_variance(x, y, instruments, beta_hat, weight, scores)
        se = float(standard_error(variance, len(y)).item())
        ci_low, ci_high = wald_ci(beta_hat, se)
        _, p_value = wald_test(beta_hat, 0.0, se)
        results[method] = (beta_hat, se, float(ci_low), float(ci_high), float(p_value))
    return results


def run(replications=1000, seed=DEFAULT_SEED):
    rng = np.random.default_rng(seed)
    rows = []
    scenarios = (("estimation", 1.0), ("size", 0.0), ("power_1", None), ("power_2", None))

    for a in A_VALUES:
        for n in N_VALUES:
            for scenario, fixed_beta in scenarios:
                beta = fixed_beta if fixed_beta is not None else int(scenario[-1]) / np.sqrt(n)
                for replication in range(replications):
                    draw_seed = int(rng.integers(0, 2**63))
                    data = generate_data(n=n, a=a, c=1.0, beta=beta, seed=draw_seed)
                    data["a"] = a
                    for method, (beta_hat, se, ci_low, ci_high, p_value) in estimate(data).items():
                        rows.append({
                            "seed": seed, "draw_seed": draw_seed,
                            "scenario": scenario, "a": a, "n": n,
                            "replication": replication, "method": method,
                            "beta": beta, "beta_hat": beta_hat, "se": se,
                            "ci_low": ci_low, "ci_high": ci_high,
                            "ci_length": ci_high - ci_low,
                            "covered": ci_low <= beta <= ci_high,
                            "p_value_h0_beta_0": p_value,
                            "reject_h0_beta_0": p_value < 0.05,
                        })
                print(f"Finished a={a:.4f}, n={n}, scenario={scenario}", flush=True)

    raw = pd.DataFrame(rows)
    grouped = raw.groupby(["scenario", "a", "n", "method"], sort=False)
    summary = grouped.agg(
        beta=("beta", "first"),
        replications=("beta_hat", "size"),
        mean_beta_hat=("beta_hat", "mean"),
        empirical_variance=("beta_hat", lambda values: values.var(ddof=0)),
        mean_se=("se", "mean"),
        mean_ci_length=("ci_length", "mean"),
        coverage=("covered", "mean"),
        rejection_h0_beta_0=("reject_h0_beta_0", "mean"),
    ).reset_index()
    summary["bias"] = summary["mean_beta_hat"] - summary["beta"]
    summary["squared_bias"] = summary["bias"] ** 2
    raw["squared_error"] = (raw["beta_hat"] - raw["beta"]) ** 2
    mse = raw.groupby(["scenario", "a", "n", "method"], sort=False)["squared_error"].mean()
    summary["mse"] = summary.set_index(["scenario", "a", "n", "method"]).index.map(mse)
    summary["scaled_mse"] = summary["n"] * summary["mse"]
    summary["scaled_variance"] = summary["n"] * summary["empirical_variance"]
    equal_variance = (summary["a"] ** 6 / 4 + 1) / (summary["a"] ** 4 / 4 + 1) ** 2
    summary["theoretical_asymptotic_variance"] = np.select(
        [summary["method"].eq("iv"), summary["method"].eq("sv"), summary["method"].eq("equal")],
        [4 / summary["a"] ** 2, 1.0, equal_variance],
        default=4 / (summary["a"] ** 2 + 4),
    )
    summary["theoretical_finite_n_variance"] = (
        summary["theoretical_asymptotic_variance"] / summary["n"]
    )
    summary["theoretical_ci_length"] = (
        2 * norm.ppf(0.975) * np.sqrt(summary["theoretical_finite_n_variance"])
    )

    output_dir = Path(__file__).resolve().parents[3] / "result"
    output_dir.mkdir(exist_ok=True)
    raw_path = output_dir / "efficiency_replications.csv"
    summary_path = output_dir / "efficiency_summary.csv"
    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"Saved {raw_path}")
    print(f"Saved {summary_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    if args.replications < 2:
        parser.error("--replications must be at least 2")
    run(args.replications, args.seed)


if __name__ == "__main__":
    main()
