"""Monte Carlo experiment for Section 5.1 of the paper."""

import argparse
import csv
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
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
DEFAULT_N_JOBS = 8
RAW_FIELDS = (
    "seed", "draw_seed", "scenario", "a", "n", "replication", "method",
    "beta", "beta_hat", "se", "ci_low", "ci_high", "ci_length", "covered",
    "p_value_h0_beta_0", "reject_h0_beta_0",
)


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


def run_configuration(index, a, n, scenario, fixed_beta, draw_seeds, seed, part_path):
    """Run one design configuration and write its replications in order."""
    beta = fixed_beta if fixed_beta is not None else int(scenario[-1]) / np.sqrt(n)
    with part_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RAW_FIELDS)
        for replication, draw_seed in enumerate(draw_seeds):
            draw_seed = int(draw_seed)
            data = generate_data(n=n, a=a, c=1.0, beta=beta, seed=draw_seed)
            data["a"] = a
            for method, (beta_hat, se, ci_low, ci_high, p_value) in estimate(data).items():
                writer.writerow({
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
    return index, a, n, scenario


def run(replications=1000, seed=DEFAULT_SEED, n_jobs=DEFAULT_N_JOBS):
    rng = np.random.default_rng(seed)
    scenarios = (("estimation", 1.0), ("size", 0.0), ("power_1", None), ("power_2", None))
    configurations = []
    for a in A_VALUES:
        for n in N_VALUES:
            for scenario, fixed_beta in scenarios:
                draw_seeds = rng.integers(0, 2**63, size=replications, dtype=np.int64)
                configurations.append((a, n, scenario, fixed_beta, draw_seeds))

    output_dir = Path(__file__).resolve().parents[3] / "results" / "efficiency"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "efficiency_replications.csv"
    summary_path = output_dir / "efficiency_summary.csv"

    with TemporaryDirectory(prefix="efficiency_parts_", dir=output_dir) as temp_dir:
        temp_dir = Path(temp_dir)
        tasks = [
            delayed(run_configuration)(
                index, a, n, scenario, fixed_beta, draw_seeds, seed,
                temp_dir / f"part_{index:04d}.csv",
            )
            for index, (a, n, scenario, fixed_beta, draw_seeds)
            in enumerate(configurations)
        ]
        with parallel_config(backend="loky", inner_max_num_threads=1):
            completed = Parallel(n_jobs=n_jobs, verbose=10)(tasks)

        with raw_path.open("w", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=RAW_FIELDS)
            writer.writeheader()
            for index, a, n, scenario in sorted(completed):
                with (temp_dir / f"part_{index:04d}.csv").open() as part_file:
                    shutil.copyfileobj(part_file, output_file)
                print(f"Finished a={a:.4f}, n={n}, scenario={scenario}", flush=True)

    raw = pd.read_csv(raw_path)
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

    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"Saved {raw_path}")
    print(f"Saved {summary_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    args = parser.parse_args()
    if args.replications < 2:
        parser.error("--replications must be at least 2")
    if args.n_jobs == 0:
        parser.error("--n-jobs cannot be zero")
    run(args.replications, args.seed, args.n_jobs)


if __name__ == "__main__":
    main()
