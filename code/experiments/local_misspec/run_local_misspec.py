"""Monte Carlo experiment for local misspecification in Section 5.2."""

import argparse
import csv
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ucm.dgp import generate_data
from ucm.estimators import combined, iv, sv
from ucm.inference import robust_ci, wald_ci
from ucm.projected import projected_covariance, projected_scores, sample_projections
from ucm.risk import (
    DEFAULT_IDENTIFICATION_FLOOR, box_bias_bound, box_risk, optimal_box_weight,
    sensitivity,
)


A = 2.0
BETA = 1.0
N_VALUES = tuple(range(200, 2001, 200))
KAPPA_VALUES = (0.0, 0.5, 1.0)
REGIMES = ("iv", "sv", "both")
METHODS = ("iv", "sv", "equal", "local_risk")
DEFAULT_SEED = 20260920
IDENTIFICATION_FLOOR = DEFAULT_IDENTIFICATION_FLOOR
DEFAULT_N_JOBS = 8
RAW_FIELDS = (
    "seed", "identification_floor", "draw_seed", "regime", "kappa", "n",
    "replication", "tau_z", "tau_w", "method", "beta_hat", "sigma",
    "iv_share", "bias_bound", "squared_error", "wald_covered", "wald_length",
    "robust_covered", "robust_length",
)
TOTAL_FIELDS = (
    "estimate", "estimate_sq", "squared_error", "squared_error_sq",
    "wald_covered", "wald_length", "robust_covered", "robust_length",
    "iv_share",
)


def design_points(kappas):
    """Return prespecified symmetric boxes and all of their vertices."""
    for kappa in kappas:
        if kappa == 0:
            yield "none", kappa, (0.0, 0.0), ((0.0, 0.0),)
            continue
        yield "iv", kappa, (kappa, 0.0), ((-kappa, 0.0), (kappa, 0.0))
        yield "sv", kappa, (0.0, kappa), ((0.0, -kappa), (0.0, kappa))
        yield "both", kappa, (kappa, kappa), (
            (-kappa, -kappa), (-kappa, kappa), (kappa, -kappa), (kappa, kappa)
        )


def estimate(data, y, h_iv, h_sv, bounds):
    """Estimate equal-weight and sample-optimal local-risk GMM."""
    x, z, w = data["x"], data["z"], data["w"]
    n = len(y)
    H = np.column_stack((h_iv, h_sv))
    G = H.T @ x.reshape(-1, 1) / n
    beta0 = combined(x, y, h_iv, h_sv)
    first_scores = projected_scores(x, y, z, w, beta0, h_iv, h_sv)
    Omega0 = projected_covariance(first_scores)
    equal_weight = np.eye(2)
    risk_weight = optimal_box_weight(G, Omega0, bounds, IDENTIFICATION_FLOOR)
    methods = {
        "iv": (h_iv, None, iv(x, y, h_iv)),
        "sv": (h_sv, None, sv(x, y, h_sv)),
        "equal": (H, equal_weight, combined(x, y, h_iv, h_sv, equal_weight)),
        "local_risk": (H, risk_weight, combined(x, y, h_iv, h_sv, risk_weight)),
    }

    results = {}
    for method, (instruments, weight, beta_hat) in methods.items():
        beta_hat = float(np.asarray(beta_hat).item())
        scores = projected_scores(x, y, z, w, beta_hat, h_iv, h_sv)
        if method == "iv":
            scores = scores[:, :1]
        elif method == "sv":
            scores = scores[:, 1:]
        Omega = projected_covariance(scores)
        jacobian = instruments.T @ x.reshape(-1, 1) / n
        M = np.eye(instruments.shape[1]) if weight is None else weight
        K = sensitivity(jacobian, M).reshape(-1)
        sigma = np.sqrt(float(K @ Omega @ K))
        wald_low, wald_high = wald_ci(beta_hat, sigma / np.sqrt(n))
        if method in ("equal", "local_risk"):
            bias_bound = box_bias_bound(K, bounds)
            robust_low, robust_high = robust_ci(beta_hat, sigma, bias_bound, n)
            robust_covered = robust_low <= BETA <= robust_high
            robust_length = robust_high - robust_low
            iv_share = K[0] * G[0, 0]
        else:
            bias_bound = np.nan
            robust_covered = np.nan
            robust_length = np.nan
            iv_share = 1.0 if method == "iv" else 0.0
        results[method] = {
            "beta_hat": beta_hat,
            "sigma": sigma,
            "iv_share": iv_share,
            "bias_bound": bias_bound,
            "squared_error": (beta_hat - BETA) ** 2,
            "wald_covered": wald_low <= BETA <= wald_high,
            "wald_length": wald_high - wald_low,
            "robust_covered": robust_covered,
            "robust_length": robust_length,
        }
    return results


def population_risk(method, bounds):
    G = np.array([[A**2 / 2], [1.0]])
    Omega = np.diag([A**2, 1.0])
    if method == "iv":
        M = np.diag([1.0, 0.0])
    elif method == "sv":
        M = np.diag([0.0, 1.0])
    elif method == "equal":
        M = np.eye(2)
    else:
        M = optimal_box_weight(G, Omega, bounds, IDENTIFICATION_FLOOR)
    return box_risk(G, M, Omega, bounds)


def run_configuration(
    index, n, regime, kappa, bounds, vertices, draw_seeds, seed, part_path
):
    """Run one local-misspecification design configuration."""
    totals = {}
    with part_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RAW_FIELDS)
        for replication, draw_seed in enumerate(draw_seeds):
            draw_seed = int(draw_seed)
            base = generate_data(n=n, a=A, c=1.0, beta=BETA, seed=draw_seed)
            h_iv, h_sv = sample_projections(base["x"], base["z"], base["w"])
            for tau_z, tau_w in vertices:
                y = base["y"] + (
                    tau_z * base["z"] / (A * np.sqrt(n))
                    + tau_w * base["h_sv"] / np.sqrt(n)
                )
                for method, result in estimate(base, y, h_iv, h_sv, bounds).items():
                    writer.writerow({
                        "seed": seed,
                        "identification_floor": IDENTIFICATION_FLOOR,
                        "draw_seed": draw_seed, "regime": regime,
                        "kappa": kappa, "n": n, "replication": replication,
                        "tau_z": tau_z, "tau_w": tau_w, "method": method,
                        **result,
                    })
                    key = (regime, kappa, n, tau_z, tau_w, method)
                    if key not in totals:
                        totals[key] = {name: 0.0 for name in TOTAL_FIELDS}
                    total = totals[key]
                    total["estimate"] += result["beta_hat"]
                    total["estimate_sq"] += result["beta_hat"] ** 2
                    total["squared_error"] += result["squared_error"]
                    total["squared_error_sq"] += result["squared_error"] ** 2
                    total["wald_covered"] += result["wald_covered"]
                    total["wald_length"] += result["wald_length"]
                    total["iv_share"] += result["iv_share"]
                    if method in ("equal", "local_risk"):
                        total["robust_covered"] += result["robust_covered"]
                        total["robust_length"] += result["robust_length"]
    return index, n, regime, kappa, totals


def run(
    replications=1000, seed=DEFAULT_SEED, n_values=N_VALUES,
    kappas=KAPPA_VALUES, n_jobs=DEFAULT_N_JOBS,
):
    rng = np.random.default_rng(seed)
    configurations = []
    for n in n_values:
        for regime, kappa, bounds, vertices in design_points(kappas):
            draw_seeds = rng.integers(0, 2**63, size=replications, dtype=np.int64)
            configurations.append((n, regime, kappa, bounds, vertices, draw_seeds))

    output_dir = Path(__file__).resolve().parents[3] / "results" / "local_misspec"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "local_misspec_replications.csv"
    corner_path = output_dir / "local_misspec_corners.csv"
    summary_path = output_dir / "local_misspec_worst_case.csv"
    totals = {}

    with TemporaryDirectory(prefix="local_misspec_parts_", dir=output_dir) as temp_dir:
        temp_dir = Path(temp_dir)
        tasks = [
            delayed(run_configuration)(
                index, n, regime, kappa, bounds, vertices, draw_seeds, seed,
                temp_dir / f"part_{index:04d}.csv",
            )
            for index, (n, regime, kappa, bounds, vertices, draw_seeds)
            in enumerate(configurations)
        ]
        with parallel_config(backend="loky", inner_max_num_threads=1):
            completed = Parallel(n_jobs=n_jobs, verbose=10)(tasks)

        with raw_path.open("w", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=RAW_FIELDS)
            writer.writeheader()
            for index, n, regime, kappa, configuration_totals in sorted(completed):
                with (temp_dir / f"part_{index:04d}.csv").open() as part_file:
                    shutil.copyfileobj(part_file, output_file)
                totals.update(configuration_totals)
                print(f"Finished n={n}, regime={regime}, kappa={kappa}", flush=True)

    rows = []
    for (regime, kappa, n, tau_z, tau_w, method), total in totals.items():
        mean = total["estimate"] / replications
        bias = mean - BETA
        variance = max(0.0, total["estimate_sq"] / replications - mean**2)
        mean_squared_error = total["squared_error"] / replications
        squared_error_variance = max(
            0.0, total["squared_error_sq"] / replications - mean_squared_error**2
        )
        rows.append({
            "regime": regime, "kappa": kappa, "n": n,
            "identification_floor": IDENTIFICATION_FLOOR, "tau_z": tau_z, "tau_w": tau_w, "method": method,
            "replications": replications, "bias": bias,
            "scaled_squared_bias": n * bias**2,
            "scaled_variance": n * variance,
            "scaled_mse": n * mean_squared_error,
            "scaled_mse_mc_se": n * np.sqrt(squared_error_variance / replications),
            "wald_coverage": total["wald_covered"] / replications,
            "mean_wald_length": total["wald_length"] / replications,
            "robust_coverage": (
                total["robust_covered"] / replications
                if method in ("equal", "local_risk") else np.nan
            ),
            "mean_robust_length": (
                total["robust_length"] / replications
                if method in ("equal", "local_risk") else np.nan
            ),
            "mean_iv_share": total["iv_share"] / replications,
        })
    corners = pd.DataFrame(rows)
    corners.to_csv(corner_path, index=False)

    worst_rows = []
    for (regime, kappa, n, method), frame in corners.groupby(
        ["regime", "kappa", "n", "method"], sort=False
    ):
        worst = frame.loc[frame["scaled_mse"].idxmax()]
        bounds = (kappa if regime in ("iv", "both") else 0.0,
                  kappa if regime in ("sv", "both") else 0.0)
        worst_rows.append({
            "regime": regime, "kappa": kappa, "n": n, "method": method,
            "identification_floor": IDENTIFICATION_FLOOR,
            "replications": replications,
            "worst_tau_z": worst["tau_z"], "worst_tau_w": worst["tau_w"],
            "worst_scaled_mse": worst["scaled_mse"],
            "worst_scaled_mse_mc_se": worst["scaled_mse_mc_se"],
            "worst_scaled_squared_bias": worst["scaled_squared_bias"],
            "worst_scaled_variance": worst["scaled_variance"],
            "theoretical_worst_risk": population_risk(method, bounds),
            "min_wald_coverage": frame["wald_coverage"].min(),
            "mean_wald_length": frame["mean_wald_length"].mean(),
            "min_robust_coverage": frame["robust_coverage"].min(),
            "mean_robust_length": frame["mean_robust_length"].mean(),
            "mean_iv_share": frame["mean_iv_share"].mean(),
        })
    worst = pd.DataFrame(worst_rows)
    worst.to_csv(summary_path, index=False)
    print(f"Saved {raw_path}")
    print(f"Saved {corner_path}")
    print(f"Saved {summary_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replications", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-values", type=int, nargs="+", default=N_VALUES)
    parser.add_argument("--kappas", type=float, nargs="+", default=KAPPA_VALUES)
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    args = parser.parse_args()
    if args.replications < 2 or any(n < 10 for n in args.n_values):
        parser.error("Use at least 2 replications and n >= 10.")
    if any(kappa < 0 for kappa in args.kappas):
        parser.error("All kappa values must be nonnegative.")
    if args.n_jobs == 0:
        parser.error("--n-jobs cannot be zero")
    run(args.replications, args.seed, args.n_values, args.kappas, args.n_jobs)


if __name__ == "__main__":
    main()
