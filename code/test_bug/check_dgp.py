import csv
from pathlib import Path

import numpy as np

from ucm.dgp import generate_data


def check_dgp():
    """Check the IV-SV DGP and save a summary of its moments."""

    n = 500_000
    a = 1 / np.sqrt(2)
    c = np.sqrt(1 - a**2)
    beta = 1.0
    seed = 12345
    tau_z = 1.5
    tau_w = -0.75

    data = generate_data(n=n, a=a, beta=beta, seed=seed)
    misspec = generate_data(
        n=n, a=a, beta=beta, tau_z=tau_z, tau_w=tau_w, seed=seed
    )

    y = data["y"]
    x = data["x"]
    z = data["z"]
    w = data["w"]
    h_iv = data["h_iv"]
    h_sv = data["h_sv"]
    resid = y - beta * x
    resid_misspec = misspec["y"] - beta * misspec["x"]

    expected_difference = (
        tau_z * z / (a * np.sqrt(n))
        + tau_w * h_sv / (c**2 * np.sqrt(n))
    )

    rows = []

    def record(group, metric, empirical, theoretical):
        rows.append({
            "group": group,
            "metric": metric,
            "empirical": empirical,
            "theoretical": theoretical,
            "difference": float(empirical) - float(theoretical),
        })

    for name, shape in {
        "y": (n,), "x": (n,), "z": (n,), "w": (n, 2),
        "h_iv": (n,), "h_sv": (n,),
    }.items():
        record("basic", f"{name}_shape_valid", data[name].shape == shape, True)
        record("basic", f"{name}_finite", np.isfinite(data[name]).all(), True)

    record("projection", "max_abs_h_iv_error", np.max(np.abs(h_iv - a * z / 2)), 0.0)
    record("projection", "max_abs_h_sv_error", np.max(np.abs(h_sv - (x - w @ np.array([1.0, a])))), 0.0)
    record("projection", "sample_iv_coefficient", np.mean(z * x) / np.mean(z**2), a / 2)
    record("projection", "mean_h_sv_u", np.mean(h_sv * w[:, 0]), 0.0)
    record("projection", "mean_h_sv_s", np.mean(h_sv * w[:, 1]), 0.0)

    record("population", "mean_z_squared", np.mean(z**2), 2.0)
    record("population", "mean_h_sv_squared", np.mean(h_sv**2), c**2)
    record("population", "mean_z_x", np.mean(z * x), a)
    record("population", "mean_h_iv_x", np.mean(h_iv * x), a**2 / 2)
    record("population", "mean_h_sv_x", np.mean(h_sv * x), c**2)

    record("correct_specification", "mean_h_iv_residual", np.mean(h_iv * resid), 0.0)
    record("correct_specification", "mean_h_sv_residual", np.mean(h_sv * resid), 0.0)

    record("local_misspecification", "sqrt_n_mean_h_iv_residual", np.sqrt(n) * np.mean(misspec["h_iv"] * resid_misspec), tau_z)
    record("local_misspecification", "sqrt_n_mean_h_sv_residual", np.sqrt(n) * np.mean(misspec["h_sv"] * resid_misspec), tau_w)
    record("local_misspecification", "sqrt_n_iv_moment_shift", np.sqrt(n) * np.mean(h_iv * (misspec["y"] - y)), tau_z)
    record("local_misspecification", "sqrt_n_sv_moment_shift", np.sqrt(n) * np.mean(h_sv * (misspec["y"] - y)), tau_w)
    record("local_misspecification", "max_abs_outcome_change_error", np.max(np.abs(misspec["y"] - y - expected_difference)), 0.0)

    result_dir = Path(__file__).resolve().parents[2] / "result"
    result_dir.mkdir(exist_ok=True)
    output_path = result_dir / "check_dgp.csv"

    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    print(f"DGP diagnostic summary saved to {output_path}")
    return rows


if __name__ == "__main__":
    check_dgp()
