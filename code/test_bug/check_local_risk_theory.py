"""Check local-risk optima and positive-definite weights from Section 4.1."""

import csv
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ucm.risk import ellipsoid_risk, optimal_ellipsoid_weight, sensitivity


TOLERANCE = 1e-12
RHO = 1.0


def check_local_risk_theory():
    """Check simple population optima for the a=2, kappa=1 design."""
    jacobian = np.array([[2.0], [1.0]])
    covariance = np.diag([4.0, 1.0])
    # N_J=(1,-2)'/sqrt(5) spans C(J)^perp. The paper's weight family
    # K' A K + N_J B N_J' gives this implementation for A=1 and B=rho.
    orthogonal_projector = np.array([[1.0, -2.0], [-2.0, 4.0]]) / 5

    # Write K=(k, 1-2k), using the restriction KJ=1.
    # Each case below follows by minimizing its one-variable quadratic risk.
    cases = {
        "correct_specification": {
            "radii": (0.0, 0.0),
            "sensitivity": np.array([1 / 4, 1 / 2]),
            "risk": 1 / 2,
            "risk_formula": "4*k^2 + (1-2*k)^2",
        },
        "iv_may_deviate": {
            "radii": (1.0, 0.0),
            "sensitivity": np.array([2 / 9, 5 / 9]),
            "risk": 5 / 9,
            "risk_formula": "k^2 + 4*k^2 + (1-2*k)^2",
        },
        "sv_may_deviate": {
            "radii": (0.0, 1.0),
            "sensitivity": np.array([1 / 3, 1 / 3]),
            "risk": 2 / 3,
            "risk_formula": "4*k^2 + 2*(1-2*k)^2",
        },
        "both_may_deviate": {
            "radii": (1.0, 1.0),
            "sensitivity": np.array([4 / 13, 5 / 13]),
            "risk": 10 / 13,
            "risk_formula": "k^2 + (1-2*k)^2 + 4*k^2 + (1-2*k)^2",
        },
    }

    rows = []

    def record(case, formula, quantity, hand_value, algorithm_value):
        hand_value = float(hand_value)
        algorithm_value = float(algorithm_value)
        error = abs(algorithm_value - hand_value)
        rows.append({
            "case": case,
            "risk_formula": formula,
            "quantity": quantity,
            "hand_calculation": hand_value,
            "algorithm": algorithm_value,
            "absolute_error": error,
            "passed": error <= TOLERANCE,
        })

    for case, expected in cases.items():
        radii = expected["radii"]
        hand_k = expected["sensitivity"]
        hand_weight = np.outer(hand_k, hand_k) + RHO * orthogonal_projector

        algorithm_weight = optimal_ellipsoid_weight(
            jacobian, covariance, radii, rho=RHO
        )
        algorithm_k = sensitivity(jacobian, algorithm_weight).reshape(-1)
        algorithm_risk = ellipsoid_risk(
            jacobian, algorithm_weight, covariance, radii
        )

        formula = expected["risk_formula"]
        record(case, formula, "K_iv", hand_k[0], algorithm_k[0])
        record(case, formula, "K_sv", hand_k[1], algorithm_k[1])
        record(case, formula, "IV_information_share", 2 * hand_k[0], 2 * algorithm_k[0])
        record(case, formula, "optimal_risk", expected["risk"], algorithm_risk)
        record(case, formula, "KJ", 1.0, algorithm_k @ jacobian[:, 0])
        record(
            case, formula, "M_positive_definite", True,
            np.linalg.eigvalsh(algorithm_weight).min() > 0,
        )
        for row in range(2):
            for column in range(2):
                record(
                    case, formula, f"M_{row + 1}{column + 1}",
                    hand_weight[row, column], algorithm_weight[row, column],
                )

    if not all(row["passed"] for row in rows):
        failed = [row for row in rows if not row["passed"]]
        raise AssertionError(f"Theory comparison failed: {failed}")

    result_dir = Path(__file__).resolve().parents[2] / "results" / "theory_checks"
    result_dir.mkdir(parents=True, exist_ok=True)
    output_path = result_dir / "local_risk_theory_check.csv"
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    print("All hand calculations agree with the positive-definite local-risk weights.")
    for case, expected in cases.items():
        case_rows = {row["quantity"]: row for row in rows if row["case"] == case}
        print(
            f"{case}: K=({case_rows['K_iv']['algorithm']:.6f}, "
            f"{case_rows['K_sv']['algorithm']:.6f}), "
            f"risk={case_rows['optimal_risk']['algorithm']:.6f}"
        )
    print(f"Theory check saved to {output_path}")
    return rows


if __name__ == "__main__":
    check_local_risk_theory()
