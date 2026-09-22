#!/usr/bin/env python3
"""Reproduce the numerical difference diagnostics for the two main runs.

The script is read-only: it consumes the archived CSV/JSON results and prints a
JSON summary.  Regular output snapshots are matched by their exact 60 s time
stamp; the separately interpolated terminal rows are handled through metadata.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROBLEM3 = PROJECT_ROOT / "problem3"
FINE_ROOT = PROBLEM3 / "experiments" / "N160_dt5_last_hour_mean"


def load_metadata(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_regular_snapshots(path: Path) -> dict[float, list[float]]:
    result: dict[float, list[float]] = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.reader(stream)
        next(rows)
        for row in rows:
            time_s = float(row[0])
            if abs(time_s / 60.0 - round(time_s / 60.0)) <= 1.0e-9:
                result[time_s] = [float(value) for value in row[1:]]
    return result


def profile_comparison(
    left: dict[float, list[float]],
    right: dict[float, list[float]],
    start_time_s: float = 0.0,
) -> dict[str, float]:
    errors: list[float] = []
    maximum = (-1.0, 0.0, 0.0)
    common_times = sorted(set(left) & set(right))
    for time_s in common_times:
        if time_s < start_time_s:
            continue
        for index, (left_value, right_value) in enumerate(
            zip(left[time_s], right[time_s])
        ):
            error = abs(right_value - left_value)
            errors.append(error)
            if error > maximum[0]:
                maximum = (error, time_s, 0.1 * index)
    return {
        "samples": len(errors),
        "max_abs_kg_kg": maximum[0],
        "max_time_s": maximum[1],
        "max_radius_cm": maximum[2],
        "mean_abs_kg_kg": sum(errors) / len(errors),
        "rms_kg_kg": math.sqrt(sum(value * value for value in errors) / len(errors)),
    }


def richardson(metadata: dict[str, object]) -> dict[str, float]:
    convergence = metadata["convergence"]
    assert isinstance(convergence, list) and len(convergence) >= 2
    first = convergence[-2]
    second = convergence[-1]
    d1 = float(first["drying_time_difference_h"])
    d2 = float(second["drying_time_difference_h"])
    order = math.log(d1 / d2, 2.0)
    fine_error_h = d2 / (2.0**order - 1.0)
    drying_time_h = float(metadata["drying_time_h"])
    return {
        "observed_order": order,
        "fine_error_h": fine_error_h,
        "fine_error_s": 3600.0 * fine_error_h,
        "extrapolated_time_h": drying_time_h - fine_error_h,
    }


def main() -> None:
    old_metadata = load_metadata(PROBLEM3 / "run_metadata.json")
    new_metadata = load_metadata(FINE_ROOT / "run_metadata.json")
    boundary_sensitivity_metadata = load_metadata(
        PROBLEM3 / "last_point_experiment" / "run_metadata.json"
    )
    old = load_regular_snapshots(PROBLEM3 / "moisture_full_precision.csv")
    medium = load_regular_snapshots(
        FINE_ROOT
        / "convergence_levels"
        / "N80_dt10s"
        / "moisture_full_precision.csv"
    )
    new = load_regular_snapshots(FINE_ROOT / "moisture_full_precision.csv")

    old_time_s = float(old_metadata["drying_time_s"])
    new_time_s = float(new_metadata["drying_time_s"])
    convergence = new_metadata["convergence"]
    assert isinstance(convergence, list)
    medium_time_s = float(convergence[-1]["coarse_drying_time_h"]) * 3600.0

    old_richardson = richardson(old_metadata)
    new_richardson = richardson(new_metadata)
    output = {
        "drying_time": {
            "old_s": old_time_s,
            "new_s": new_time_s,
            "new_minus_old_s": new_time_s - old_time_s,
            "relative_change_percent": 100.0 * (new_time_s - old_time_s) / old_time_s,
        },
        "controlled_inputs": {
            "same_input_sha256": old_metadata["input_sha256"]
            == new_metadata["input_sha256"],
            "same_physical_parameters": old_metadata["physical_parameters"]
            == new_metadata["physical_parameters"],
            "same_environment": old_metadata["environment"]
            == new_metadata["environment"],
        },
        "decomposition": {
            "N80_dt30_to_N80_dt10_s": medium_time_s - old_time_s,
            "N80_dt10_to_N160_dt5_s": new_time_s - medium_time_s,
            "residual_s": (new_time_s - old_time_s)
            - (medium_time_s - old_time_s)
            - (new_time_s - medium_time_s),
        },
        "profiles": {
            "old_to_new_all": profile_comparison(old, new),
            "old_to_new_after_6h": profile_comparison(old, new, 6.0 * 3600.0),
            "old_to_medium_all": profile_comparison(old, medium),
            "medium_to_new_all": profile_comparison(medium, new),
        },
        "richardson": {
            "old": old_richardson,
            "new": new_richardson,
            "extrapolated_time_difference_s": 3600.0
            * abs(
                old_richardson["extrapolated_time_h"]
                - new_richardson["extrapolated_time_h"]
            ),
        },
        "boundary_sensitivity": {
            "last_point_minus_last_hour_mean_s": 3600.0
            * (
                float(boundary_sensitivity_metadata["drying_time_h"])
                - float(old_metadata["drying_time_h"])
            )
        },
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
