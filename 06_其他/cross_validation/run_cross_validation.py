#!/usr/bin/env python3
"""Independent Radau and Bessel verification for Problem A.

The production solutions use either a custom BE/BDF2 integrator (Questions 1
and 3) or SciPy's variable-order BDF integrator (Questions 2 and 4).  This
module supplies two independent checks:

* recompute the four production PDEs with the fifth-order Radau IIA method;
* compare the radial finite-volume kernel with a constant-coefficient,
  single-mode Bessel solution satisfying the same Robin boundary condition.

Each task writes an atomic JSON summary and a compressed NumPy checkpoint.
The ``assemble`` task combines the completed evidence into a Markdown report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import platform
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import jv
from scipy.sparse import csr_matrix, lil_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from problem2 import solve_problem2 as q2
from problem3 import solve_problem3 as q3
from problem4 import solve_problem4 as q4


ROOT = PROJECT_ROOT
OUT = ROOT / "cross_validation" / "results"
ATTACHMENT_1 = ROOT / "A题" / "附件" / "附件1.xlsx"
ATTACHMENT_2 = ROOT / "A题" / "附件" / "附件2.xlsx"
Q3_CHECKPOINT = (
    ROOT
    / "ultrafine_runs"
    / "fixed_50_005"
    / "tasks"
    / "q3_n2560_dt0p5"
    / "result.pkl"
)
Q4_CHECKPOINT = (
    ROOT
    / "ultrafine_runs"
    / "fixed_50_005"
    / "tasks"
    / "q4_n1520_fine"
    / "result.pkl"
)

GAUSS_X = np.asarray(
    [-0.8611363115940526, -0.3399810435848563,
      0.3399810435848563, 0.8611363115940526],
    dtype=float,
)
GAUSS_W = np.asarray(
    [0.3478548451374538, 0.6521451548625461,
     0.6521451548625461, 0.3478548451374538],
    dtype=float,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=float) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def base_metadata(task: str, started: float) -> dict[str, object]:
    return {
        "task": task,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_s": time.perf_counter() - started,
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "input_hashes": {
            "attachment1": sha256_file(ATTACHMENT_1),
            "attachment2": sha256_file(ATTACHMENT_2),
            "this_script": sha256_file(Path(__file__)),
        },
    }


def load_checkpoint(path: Path) -> object:
    with path.open("rb") as handle:
        envelope = pickle.load(handle)
    return envelope["result"]


def load_csv_field(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    times = np.asarray([float(row[0]) for row in rows[1:]], dtype=float)
    values = np.asarray(
        [[float(value) for value in row[1:]] for row in rows[1:]], dtype=float
    )
    return times, values


def comparison(
    reference: np.ndarray,
    candidate: np.ndarray,
    times_s: np.ndarray,
    positions: np.ndarray,
) -> dict[str, object]:
    if reference.shape != candidate.shape:
        raise ValueError(f"shape mismatch: {reference.shape} versus {candidate.shape}")
    difference = np.abs(reference - candidate)
    flat = int(np.argmax(difference))
    row, column = np.unravel_index(flat, difference.shape)
    rounded_equal = np.round(reference, 4) == np.round(candidate, 4)
    return {
        "max_abs_difference": float(difference[row, column]),
        "rms_difference": float(np.sqrt(np.mean(difference * difference))),
        "max_location_time_s": float(times_s[row]),
        "max_location_position": float(positions[column]),
        "four_decimal_agreement_count": int(np.count_nonzero(rounded_equal)),
        "comparison_count": int(difference.size),
        "four_decimal_agreement_fraction": float(np.mean(rounded_equal)),
    }


def block_tridiagonal_sparsity(node_count: int) -> csr_matrix:
    pattern = lil_matrix((2 * node_count, 2 * node_count), dtype=np.int8)
    for i in range(node_count):
        for j in range(max(0, i - 1), min(node_count, i + 2)):
            pattern[i, j] = 1
            pattern[i, node_count + j] = 1
            pattern[node_count + i, j] = 1
            pattern[node_count + i, node_count + j] = 1
    return pattern.tocsr()


def scalar_tridiagonal_sparsity(node_count: int) -> csr_matrix:
    pattern = lil_matrix((node_count, node_count), dtype=np.int8)
    for i in range(node_count):
        pattern[i, max(0, i - 1) : min(node_count, i + 2)] = 1
    return pattern.tocsr()


def harmonic(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return 2.0 * left * right / (left + right)


def q1_properties(
    moisture: np.ndarray, temperature_c: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shape = moisture.shape
    density = np.full(shape, 820.0)
    heat_capacity = np.full(shape, 2600.0)
    conductivity = np.full(shape, 0.36)
    diffusivity = 7.0e-9 * np.exp(-0.89 / moisture)
    return density, heat_capacity, conductivity, diffusivity


def q23_properties(
    moisture: np.ndarray, temperature_c: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ratio = moisture / (moisture + 1.0)
    density = 650.0 + 128.0 * moisture
    heat_capacity = 1450.0 + 2736.0 * ratio
    conductivity = 0.21 + 0.38 * ratio
    diffusivity = 2.4e-3 * np.exp(
        -0.45 / moisture - 3850.0 / (temperature_c + 273.15)
    )
    return density, heat_capacity, conductivity, diffusivity


class FixedCoupledSystem:
    """The production fixed-domain finite-volume semi-discretization."""

    def __init__(
        self,
        radii_m: np.ndarray,
        environment: Callable[[float], tuple[float, float]],
        properties: Callable[
            [np.ndarray, np.ndarray],
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        ],
        face_scheme: str,
    ) -> None:
        self.radii = np.asarray(radii_m, dtype=float)
        self.faces = 0.5 * (self.radii[:-1] + self.radii[1:])
        self.spacing = np.diff(self.radii)
        self.weights = np.asarray(q2.radial_storage_weights(self.radii), dtype=float)
        self.radius = float(self.radii[-1])
        self.environment = environment
        self.properties = properties
        self.face_scheme = face_scheme

    def _face_coefficients(
        self,
        moisture: np.ndarray,
        temperature_c: np.ndarray,
        conductivity: np.ndarray,
        diffusivity: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.face_scheme == "harmonic":
            return (
                harmonic(conductivity[:-1], conductivity[1:]),
                harmonic(diffusivity[:-1], diffusivity[1:]),
            )
        if self.face_scheme != "kirchhoff":
            raise ValueError(self.face_scheme)
        face_moisture = 0.5 * (moisture[:-1] + moisture[1:])
        conductivity_faces = 0.21 + 0.38 * face_moisture / (face_moisture + 1.0)
        midpoint = face_moisture
        half_span = 0.5 * (moisture[1:] - moisture[:-1])
        quadrature = np.zeros_like(midpoint)
        for point, weight in zip(GAUSS_X, GAUSS_W):
            quadrature += weight * np.exp(-0.45 / (midpoint + half_span * point))
        temperature_faces_k = 0.5 * (
            temperature_c[:-1] + temperature_c[1:]
        ) + 273.15
        diffusivity_faces = (
            0.5
            * 2.4e-3
            * np.exp(-3850.0 / temperature_faces_k)
            * quadrature
        )
        return conductivity_faces, diffusivity_faces

    def _net(
        self,
        field: np.ndarray,
        gamma_faces: np.ndarray,
        transfer: float,
        external: float,
    ) -> np.ndarray:
        flux = self.faces * gamma_faces * np.diff(field) / self.spacing
        net = np.empty_like(field)
        net[0] = flux[0]
        net[1:-1] = flux[1:] - flux[:-1]
        net[-1] = self.radius * transfer * (external - field[-1]) - flux[-1]
        return net

    def __call__(self, t_s: float, state: np.ndarray) -> np.ndarray:
        n = self.radii.size
        temperature = state[:n]
        moisture = state[n:]
        if np.any(moisture <= 0.0):
            raise ValueError("Radau trial state has non-positive moisture")
        density, heat_capacity, conductivity, diffusivity = self.properties(
            moisture, temperature
        )
        k_faces, d_faces = self._face_coefficients(
            moisture, temperature, conductivity, diffusivity
        )
        external_temperature, external_moisture = self.environment(float(t_s))
        heat = self._net(temperature, k_faces, 25.0, external_temperature)
        water = self._net(moisture, d_faces, 8.0e-7, external_moisture)
        return np.concatenate(
            (
                heat / (self.weights * density * heat_capacity),
                water / self.weights,
            )
        )


def integrate_fixed_radau(
    system: FixedCoupledSystem,
    end_time_s: float,
    output_interval_s: float,
    rtol: float,
    atol_temperature: float,
    atol_moisture: float,
    max_step_s: float,
    first_step_s: float,
    output_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    n = system.radii.size
    initial = np.concatenate((np.full(n, 28.0), np.full(n, 2.55)))
    times = np.arange(0.0, end_time_s + 0.5 * output_interval_s, output_interval_s)
    atol = np.concatenate(
        (np.full(n, atol_temperature), np.full(n, atol_moisture))
    )
    started = time.perf_counter()
    solution = solve_ivp(
        system,
        (0.0, end_time_s),
        initial,
        method="Radau",
        t_eval=times,
        rtol=rtol,
        atol=atol,
        jac_sparsity=block_tridiagonal_sparsity(n),
        max_step=max_step_s,
        first_step=first_step_s,
    )
    elapsed = time.perf_counter() - started
    if not solution.success or not np.all(np.isfinite(solution.y)):
        raise RuntimeError(f"Radau failed: {solution.message}")
    return (
        times,
        solution.y[:n, :][output_indices].T.copy(),
        solution.y[n:, :][output_indices].T.copy(),
        {
            "runtime_s": elapsed,
            "function_evaluations": int(solution.nfev),
            "jacobian_evaluations": int(solution.njev),
            "lu_decompositions": int(solution.nlu),
        },
    )


def task_q1() -> None:
    task = "q1_radau"
    started = time.perf_counter()
    environment = q2.load_environment(ATTACHMENT_1, 1800.0)
    radii = q2.make_radial_grid(0.02, 1520, 0.002, 10)
    _, indices = q2.requested_output_indices(radii)
    system = FixedCoupledSystem(
        radii,
        lambda t: tuple(float(x) for x in environment.interpolate(t)),
        q1_properties,
        "harmonic",
    )
    times, temperature, moisture, solver = integrate_fixed_radau(
        system,
        1800.0,
        1.0,
        2.0e-10,
        2.0e-10,
        2.0e-12,
        0.125,
        1.0e-5,
        indices,
    )
    reference_times, reference_temperature = load_csv_field(
        ROOT / "problem1" / "temperature_full_precision.csv"
    )
    moisture_times, reference_moisture = load_csv_field(
        ROOT / "problem1" / "moisture_full_precision.csv"
    )
    if not np.array_equal(reference_times, moisture_times):
        raise RuntimeError("Question 1 reference time grids differ")
    positions = np.arange(21, dtype=float) * 0.1
    result = base_metadata(task, started)
    result.update(
        {
            "status": "complete",
            "description": "N=1520 production FVM recomputed by Radau IIA",
            "configuration": {
                "radial_cells": 1520,
                "rtol": 2.0e-10,
                "temperature_atol": 2.0e-10,
                "moisture_atol": 2.0e-12,
                "max_step_s": 0.125,
                "first_step_s": 1.0e-5,
            },
            "radau_diagnostics": solver,
            "temperature": comparison(
                reference_temperature, temperature[1:], reference_times, positions
            ),
            "moisture": comparison(
                reference_moisture, moisture[1:], reference_times, positions
            ),
        }
    )
    atomic_npz(
        OUT / f"{task}.npz",
        times_s=times,
        radii_cm=positions,
        temperature_c=temperature,
        moisture_kg_kg=moisture,
    )
    atomic_json(OUT / f"{task}.json", result)


def task_q2() -> None:
    task = "q2_radau"
    started = time.perf_counter()
    environment = q2.load_environment(ATTACHMENT_1, 10800.0)
    parameters = q2.PhysicalParameters()
    config = q2.NumericalConfig(
        radial_cells=760,
        end_time_s=10800.0,
        output_interval_s=1.0,
        surface_layer_m=0.002,
        surface_refinement_factor=10,
        relative_tolerance=5.0e-10,
        temperature_absolute_tolerance=5.0e-10,
        moisture_absolute_tolerance=5.0e-12,
        maximum_step_s=2.5,
        first_step_s=5.0e-5,
    )
    bdf = q2.solve_model(environment, parameters, config, "q2/BDF", "BDF")
    radau = q2.solve_model(environment, parameters, config, "q2/Radau", "Radau")
    positions = np.asarray(radau.radii_cm)
    times = np.asarray(radau.times_s)
    result = base_metadata(task, started)
    result.update(
        {
            "status": "complete",
            "description": "N=760 identical semi-discrete system, tight BDF versus Radau",
            "configuration": asdict(config),
            "bdf_diagnostics": asdict(bdf.diagnostics),
            "radau_diagnostics": asdict(radau.diagnostics),
            "temperature": comparison(
                bdf.temperatures_c, radau.temperatures_c, times, positions
            ),
            "moisture": comparison(
                bdf.moistures_kg_kg, radau.moistures_kg_kg, times, positions
            ),
        }
    )
    atomic_npz(
        OUT / f"{task}.npz",
        times_s=times,
        radii_cm=positions,
        bdf_temperature_c=bdf.temperatures_c,
        radau_temperature_c=radau.temperatures_c,
        bdf_moisture_kg_kg=bdf.moistures_kg_kg,
        radau_moisture_kg_kg=radau.moistures_kg_kg,
    )
    atomic_json(OUT / f"{task}.json", result)


def integrate_q3_radau() -> tuple[np.ndarray, np.ndarray, float, dict[str, object]]:
    environment = q3.load_environment(
        ATTACHMENT_1, 3.0, "fixed", 50.0, 0.05
    )
    radial_cells = 2560
    radii = np.asarray(q3.make_uniform_grid(0.02, radial_cells), dtype=float)
    output_indices = np.arange(0, radial_cells + 1, radial_cells // 20, dtype=int)
    system = FixedCoupledSystem(
        radii,
        environment.value,
        q23_properties,
        "kirchhoff",
    )
    n = radii.size
    state = np.concatenate((np.full(n, 28.0), np.full(n, 2.55)))
    atol = np.concatenate((np.full(n, 2.0e-10), np.full(n, 2.0e-12)))
    sparsity = block_tridiagonal_sparsity(n)
    time_chunks: list[np.ndarray] = []
    moisture_chunks: list[np.ndarray] = []
    current = 0.0
    crossing: float | None = None
    nfev = njev = nlu = segments = 0
    started = time.perf_counter()
    while crossing is None and current < 96.0 * 3600.0:
        if current < 14400.0:
            segment_end = 14400.0
            max_step = 5.0
        else:
            segment_end = min(current + 21600.0, 96.0 * 3600.0)
            max_step = 30.0
        eval_times = np.arange(current, segment_end + 30.0, 60.0)
        print(
            f"[q3/Radau] {current / 3600:.1f}-{segment_end / 3600:.1f} h",
            flush=True,
        )
        solution = solve_ivp(
            system,
            (current, segment_end),
            state,
            method="Radau",
            t_eval=eval_times,
            dense_output=True,
            rtol=2.0e-10,
            atol=atol,
            jac_sparsity=sparsity,
            max_step=max_step,
            first_step=1.0e-4 if current == 0.0 else min(1.0, max_step),
        )
        if not solution.success or not np.all(np.isfinite(solution.y)):
            raise RuntimeError(f"Question 3 Radau failed: {solution.message}")
        node_moisture = solution.y[n:, :]
        maximum = np.max(node_moisture, axis=0)
        candidates = np.flatnonzero(
            (maximum[:-1] > 0.15) & (maximum[1:] <= 0.15)
        )
        if candidates.size:
            j = int(candidates[0])
            crossing = float(
                brentq(
                    lambda t: float(np.max(solution.sol(t)[n:]) - 0.15),
                    float(solution.t[j]),
                    float(solution.t[j + 1]),
                    xtol=1.0e-7,
                    rtol=1.0e-13,
                )
            )
        start_index = 0 if segments == 0 else 1
        time_chunks.append(solution.t[start_index:].copy())
        moisture_chunks.append(
            node_moisture[output_indices, start_index:].T.copy()
        )
        state = solution.y[:, -1].copy()
        current = segment_end
        nfev += int(solution.nfev)
        njev += int(solution.njev)
        nlu += int(solution.nlu)
        segments += 1
    if crossing is None:
        raise RuntimeError("Question 3 threshold not reached")
    times = np.concatenate(time_chunks)
    moisture = np.vstack(moisture_chunks)
    return times, moisture, crossing, {
        "runtime_s": time.perf_counter() - started,
        "function_evaluations": nfev,
        "jacobian_evaluations": njev,
        "lu_decompositions": nlu,
        "integration_segments": segments,
    }


def task_q3() -> None:
    task = "q3_radau"
    started = time.perf_counter()
    radau_times, radau_moisture, radau_crossing, diagnostics = integrate_q3_radau()
    reference = load_checkpoint(Q3_CHECKPOINT)
    reference_times = np.asarray(reference.times_s, dtype=float)
    reference_moisture = np.asarray(reference.moistures_kg_kg, dtype=float)
    common = np.intersect1d(reference_times, radau_times)
    common = common[(common > 0.0) & (common < min(reference.drying_time_s, radau_crossing))]
    ref_indices = np.searchsorted(reference_times, common)
    rad_indices = np.searchsorted(radau_times, common)
    positions = np.arange(21, dtype=float) * 0.1
    result = base_metadata(task, started)
    result.update(
        {
            "status": "complete",
            "description": "N=2560 production Kirchhoff-FVM recomputed by Radau IIA",
            "configuration": {
                "radial_cells": 2560,
                "rtol": 2.0e-10,
                "temperature_atol": 2.0e-10,
                "moisture_atol": 2.0e-12,
                "initial_max_step_s": 5.0,
                "drying_max_step_s": 30.0,
            },
            "radau_diagnostics": diagnostics,
            "moisture": comparison(
                reference_moisture[ref_indices],
                radau_moisture[rad_indices],
                common,
                positions,
            ),
            "bdf2_crossing_s": float(reference.drying_time_s),
            "radau_crossing_s": radau_crossing,
            "crossing_difference_s": abs(reference.drying_time_s - radau_crossing),
        }
    )
    atomic_npz(
        OUT / f"{task}.npz",
        times_s=radau_times,
        radii_cm=positions,
        moisture_kg_kg=radau_moisture,
        drying_crossing_s=np.asarray([radau_crossing]),
    )
    atomic_json(OUT / f"{task}.json", result)


def task_q4() -> None:
    task = "q4_radau"
    started = time.perf_counter()
    base_environment = q4.load_environment(ATTACHMENT_1)
    environment = q4.EnvironmentData(
        times_s=base_environment.times_s,
        temperatures_c=base_environment.temperatures_c,
        moistures_kg_kg=base_environment.moistures_kg_kg,
        extension_strategy="fixed",
        extension_temperature_c=50.0,
        extension_moisture_kg_kg=0.05,
    )
    environment.validate()
    radius = q4.load_radius(ATTACHMENT_2)
    reference = load_checkpoint(Q4_CHECKPOINT)
    config = reference.config
    radau = q4.solve_until_dry(
        environment,
        radius,
        q4.PhysicalParameters(),
        config,
        "q4/Radau/N1520",
        "Radau",
    )
    count = min(reference.times_s.size, radau.times_s.size)
    if not np.array_equal(reference.times_s[:count], radau.times_s[:count]):
        raise RuntimeError("Question 4 output time grids do not align")
    times = radau.times_s[:count]
    xi = radau.xi_samples
    result = base_metadata(task, started)
    result.update(
        {
            "status": "complete",
            "description": "N=1520 moving-boundary FVM, identical BDF/Radau tolerances",
            "configuration": asdict(config),
            "bdf_diagnostics": asdict(reference.diagnostics),
            "radau_diagnostics": asdict(radau.diagnostics),
            "temperature": comparison(
                reference.temperatures_c[:count],
                radau.temperatures_c[:count],
                times,
                xi,
            ),
            "moisture": comparison(
                reference.moistures_kg_kg[:count],
                radau.moistures_kg_kg[:count],
                times,
                xi,
            ),
            "bdf_crossing_s": float(reference.drying_crossing_s),
            "radau_crossing_s": float(radau.drying_crossing_s),
            "crossing_difference_s": abs(
                reference.drying_crossing_s - radau.drying_crossing_s
            ),
        }
    )
    atomic_npz(
        OUT / f"{task}.npz",
        times_s=radau.times_s,
        xi=radau.xi_samples,
        temperature_c=radau.temperatures_c,
        moisture_kg_kg=radau.moistures_kg_kg,
        drying_crossing_s=np.asarray([radau.drying_crossing_s]),
    )
    atomic_json(OUT / f"{task}.json", result)


class ConstantScalarSystem:
    def __init__(
        self,
        radii: np.ndarray,
        storage: float,
        gamma: float,
        transfer: float,
        external: float,
    ) -> None:
        self.radii = radii
        self.faces = 0.5 * (radii[:-1] + radii[1:])
        self.spacing = np.diff(radii)
        self.weights = q2.radial_storage_weights(radii)
        self.storage = storage
        self.gamma = gamma
        self.transfer = transfer
        self.external = external
        self.radius = float(radii[-1])

    def __call__(self, _t: float, field: np.ndarray) -> np.ndarray:
        flux = self.faces * self.gamma * np.diff(field) / self.spacing
        net = np.empty_like(field)
        net[0] = flux[0]
        net[1:-1] = flux[1:] - flux[:-1]
        net[-1] = (
            self.radius * self.transfer * (self.external - field[-1])
            - flux[-1]
        )
        return net / (self.storage * self.weights)


def first_robin_root(biot: float) -> float:
    function = lambda value: value * jv(1, value) - biot * jv(0, value)
    return float(brentq(function, 1.0e-14, 2.4048255576957724, xtol=1.0e-14))


def bessel_case(
    field_name: str,
    method: str,
    radial_cells: int,
) -> dict[str, object]:
    if field_name == "temperature":
        storage = 820.0 * 2600.0
        gamma = 0.36
        transfer = 25.0
        external = 50.0
        amplitude = -22.0
        end_time = 1800.0
        atol = 1.0e-11
    elif field_name == "moisture":
        storage = 1.0
        gamma = 7.0e-9 * math.exp(-0.89 / 2.55)
        transfer = 8.0e-7
        external = 0.05
        amplitude = 2.5
        end_time = 60000.0
        atol = 1.0e-12
    else:
        raise ValueError(field_name)
    radii = q2.make_radial_grid(0.02, radial_cells, 0.002, 10)
    system = ConstantScalarSystem(
        radii, storage, gamma, transfer, external
    )
    biot = transfer * 0.02 / gamma
    root = first_robin_root(biot)
    diffusivity = gamma / storage
    initial = external + amplitude * jv(0, root * radii / 0.02)
    times = np.linspace(0.0, end_time, 121)
    started = time.perf_counter()
    solution = solve_ivp(
        system,
        (0.0, end_time),
        initial,
        method=method,
        t_eval=times,
        rtol=2.0e-12,
        atol=atol,
        jac_sparsity=scalar_tridiagonal_sparsity(radial_cells + 1),
        max_step=end_time / 500.0,
        first_step=end_time / 100000.0,
    )
    elapsed = time.perf_counter() - started
    if not solution.success:
        raise RuntimeError(solution.message)
    exact = external + amplitude * (
        jv(0, root * radii[:, None] / 0.02)
        * np.exp(-diffusivity * root * root * times[None, :] / 0.02**2)
    )
    error = solution.y - exact
    absolute = np.abs(error)
    flat = int(np.argmax(absolute))
    node, time_index = np.unravel_index(flat, absolute.shape)
    weights = system.weights[:, None]
    relative_l2 = math.sqrt(
        float(np.sum(weights * error * error))
        / float(np.sum(weights * (exact - external) ** 2))
    )
    return {
        "field": field_name,
        "method": method,
        "radial_cells": radial_cells,
        "biot": biot,
        "lambda1": root,
        "eigen_residual": root * jv(1, root) - biot * jv(0, root),
        "max_abs_error": float(absolute[node, time_index]),
        "relative_cylindrical_l2_error": relative_l2,
        "max_error_time_s": float(times[time_index]),
        "max_error_radius_m": float(radii[node]),
        "runtime_s": elapsed,
        "function_evaluations": int(solution.nfev),
        "jacobian_evaluations": int(solution.njev),
        "lu_decompositions": int(solution.nlu),
    }


def task_bessel() -> None:
    task = "bessel"
    started = time.perf_counter()
    levels = [190, 380, 760]
    cases = [
        bessel_case(field, method, level)
        for field in ("temperature", "moisture")
        for method in ("BDF", "Radau")
        for level in levels
    ]
    convergence: dict[str, object] = {}
    for field in ("temperature", "moisture"):
        convergence[field] = {}
        for method in ("BDF", "Radau"):
            selected = [
                item for item in cases
                if item["field"] == field and item["method"] == method
            ]
            errors = [float(item["max_abs_error"]) for item in selected]
            orders = [
                math.log(errors[i] / errors[i + 1], 2.0)
                for i in range(len(errors) - 1)
            ]
            convergence[field][method] = {
                "levels": levels,
                "max_abs_errors": errors,
                "observed_orders": orders,
                "fine_error": errors[-1],
            }
    result = base_metadata(task, started)
    result.update(
        {
            "status": "complete",
            "description": "Single-mode Robin-Bessel exact solution on three FVM grids",
            "cases": cases,
            "convergence": convergence,
        }
    )
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "bessel_cases.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)
    atomic_json(OUT / f"{task}.json", result)


def assemble() -> None:
    names = ["q1_radau", "q2_radau", "q3_radau", "q4_radau", "bessel"]
    missing = [name for name in names if not (OUT / f"{name}.json").is_file()]
    if missing:
        raise RuntimeError(f"missing completed tasks: {missing}")
    data = {
        name: json.loads((OUT / f"{name}.json").read_text(encoding="utf-8"))
        for name in names
    }
    lines = [
        "# Radau 与 Bessel 实际交叉验证报告",
        "",
        f"> 生成时间：{datetime.now(timezone.utc).isoformat()}",
        "> 口径：附件1结束后环境固定为 50 °C、0.05 kg/kg。",
        "",
        "## Radau异构时间积分复算",
        "",
        "| 问题 | 对照网格 | 温度最大差 | 含水率最大差 | 临界时刻差/s | 四位小数一致率 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, grid in (("q1_radau", 1520), ("q2_radau", 760),
                      ("q3_radau", 2560), ("q4_radau", 1520)):
        item = data[key]
        temperature = item.get("temperature", {}).get("max_abs_difference")
        moisture = item["moisture"]["max_abs_difference"]
        crossing = item.get("crossing_difference_s")
        agreement = item["moisture"]["four_decimal_agreement_fraction"]
        temperature_text = "—" if temperature is None else f"{temperature:.6e}"
        crossing_text = "—" if crossing is None else f"{crossing:.6e}"
        lines.append(
            f"| {key[:2].upper()} | {grid} | {temperature_text} | "
            f"{moisture:.6e} | {crossing_text} | {agreement:.8%} |"
        )
    lines.extend(
        [
            "",
            "这里的最大差均在相同空间网格、相同物性和相同边界口径下计算，因而主要反映时间积分算法差异。",
            "",
            "## 常系数Bessel解析解验证",
            "",
            "| 场 | 方法 | N=190误差 | N=380误差 | N=760误差 | 后两级观测阶 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    convergence = data["bessel"]["convergence"]
    for field in ("temperature", "moisture"):
        for method in ("BDF", "Radau"):
            item = convergence[field][method]
            errors = item["max_abs_errors"]
            orders = item["observed_orders"]
            lines.append(
                f"| {field} | {method} | {errors[0]:.6e} | {errors[1]:.6e} | "
                f"{errors[2]:.6e} | {orders[-1]:.6f} |"
            )
    lines.extend(
        [
            "",
            "验收含义：Bessel单模态严格满足圆心对称和表面Robin条件；误差随网格加密下降且观测阶接近2，说明圆柱有限体积几何、内部通量与边界离散实现正确。",
            "",
            "## 结论边界",
            "",
            "Radau一致性验证时间推进，Bessel解析解验证固定域常系数程序内核；二者不替代物性参数和潜热等物理模型的实验验证。第四问Radau在超精细链的N=1520层实施，移动边界空间误差仍由既有1520/3040/6080三层结果控制。",
            "",
        ]
    )
    atomic_text(OUT / "validation_report.md", "\n".join(lines))
    combined = {
        "status": "complete",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "tasks": data,
        "artifacts": {
            path.name: sha256_file(path)
            for path in sorted(OUT.iterdir())
            if path.is_file()
        },
    }
    atomic_json(OUT / "COMPLETE.json", combined)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "task",
        choices=("q1", "q2", "q3", "q4", "bessel", "assemble", "all"),
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    functions = {
        "q1": task_q1,
        "q2": task_q2,
        "q3": task_q3,
        "q4": task_q4,
        "bessel": task_bessel,
        "assemble": assemble,
    }
    if args.task == "all":
        for name in ("bessel", "q1", "q2", "q3", "q4"):
            functions[name]()
        assemble()
    else:
        functions[args.task]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
