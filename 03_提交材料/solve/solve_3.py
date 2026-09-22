#!/usr/bin/env python3
"""Solve Problem A, Question 3 with a coupled radial finite-volume model.

Only the Python standard library is required.  The program reads Attachment 1
directly from XLSX, solves the fixed-radius variable-property heat/moisture
model until every radial node is below the prescribed moisture threshold, and
writes the requested result3.xlsx plus reproducibility and validation files.

Model
-----
    rho(C) cp(C) dT/dt = (1/r) d/dr [r k(C) dT/dr]
    dC/dt              = (1/r) d/dr [r D(C,T_K) dC/dr]

The centre is symmetric and the surface uses Robin heat/mass-transfer
conditions.  Attachment 1 is linearly interpolated through 4 h.  Afterwards,
the default constant-stage boundary is the mean over the final measured hour.

The moisture face flux is important: D changes by orders of magnitude near the
dry surface.  Instead of a harmonic mean of nodal D values, this code uses a
four-point Gauss-Legendre approximation to the Kirchhoff/secant mean

    D_bar = integral_0^1 D(C_L+s(C_R-C_L), T_face) ds.

This greatly reduces artificial surface resistance and grid dependence.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence
from zipfile import ZIP_DEFLATED, ZipFile


XLSX_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
OFFICE_REL_NS = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
)

# Four-point Gauss-Legendre rule on [-1, 1].
GAUSS_X = (
    -0.8611363115940526,
    -0.3399810435848563,
    0.3399810435848563,
    0.8611363115940526,
)
GAUSS_W = (
    0.3478548451374538,
    0.6521451548625461,
    0.6521451548625461,
    0.3478548451374538,
)


@dataclass(frozen=True)
class PhysicalParameters:
    radius_m: float = 0.02
    initial_temperature_c: float = 28.0
    initial_moisture_kg_kg: float = 2.55
    heat_transfer_w_m2_k: float = 25.0
    mass_transfer_m_s: float = 8.0e-7
    moisture_threshold_kg_kg: float = 0.15

    def density(self, moisture: float) -> float:
        self._check_moisture(moisture)
        return 650.0 + 128.0 * moisture

    def heat_capacity(self, moisture: float) -> float:
        self._check_moisture(moisture)
        return 1450.0 + 2736.0 * moisture / (moisture + 1.0)

    def conductivity(self, moisture: float) -> float:
        self._check_moisture(moisture)
        return 0.21 + 0.38 * moisture / (moisture + 1.0)

    def diffusivity(self, moisture: float, temperature_k: float) -> float:
        self._check_moisture(moisture)
        if not math.isfinite(temperature_k) or temperature_k <= 0.0:
            raise ValueError(f"temperature must be positive K, got {temperature_k!r}")
        return (
            2.4e-3
            * math.exp(-0.45 / moisture)
            * math.exp(-3850.0 / temperature_k)
        )

    @staticmethod
    def _check_moisture(moisture: float) -> None:
        if not math.isfinite(moisture) or moisture <= 0.0:
            raise ValueError(f"moisture must be positive and finite, got {moisture!r}")


@dataclass(frozen=True)
class NumericalConfig:
    radial_cells: int = 160
    time_step_s: float = 5.0
    output_interval_s: float = 60.0
    maximum_time_h: float = 96.0
    plateau_start_h: float = 3.0
    environment_extension: str = "last_hour_mean"
    fixed_environment_temperature_c: float = 50.0
    fixed_environment_moisture_kg_kg: float = 0.05
    picard_tolerance: float = 1.0e-10
    picard_max_iterations: int = 40

    def validate(self) -> None:
        if self.radial_cells < 20 or self.radial_cells % 20 != 0:
            raise ValueError("radial_cells must be a multiple of 20 and at least 20")
        if not math.isfinite(self.time_step_s) or self.time_step_s <= 0.0:
            raise ValueError("time_step_s must be positive and finite")
        if not math.isfinite(self.output_interval_s) or self.output_interval_s <= 0.0:
            raise ValueError("output_interval_s must be positive and finite")
        stride = self.output_interval_s / self.time_step_s
        if abs(stride - round(stride)) > 1.0e-12:
            raise ValueError("output_interval_s must be an integer multiple of time_step_s")
        if self.maximum_time_h <= 4.0:
            raise ValueError("maximum_time_h must exceed the 4 h measured period")
        if not 0.0 <= self.plateau_start_h < 4.0:
            raise ValueError("plateau_start_h must lie in [0, 4)")
        if self.environment_extension not in {"last_hour_mean", "last_point", "fixed"}:
            raise ValueError("unknown environment extension strategy")
        if not math.isfinite(self.fixed_environment_temperature_c):
            raise ValueError("fixed environment temperature must be finite")
        if (
            not math.isfinite(self.fixed_environment_moisture_kg_kg)
            or self.fixed_environment_moisture_kg_kg <= 0.0
        ):
            raise ValueError("fixed environment moisture must be positive and finite")
        if self.picard_tolerance <= 0.0 or self.picard_max_iterations < 1:
            raise ValueError("invalid Picard settings")


@dataclass
class EnvironmentData:
    times_s: list[float]
    temperatures_c: list[float]
    moistures_kg_kg: list[float]
    plateau_start_s: float
    extension_strategy: str
    plateau_temperature_c: float
    plateau_moisture_kg_kg: float

    def validate(self) -> None:
        count = len(self.times_s)
        if count < 2 or len(self.temperatures_c) != count or len(self.moistures_kg_kg) != count:
            raise ValueError("environment columns have inconsistent lengths")
        if self.times_s[0] != 0.0:
            raise ValueError("environment data must start at t=0")
        if any(b <= a for a, b in zip(self.times_s, self.times_s[1:])):
            raise ValueError("environment times must be strictly increasing")
        if any(not math.isfinite(value) for value in self.temperatures_c):
            raise ValueError("non-finite environment temperature")
        if any(not math.isfinite(value) or value <= 0.0 for value in self.moistures_kg_kg):
            raise ValueError("environment moisture must be positive and finite")
        if not self.times_s[0] <= self.plateau_start_s <= self.times_s[-1]:
            raise ValueError("plateau start lies outside measured data")

    def value(self, t_s: float) -> tuple[float, float]:
        """Linearly interpolate measured data, then apply constant extension."""
        if t_s <= self.times_s[0]:
            return self.temperatures_c[0], self.moistures_kg_kg[0]
        if t_s >= self.times_s[-1]:
            return self.plateau_temperature_c, self.plateau_moisture_kg_kg
        j = bisect.bisect_right(self.times_s, t_s) - 1
        t0, t1 = self.times_s[j], self.times_s[j + 1]
        weight = (t_s - t0) / (t1 - t0)
        temperature = self.temperatures_c[j] + weight * (
            self.temperatures_c[j + 1] - self.temperatures_c[j]
        )
        moisture = self.moistures_kg_kg[j] + weight * (
            self.moistures_kg_kg[j + 1] - self.moistures_kg_kg[j]
        )
        return temperature, moisture


@dataclass
class SolverDiagnostics:
    steps: int = 0
    max_picard_iterations: int = 0
    total_picard_iterations: int = 0
    final_picard_update: float = 0.0
    max_heat_balance_abs: float = 0.0
    max_heat_balance_rel: float = 0.0
    max_moisture_balance_abs: float = 0.0
    max_moisture_balance_rel: float = 0.0
    max_radial_monotonicity_violation: float = 0.0
    min_temperature_c: float = math.inf
    max_temperature_c: float = -math.inf
    min_moisture_kg_kg: float = math.inf
    max_moisture_kg_kg: float = -math.inf

    @property
    def mean_picard_iterations(self) -> float:
        return 0.0 if self.steps == 0 else self.total_picard_iterations / self.steps


@dataclass
class SimulationResult:
    times_s: list[float]
    radii_cm: list[float]
    moistures_kg_kg: list[list[float]]
    drying_time_s: float
    drying_moistures_kg_kg: list[float]
    diagnostics: SolverDiagnostics
    runtime_s: float
    config: NumericalConfig


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def excel_column_index(reference: str) -> int:
    value = 0
    for character in reference:
        if not character.isalpha():
            break
        value = value * 26 + ord(character.upper()) - ord("A") + 1
    return value - 1


def excel_column_name(index_zero_based: int) -> str:
    value = index_zero_based + 1
    result: list[str] = []
    while value:
        value, remainder = divmod(value - 1, 26)
        result.append(chr(ord("A") + remainder))
    return "".join(reversed(result))


def read_first_worksheet(path: Path) -> list[list[object]]:
    """Read cell values from the first XLSX worksheet using stdlib only."""
    with ZipFile(path) as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall(f"{{{XLSX_NS}}}si"):
                shared_strings.append(
                    "".join(node.text or "" for node in item.iter(f"{{{XLSX_NS}}}t"))
                )

        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        sheet = workbook.find(f".//{{{XLSX_NS}}}sheet")
        if sheet is None:
            raise ValueError(f"workbook contains no worksheet: {path}")
        relation_id = sheet.attrib[f"{{{OFFICE_REL_NS}}}id"]
        relations = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        relation_map = {item.attrib["Id"]: item.attrib["Target"] for item in relations}
        target = relation_map[relation_id].lstrip("/")
        if not target.startswith("xl/"):
            target = f"xl/{target}"

        root = ET.fromstring(archive.read(target))
        rows: list[list[object]] = []
        for row in root.findall(f".//{{{XLSX_NS}}}sheetData/{{{XLSX_NS}}}row"):
            by_column: dict[int, object] = {}
            for cell in row.findall(f"{{{XLSX_NS}}}c"):
                column = excel_column_index(cell.attrib["r"])
                cell_type = cell.attrib.get("t")
                value_node = cell.find(f"{{{XLSX_NS}}}v")
                if cell_type == "inlineStr":
                    inline = cell.find(f"{{{XLSX_NS}}}is")
                    value: object = "" if inline is None else "".join(
                        node.text or "" for node in inline.iter(f"{{{XLSX_NS}}}t")
                    )
                elif value_node is None or value_node.text is None:
                    value = ""
                elif cell_type == "s":
                    value = shared_strings[int(value_node.text)]
                else:
                    value = float(value_node.text)
                by_column[column] = value
            if by_column:
                rows.append([by_column.get(i, "") for i in range(max(by_column) + 1)])
        return rows


def load_environment(
    path: Path,
    plateau_start_h: float,
    extension_strategy: str,
    fixed_temperature_c: float = 50.0,
    fixed_moisture_kg_kg: float = 0.05,
) -> EnvironmentData:
    rows = read_first_worksheet(path)
    expected = ["时间", "温度", "水分浓度"]
    if not rows or [str(value).strip() for value in rows[0][:3]] != expected:
        raise ValueError(f"unexpected Attachment 1 headers; expected {expected!r}")

    times: list[float] = []
    temperatures: list[float] = []
    moistures: list[float] = []
    for row_number, row in enumerate(rows[1:], start=2):
        if len(row) < 3 or any(value == "" for value in row[:3]):
            raise ValueError(f"missing Attachment 1 value at row {row_number}")
        t_s, temperature_c, moisture = (float(value) for value in row[:3])
        times.append(t_s)
        temperatures.append(temperature_c)
        moistures.append(moisture)

    plateau_start_s = plateau_start_h * 3600.0
    if extension_strategy == "last_hour_mean":
        indices = [i for i, t_s in enumerate(times) if t_s >= plateau_start_s]
        if not indices:
            raise ValueError("no measured data in plateau interval")
        plateau_temperature = sum(temperatures[i] for i in indices) / len(indices)
        plateau_moisture = sum(moistures[i] for i in indices) / len(indices)
    elif extension_strategy == "last_point":
        plateau_temperature = temperatures[-1]
        plateau_moisture = moistures[-1]
    elif extension_strategy == "fixed":
        if not math.isfinite(fixed_temperature_c):
            raise ValueError("fixed environment temperature must be finite")
        if not math.isfinite(fixed_moisture_kg_kg) or fixed_moisture_kg_kg <= 0.0:
            raise ValueError("fixed environment moisture must be positive and finite")
        plateau_temperature = fixed_temperature_c
        plateau_moisture = fixed_moisture_kg_kg
    else:
        raise ValueError(f"unknown environment extension: {extension_strategy}")

    environment = EnvironmentData(
        times_s=times,
        temperatures_c=temperatures,
        moistures_kg_kg=moistures,
        plateau_start_s=plateau_start_s,
        extension_strategy=extension_strategy,
        plateau_temperature_c=plateau_temperature,
        plateau_moisture_kg_kg=plateau_moisture,
    )
    environment.validate()
    return environment


def solve_tridiagonal(
    lower: Sequence[float],
    diagonal: Sequence[float],
    upper: Sequence[float],
    rhs: Sequence[float],
) -> list[float]:
    """Solve a tridiagonal system with the Thomas algorithm."""
    n = len(diagonal)
    if n == 0 or len(rhs) != n or len(lower) != n - 1 or len(upper) != n - 1:
        raise ValueError("invalid tridiagonal dimensions")
    modified_upper = [0.0] * max(0, n - 1)
    modified_rhs = [0.0] * n
    pivot = diagonal[0]
    if not math.isfinite(pivot) or abs(pivot) < 1.0e-300:
        raise ArithmeticError("invalid tridiagonal pivot at row 0")
    if n > 1:
        modified_upper[0] = upper[0] / pivot
    modified_rhs[0] = rhs[0] / pivot
    for i in range(1, n):
        pivot = diagonal[i] - lower[i - 1] * modified_upper[i - 1]
        if not math.isfinite(pivot) or abs(pivot) < 1.0e-300:
            raise ArithmeticError(f"invalid tridiagonal pivot at row {i}")
        if i < n - 1:
            modified_upper[i] = upper[i] / pivot
        modified_rhs[i] = (rhs[i] - lower[i - 1] * modified_rhs[i - 1]) / pivot
    result = [0.0] * n
    result[-1] = modified_rhs[-1]
    for i in range(n - 2, -1, -1):
        result[i] = modified_rhs[i] - modified_upper[i] * result[i + 1]
    return result


def make_uniform_grid(radius_m: float, radial_cells: int) -> list[float]:
    return [radius_m * i / radial_cells for i in range(radial_cells + 1)]


def radial_storage_weights(radii_m: Sequence[float]) -> list[float]:
    n = len(radii_m) - 1
    radius = radii_m[-1]
    weights = [0.0] * (n + 1)
    first_face = 0.5 * (radii_m[0] + radii_m[1])
    weights[0] = first_face**2 / 2.0
    for i in range(1, n):
        west = 0.5 * (radii_m[i - 1] + radii_m[i])
        east = 0.5 * (radii_m[i] + radii_m[i + 1])
        weights[i] = (east**2 - west**2) / 2.0
    last_face = 0.5 * (radii_m[-2] + radius)
    weights[-1] = (radius**2 - last_face**2) / 2.0
    return weights


def time_coefficients(
    old: Sequence[float], previous: Sequence[float] | None, dt_s: float
) -> tuple[float, list[float]]:
    """Return BE coefficients for step one and BDF2 thereafter."""
    if previous is None:
        return 1.0 / dt_s, [value / dt_s for value in old]
    factor = 1.0 / (2.0 * dt_s)
    return 3.0 * factor, [
        (4.0 * current - prior) * factor
        for current, prior in zip(old, previous)
    ]


def implicit_radial_step(
    old: Sequence[float],
    previous: Sequence[float] | None,
    dt_s: float,
    radii_m: Sequence[float],
    storage_nodes: Sequence[float],
    gamma_faces: Sequence[float],
    boundary_transfer: float,
    external_value: float,
) -> list[float]:
    """One conservative BE/BDF2 radial finite-volume step."""
    node_count = len(old)
    radial_cells = node_count - 1
    if node_count < 2:
        raise ValueError("at least two radial nodes are required")
    if previous is not None and len(previous) != node_count:
        raise ValueError("invalid history arrays")
    if len(radii_m) != node_count or len(storage_nodes) != node_count:
        raise ValueError("invalid nodal arrays")
    if len(gamma_faces) != radial_cells:
        raise ValueError("invalid face coefficient array")
    if any(value <= 0.0 or not math.isfinite(value) for value in storage_nodes):
        raise ValueError("storage coefficients must be positive")
    if any(value <= 0.0 or not math.isfinite(value) for value in gamma_faces):
        raise ValueError("face coefficients must be positive")

    time_diagonal, history = time_coefficients(old, previous, dt_s)
    lower = [0.0] * radial_cells
    diagonal = [0.0] * node_count
    upper = [0.0] * radial_cells
    rhs = [0.0] * node_count

    east = 0.5 * (radii_m[0] + radii_m[1])
    east_conductance = east * gamma_faces[0] / (radii_m[1] - radii_m[0])
    centre_weight = east**2 / 2.0
    diagonal[0] = storage_nodes[0] * centre_weight * time_diagonal + east_conductance
    upper[0] = -east_conductance
    rhs[0] = storage_nodes[0] * centre_weight * history[0]

    for i in range(1, radial_cells):
        west = 0.5 * (radii_m[i - 1] + radii_m[i])
        east = 0.5 * (radii_m[i] + radii_m[i + 1])
        west_conductance = west * gamma_faces[i - 1] / (radii_m[i] - radii_m[i - 1])
        east_conductance = east * gamma_faces[i] / (radii_m[i + 1] - radii_m[i])
        weight = (east**2 - west**2) / 2.0
        lower[i - 1] = -west_conductance
        diagonal[i] = (
            storage_nodes[i] * weight * time_diagonal
            + west_conductance
            + east_conductance
        )
        upper[i] = -east_conductance
        rhs[i] = storage_nodes[i] * weight * history[i]

    radius = radii_m[-1]
    west = 0.5 * (radii_m[-2] + radius)
    west_conductance = west * gamma_faces[-1] / (radius - radii_m[-2])
    boundary_conductance = radius * boundary_transfer
    surface_weight = (radius**2 - west**2) / 2.0
    lower[-1] = -west_conductance
    diagonal[-1] = (
        storage_nodes[-1] * surface_weight * time_diagonal
        + west_conductance
        + boundary_conductance
    )
    rhs[-1] = (
        storage_nodes[-1] * surface_weight * history[-1]
        + boundary_conductance * external_value
    )
    return solve_tridiagonal(lower, diagonal, upper, rhs)


def conductivity_faces(
    moisture: Sequence[float], parameters: PhysicalParameters
) -> list[float]:
    return [
        parameters.conductivity(0.5 * (left + right))
        for left, right in zip(moisture, moisture[1:])
    ]


def integrated_face_diffusivity(
    left_moisture: float,
    right_moisture: float,
    face_temperature_k: float,
    parameters: PhysicalParameters,
) -> float:
    """Kirchhoff/secant mean of D across one moisture face."""
    midpoint = 0.5 * (left_moisture + right_moisture)
    half_span = 0.5 * (right_moisture - left_moisture)
    # The temperature-dependent factor is constant across this face integral;
    # evaluate it once instead of four times.  This matters for the N=160,
    # dt=5 s production experiment, which evaluates millions of face fluxes.
    thermal_factor = 2.4e-3 * math.exp(-3850.0 / face_temperature_k)
    total = 0.0
    for point, weight in zip(GAUSS_X, GAUSS_W):
        moisture = midpoint + half_span * point
        parameters._check_moisture(moisture)
        total += weight * math.exp(-0.45 / moisture)
    return 0.5 * thermal_factor * total


def diffusivity_faces(
    moisture: Sequence[float],
    temperature_c: Sequence[float],
    parameters: PhysicalParameters,
) -> list[float]:
    result = []
    for c_left, c_right, t_left, t_right in zip(
        moisture, moisture[1:], temperature_c, temperature_c[1:]
    ):
        face_temperature_k = 0.5 * (t_left + t_right) + 273.15
        result.append(
            integrated_face_diffusivity(
                c_left, c_right, face_temperature_k, parameters
            )
        )
    return result


def requested_node_indices(radial_cells: int) -> tuple[list[float], list[int]]:
    radii_cm = [0.1 * i for i in range(21)]
    stride = radial_cells // 20
    return radii_cm, [i * stride for i in range(21)]


def interpolate_vector(
    old: Sequence[float], new: Sequence[float], fraction: float
) -> list[float]:
    return [a + fraction * (b - a) for a, b in zip(old, new)]


def balance_residual(
    new: Sequence[float],
    old: Sequence[float],
    previous: Sequence[float] | None,
    dt_s: float,
    weights: Sequence[float],
    storage_nodes: Sequence[float],
    radius_m: float,
    boundary_transfer: float,
    external_value: float,
    relative_scale_floor: float = 1.0e-10,
) -> tuple[float, float]:
    time_diagonal, history = time_coefficients(old, previous, dt_s)
    storage_rate = sum(
        storage * weight * (time_diagonal * value - hist)
        for storage, weight, value, hist in zip(
            storage_nodes, weights, new, history
        )
    )
    boundary_rate = radius_m * boundary_transfer * (external_value - new[-1])
    absolute = abs(storage_rate - boundary_rate)
    # A relative residual is not informative after the field has equilibrated:
    # both sides then approach roundoff and their quotient can be O(1) despite
    # a tiny absolute residual.  Retain the absolute residual in all cases and
    # evaluate the relative metric only while the balance has a useful scale.
    scale = max(abs(storage_rate), abs(boundary_rate))
    relative = 0.0 if scale < relative_scale_floor else absolute / scale
    return absolute, relative


def solve_model(
    environment: EnvironmentData,
    parameters: PhysicalParameters,
    config: NumericalConfig,
    progress: bool = False,
) -> SimulationResult:
    config.validate()
    environment.validate()
    started = time.perf_counter()
    radii_m = make_uniform_grid(parameters.radius_m, config.radial_cells)
    weights = radial_storage_weights(radii_m)
    output_radii_cm, output_indices = requested_node_indices(config.radial_cells)
    output_stride = int(round(config.output_interval_s / config.time_step_s))
    maximum_steps = int(math.ceil(config.maximum_time_h * 3600.0 / config.time_step_s))

    temperature = [parameters.initial_temperature_c] * (config.radial_cells + 1)
    moisture = [parameters.initial_moisture_kg_kg] * (config.radial_cells + 1)
    previous_temperature: list[float] | None = None
    previous_moisture: list[float] | None = None
    previous_gap = max(moisture) - parameters.moisture_threshold_kg_kg

    output_times: list[float] = []
    output_moistures: list[list[float]] = []
    diagnostics = SolverDiagnostics()
    next_progress_s = 6.0 * 3600.0

    for step in range(1, maximum_steps + 1):
        t_new = step * config.time_step_s
        external_temperature, external_moisture = environment.value(t_new)
        guess_temperature = temperature.copy()
        guess_moisture = moisture.copy()

        for iteration in range(1, config.picard_max_iterations + 1):
            heat_storage = [
                parameters.density(value) * parameters.heat_capacity(value)
                for value in guess_moisture
            ]
            heat_faces = conductivity_faces(guess_moisture, parameters)
            candidate_temperature = implicit_radial_step(
                temperature,
                previous_temperature,
                config.time_step_s,
                radii_m,
                heat_storage,
                heat_faces,
                parameters.heat_transfer_w_m2_k,
                external_temperature,
            )
            moisture_faces = diffusivity_faces(
                guess_moisture, candidate_temperature, parameters
            )
            candidate_moisture = implicit_radial_step(
                moisture,
                previous_moisture,
                config.time_step_s,
                radii_m,
                [1.0] * len(radii_m),
                moisture_faces,
                parameters.mass_transfer_m_s,
                external_moisture,
            )
            temperature_update = max(
                abs(a - b)
                for a, b in zip(candidate_temperature, guess_temperature)
            )
            moisture_update = max(
                abs(a - b) for a, b in zip(candidate_moisture, guess_moisture)
            )
            combined_update = max(temperature_update / 50.0, moisture_update)
            guess_temperature = candidate_temperature
            guess_moisture = candidate_moisture
            if combined_update < config.picard_tolerance:
                break
        else:
            raise RuntimeError(
                f"Picard iteration failed at t={t_new:.3f} s; "
                f"last scaled update={combined_update:.3e}"
            )

        new_temperature = guess_temperature
        new_moisture = guess_moisture
        final_heat_storage = [
            parameters.density(value) * parameters.heat_capacity(value)
            for value in new_moisture
        ]
        heat_abs, heat_rel = balance_residual(
            new_temperature,
            temperature,
            previous_temperature,
            config.time_step_s,
            weights,
            final_heat_storage,
            parameters.radius_m,
            parameters.heat_transfer_w_m2_k,
            external_temperature,
            relative_scale_floor=1.0e-8,
        )
        moisture_abs, moisture_rel = balance_residual(
            new_moisture,
            moisture,
            previous_moisture,
            config.time_step_s,
            weights,
            [1.0] * len(radii_m),
            parameters.radius_m,
            parameters.mass_transfer_m_s,
            external_moisture,
            relative_scale_floor=1.0e-14,
        )

        diagnostics.steps += 1
        diagnostics.total_picard_iterations += iteration
        diagnostics.max_picard_iterations = max(
            diagnostics.max_picard_iterations, iteration
        )
        diagnostics.final_picard_update = combined_update
        diagnostics.max_heat_balance_abs = max(diagnostics.max_heat_balance_abs, heat_abs)
        diagnostics.max_heat_balance_rel = max(diagnostics.max_heat_balance_rel, heat_rel)
        diagnostics.max_moisture_balance_abs = max(
            diagnostics.max_moisture_balance_abs, moisture_abs
        )
        diagnostics.max_moisture_balance_rel = max(
            diagnostics.max_moisture_balance_rel, moisture_rel
        )
        diagnostics.max_radial_monotonicity_violation = max(
            diagnostics.max_radial_monotonicity_violation,
            max(
                (outer - inner for inner, outer in zip(new_moisture, new_moisture[1:])),
                default=0.0,
            ),
        )
        diagnostics.min_temperature_c = min(
            diagnostics.min_temperature_c, min(new_temperature)
        )
        diagnostics.max_temperature_c = max(
            diagnostics.max_temperature_c, max(new_temperature)
        )
        diagnostics.min_moisture_kg_kg = min(
            diagnostics.min_moisture_kg_kg, min(new_moisture)
        )
        diagnostics.max_moisture_kg_kg = max(
            diagnostics.max_moisture_kg_kg, max(new_moisture)
        )

        new_gap = max(new_moisture) - parameters.moisture_threshold_kg_kg
        if new_gap < 0.0 <= previous_gap:
            fraction = previous_gap / (previous_gap - new_gap)
            fraction = min(1.0, max(0.0, fraction))
            drying_time_s = t_new - config.time_step_s + fraction * config.time_step_s
            drying_state = interpolate_vector(moisture, new_moisture, fraction)
            requested_drying_state = [drying_state[i] for i in output_indices]
            if not output_times or abs(output_times[-1] - drying_time_s) > 1.0e-9:
                output_times.append(drying_time_s)
                output_moistures.append(requested_drying_state)
            return SimulationResult(
                times_s=output_times,
                radii_cm=output_radii_cm,
                moistures_kg_kg=output_moistures,
                drying_time_s=drying_time_s,
                drying_moistures_kg_kg=requested_drying_state,
                diagnostics=diagnostics,
                runtime_s=time.perf_counter() - started,
                config=config,
            )

        if step % output_stride == 0:
            output_times.append(t_new)
            output_moistures.append([new_moisture[i] for i in output_indices])

        previous_temperature, temperature = temperature, new_temperature
        previous_moisture, moisture = moisture, new_moisture
        previous_gap = new_gap

        if progress and t_new + 1.0e-9 >= next_progress_s:
            print(
                f"  t={t_new / 3600.0:7.2f} h  "
                f"Cmax={max(moisture):.8f}  Csurf={moisture[-1]:.8f}  "
                f"Picard={iteration}",
                flush=True,
            )
            next_progress_s += 6.0 * 3600.0

    raise RuntimeError(
        f"drying criterion was not reached before {config.maximum_time_h:g} h; "
        f"last maximum moisture={max(moisture):.9g}"
    )


def find_output_row(result: SimulationResult, time_s: float) -> list[float]:
    index = bisect.bisect_left(result.times_s, time_s)
    candidates = [i for i in (index - 1, index) if 0 <= i < len(result.times_s)]
    if not candidates:
        raise KeyError(time_s)
    nearest = min(candidates, key=lambda i: abs(result.times_s[i] - time_s))
    if abs(result.times_s[nearest] - time_s) > 1.0e-7:
        raise KeyError(f"time {time_s} s is not an output time")
    return result.moistures_kg_kg[nearest]


def table5_rows(result: SimulationResult) -> list[tuple[float, list[float]]]:
    radius_indices = [0, 5, 10, 15, 20]
    rows: list[tuple[float, list[float]]] = []
    time_s = 6.0 * 3600.0
    while time_s < result.drying_time_s - 1.0e-7:
        values = find_output_row(result, time_s)
        rows.append((time_s / 3600.0, [values[i] for i in radius_indices]))
        time_s += 6.0 * 3600.0
    rows.append(
        (
            result.drying_time_s / 3600.0,
            [result.drying_moistures_kg_kg[i] for i in radius_indices],
        )
    )
    return rows


def convergence_comparison(
    coarse: SimulationResult, fine: SimulationResult
) -> dict[str, float]:
    common_end_h = min(coarse.drying_time_s, fine.drying_time_s) / 3600.0
    comparison_times_h = [
        6.0 * i for i in range(1, int(math.floor(common_end_h / 6.0)) + 1)
    ]
    max_difference = 0.0
    for time_h in comparison_times_h:
        coarse_row = find_output_row(coarse, time_h * 3600.0)
        fine_row = find_output_row(fine, time_h * 3600.0)
        max_difference = max(
            max_difference,
            max(abs(a - b) for a, b in zip(coarse_row, fine_row)),
        )
    return {
        "coarse_radial_cells": coarse.config.radial_cells,
        "coarse_time_step_s": coarse.config.time_step_s,
        "fine_radial_cells": fine.config.radial_cells,
        "fine_time_step_s": fine.config.time_step_s,
        "coarse_drying_time_h": coarse.drying_time_s / 3600.0,
        "fine_drying_time_h": fine.drying_time_s / 3600.0,
        "drying_time_difference_h": abs(
            coarse.drying_time_s - fine.drying_time_s
        ) / 3600.0,
        "max_common_6h_profile_difference": max_difference,
    }


def write_full_precision_csv(path: Path, result: SimulationResult) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["时间/s"] + [f"{radius:.1f} cm" for radius in result.radii_cm])
        for time_s, values in zip(result.times_s, result.moistures_kg_kg):
            writer.writerow(
                [f"{time_s:.9f}".rstrip("0").rstrip(".")]
                + [f"{value:.12f}" for value in values]
            )


def write_table5_csv(path: Path, result: SimulationResult) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["时间/h", "0 cm", "0.5 cm", "1.0 cm", "1.5 cm", "2.0 cm"])
        for time_h, values in table5_rows(result):
            writer.writerow([f"{time_h:.4f}"] + [f"{value:.4f}" for value in values])


def write_table5_markdown(path: Path, result: SimulationResult) -> None:
    lines = [
        "# 表 5 药材烘干过程的水分浓度",
        "",
        "| 时间/h | 0 cm | 0.5 cm | 1.0 cm | 1.5 cm | 2.0 cm |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    rows = table5_rows(result)
    for index, (time_h, values) in enumerate(rows):
        label = f"{time_h:.4f}" if index == len(rows) - 1 else f"{time_h:.0f}"
        lines.append("| " + label + " | " + " | ".join(f"{v:.4f}" for v in values) + " |")
    lines.extend(
        [
            "",
            f"连续事件插值得到的烘干结束时间为 `{result.drying_time_s / 3600.0:.6f} h`。",
            "终止事件使用未舍入含水率判断；表格数值最后才保留四位小数。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_environment_csv(path: Path, environment: EnvironmentData) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["时间/s", "温度/°C", "水分浓度/(kg/kg)"])
        for row in zip(
            environment.times_s,
            environment.temperatures_c,
            environment.moistures_kg_kg,
        ):
            writer.writerow([f"{row[0]:.0f}", f"{row[1]:.15g}", f"{row[2]:.15g}"])


def add_four_decimal_style(styles_xml: bytes) -> tuple[bytes, int]:
    ET.register_namespace("", XLSX_NS)
    ET.register_namespace("x14", "http://schemas.microsoft.com/office/spreadsheetml/2009/9/main")
    ET.register_namespace("x15", "http://schemas.microsoft.com/office/spreadsheetml/2010/11/main")
    root = ET.fromstring(styles_xml)
    number_formats = root.find(f"{{{XLSX_NS}}}numFmts")
    if number_formats is None:
        number_formats = ET.Element(f"{{{XLSX_NS}}}numFmts", {"count": "0"})
        fonts = root.find(f"{{{XLSX_NS}}}fonts")
        root.insert(0 if fonts is None else list(root).index(fonts), number_formats)
    used_ids = {
        int(node.attrib["numFmtId"])
        for node in number_formats.findall(f"{{{XLSX_NS}}}numFmt")
    }
    format_id = 164
    while format_id in used_ids:
        format_id += 1
    ET.SubElement(
        number_formats,
        f"{{{XLSX_NS}}}numFmt",
        {"numFmtId": str(format_id), "formatCode": "0.0000"},
    )
    number_formats.set(
        "count", str(len(number_formats.findall(f"{{{XLSX_NS}}}numFmt")))
    )
    cell_formats = root.find(f"{{{XLSX_NS}}}cellXfs")
    if cell_formats is None:
        raise ValueError("template has no cellXfs")
    style_id = len(cell_formats.findall(f"{{{XLSX_NS}}}xf"))
    style = ET.SubElement(
        cell_formats,
        f"{{{XLSX_NS}}}xf",
        {
            "numFmtId": str(format_id),
            "fontId": "2",
            "fillId": "0",
            "borderId": "0",
            "xfId": "0",
            "applyNumberFormat": "1",
            "applyFont": "1",
            "applyAlignment": "1",
        },
    )
    ET.SubElement(
        style,
        f"{{{XLSX_NS}}}alignment",
        {"horizontal": "center", "vertical": "center"},
    )
    cell_formats.set("count", str(style_id + 1))
    return ET.tostring(root, encoding="utf-8", xml_declaration=True), style_id


def make_result3_worksheet_xml(
    result: SimulationResult, value_style_id: int
) -> bytes:
    last_column = excel_column_name(len(result.radii_cm))
    last_row = len(result.times_s) + 1
    pieces = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        f'<worksheet xmlns="{XLSX_NS}" xmlns:r="{OFFICE_REL_NS}">',
        f'<dimension ref="A1:{last_column}{last_row}"/>',
        '<sheetViews><sheetView tabSelected="1" workbookViewId="0">',
        '<pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>',
        '</sheetView></sheetViews>',
        '<sheetFormatPr defaultRowHeight="15"/>',
        f'<cols><col min="1" max="1" width="22" customWidth="1"/>'
        f'<col min="2" max="{len(result.radii_cm) + 1}" width="12" customWidth="1"/></cols>',
        '<sheetData>',
        f'<row r="1" spans="1:{len(result.radii_cm) + 1}">',
        '<c r="A1" s="2" t="inlineStr"><is><t>时间\\到药材中心的距离/cm</t></is></c>',
    ]
    for column_index, radius_cm in enumerate(result.radii_cm, start=1):
        column = excel_column_name(column_index)
        pieces.append(f'<c r="{column}1" s="2"><v>{radius_cm:.1f}</v></c>')
    pieces.append("</row>")
    for row_index, (time_s, values) in enumerate(
        zip(result.times_s, result.moistures_kg_kg), start=2
    ):
        pieces.append(f'<row r="{row_index}" spans="1:{len(result.radii_cm) + 1}">')
        if abs(time_s - round(time_s)) < 1.0e-9:
            time_text = str(int(round(time_s)))
        else:
            time_text = f"{time_s:.4f}"
        pieces.append(f'<c r="A{row_index}" s="1"><v>{time_text}</v></c>')
        for column_index, value in enumerate(values, start=1):
            if not math.isfinite(value):
                raise ValueError("cannot write non-finite XLSX value")
            column = excel_column_name(column_index)
            pieces.append(
                f'<c r="{column}{row_index}" s="{value_style_id}">'
                f'<v>{value:.4f}</v></c>'
            )
        pieces.append("</row>")
    pieces.extend(
        [
            "</sheetData>",
            f'<autoFilter ref="A1:{last_column}{last_row}"/>',
            '<pageMargins left="0.75" right="0.75" top="1" bottom="1" '
            'header="0.5" footer="0.5"/>',
            "</worksheet>",
        ]
    )
    return "".join(pieces).encode("utf-8")


def write_result3_xlsx(template_path: Path, output_path: Path, result: SimulationResult) -> None:
    with ZipFile(template_path, "r") as source:
        styles_xml, value_style_id = add_four_decimal_style(source.read("xl/styles.xml"))
        replacements = {
            "xl/styles.xml": styles_xml,
            "xl/worksheets/sheet1.xml": make_result3_worksheet_xml(
                result, value_style_id
            ),
        }
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=9) as target:
            for item in source.infolist():
                target.writestr(item, replacements.get(item.filename, source.read(item.filename)))
        os.replace(temporary, output_path)


def validate_result3_xlsx(path: Path, result: SimulationResult) -> dict[str, object]:
    with ZipFile(path) as archive:
        corrupt_member = archive.testzip()
        if corrupt_member is not None:
            raise ValueError(f"corrupt XLSX member: {corrupt_member}")
        root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        dimension = root.find(f"{{{XLSX_NS}}}dimension")
        rows = root.findall(f".//{{{XLSX_NS}}}sheetData/{{{XLSX_NS}}}row")
        if len(rows) != len(result.times_s) + 1:
            raise ValueError("unexpected result3.xlsx row count")
        if len(rows[-1].findall(f"{{{XLSX_NS}}}c")) != len(result.radii_cm) + 1:
            raise ValueError("unexpected result3.xlsx final-row width")
        return {
            "zip_integrity": "ok",
            "dimension": None if dimension is None else dimension.attrib.get("ref"),
            "rows_including_header": len(rows),
            "columns": len(result.radii_cm) + 1,
        }


def write_convergence_csv(
    path: Path, results: Sequence[SimulationResult]
) -> list[dict[str, float]]:
    comparisons = [
        convergence_comparison(coarse, fine)
        for coarse, fine in zip(results, results[1:])
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = list(comparisons[0]) if comparisons else []
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(comparisons)
    return comparisons


def make_unique_experiment_directory(
    output_root: Path,
    experiment_name: str | None,
    config: NumericalConfig,
) -> Path:
    """Reserve a new directory so a completed experiment is never overwritten."""
    if experiment_name:
        stem = experiment_name
    else:
        timestamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
        dt_token = f"{config.time_step_s:g}".replace(".", "p")
        stem = (
            f"{timestamp}_N{config.radial_cells}_dt{dt_token}s_"
            f"{config.environment_extension}"
        )
    candidate = output_root / stem
    suffix = 2
    while candidate.exists():
        candidate = output_root / f"{stem}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def append_experiment_index(
    index_path: Path,
    experiment_dir: Path,
    result: SimulationResult,
    environment: EnvironmentData,
    result_xlsx: Path,
) -> None:
    """Append a compact, durable record of one successfully completed run."""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not index_path.exists() or index_path.stat().st_size == 0
    with index_path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        if write_header:
            writer.writerow(
                [
                    "completed_at",
                    "experiment_directory",
                    "radial_cells",
                    "radial_step_cm",
                    "time_step_s",
                    "environment_extension",
                    "plateau_temperature_c",
                    "plateau_moisture_kg_kg",
                    "drying_time_h",
                    "result3_sha256",
                ]
            )
        writer.writerow(
            [
                datetime.now().astimezone().isoformat(),
                str(experiment_dir.resolve()),
                result.config.radial_cells,
                f"{2.0 / result.config.radial_cells:.12g}",
                f"{result.config.time_step_s:.12g}",
                environment.extension_strategy,
                f"{environment.plateau_temperature_c:.12g}",
                f"{environment.plateau_moisture_kg_kg:.12g}",
                f"{result.drying_time_s / 3600.0:.12g}",
                sha256_file(result_xlsx),
            ]
        )


def archive_convergence_level(parent: Path, result: SimulationResult) -> Path:
    """Persist the complete output of one coarse/medium convergence run."""
    dt_token = f"{result.config.time_step_s:g}".replace(".", "p")
    level_dir = parent / "convergence_levels" / (
        f"N{result.config.radial_cells}_dt{dt_token}s"
    )
    if level_dir.exists():
        raise FileExistsError(f"convergence archive already exists: {level_dir}")
    level_dir.mkdir(parents=True, exist_ok=False)
    write_full_precision_csv(level_dir / "moisture_full_precision.csv", result)
    write_table5_csv(level_dir / "table5.csv", result)
    write_table5_markdown(level_dir / "table5.md", result)
    summary = {
        "config": asdict(result.config),
        "drying_time_s": result.drying_time_s,
        "drying_time_h": result.drying_time_s / 3600.0,
        "drying_moistures_kg_kg": result.drying_moistures_kg_kg,
        "diagnostics": {
            **asdict(result.diagnostics),
            "mean_picard_iterations": result.diagnostics.mean_picard_iterations,
        },
        "runtime_s": result.runtime_s,
    }
    (level_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return level_dir


def write_validation_report(
    path: Path,
    main_result: SimulationResult,
    environment: EnvironmentData,
    comparisons: Sequence[dict[str, float]],
    xlsx_validation: dict[str, object],
) -> None:
    diagnostics = main_result.diagnostics
    lines = [
        "# 第三问数值实验验证报告",
        "",
        "## 核心结果",
        "",
        f"- 烘干结束时间：`{main_result.drying_time_s / 3600.0:.9f} h`；",
        f"- 烘干结束时间：`{main_result.drying_time_s:.6f} s`；",
        f"- 终止时刻最大含水率：`{max(main_result.drying_moistures_kg_kg):.12f} kg/kg`；",
        f"- 最大含水率位置：`{main_result.radii_cm[main_result.drying_moistures_kg_kg.index(max(main_result.drying_moistures_kg_kg))]:.1f} cm`。",
        "",
        "## 环境边界延拓",
        "",
        f"- 策略：`{environment.extension_strategy}`；",
        f"- 稳态统计起点：`{environment.plateau_start_s / 3600.0:.3f} h`；",
        f"- 恒温阶段温度：`{environment.plateau_temperature_c:.9f} °C`；",
        f"- 恒温阶段水分浓度：`{environment.plateau_moisture_kg_kg:.12f} kg/kg`。",
        "",
        "## 主网格诊断",
        "",
        f"- 径向区间数：`{main_result.config.radial_cells}`；",
        f"- 时间步长：`{main_result.config.time_step_s:g} s`；",
        f"- 时间步数：`{diagnostics.steps}`；",
        f"- 运行时间：`{main_result.runtime_s:.3f} s`；",
        f"- 最大 Picard 迭代次数：`{diagnostics.max_picard_iterations}`；",
        f"- 平均 Picard 迭代次数：`{diagnostics.mean_picard_iterations:.3f}`；",
        f"- 最大热平衡绝对残差：`{diagnostics.max_heat_balance_abs:.3e}`；",
        f"- 最大热平衡相对残差：`{diagnostics.max_heat_balance_rel:.3e}`；",
        f"- 最大水分平衡绝对残差：`{diagnostics.max_moisture_balance_abs:.3e}`；",
        f"- 最大水分平衡相对残差：`{diagnostics.max_moisture_balance_rel:.3e}`；",
        f"- 最大径向单调性违反量：`{diagnostics.max_radial_monotonicity_violation:.3e}`。",
        "",
        "## 网格收敛",
        "",
        "| 粗网格 | 细网格 | 粗时间步/s | 细时间步/s | 烘干时间差/h | 公共 6 h 剖面最大差 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for item in comparisons:
        lines.append(
            "| {coarse_radial_cells:.0f} | {fine_radial_cells:.0f} | "
            "{coarse_time_step_s:g} | {fine_time_step_s:g} | "
            "{drying_time_difference_h:.9f} | "
            "{max_common_6h_profile_difference:.3e} |".format(**item)
        )
    if len(comparisons) >= 2:
        coarse_difference = comparisons[-2]["drying_time_difference_h"]
        fine_difference = comparisons[-1]["drying_time_difference_h"]
        if coarse_difference > fine_difference > 0.0:
            observed_order = math.log(coarse_difference / fine_difference, 2.0)
            estimated_fine_error_h = fine_difference / (2.0**observed_order - 1.0)
            extrapolated_time_h = (
                main_result.drying_time_s / 3600.0 - estimated_fine_error_h
            )
            lines.extend(
                [
                    "",
                    f"- 烘干时间观测收敛阶：`{observed_order:.6f}`；",
                    f"- 主网格烘干时间 Richardson 误差估计：`{estimated_fine_error_h:.9f} h`；",
                    f"- Richardson 外推烘干时间：`{extrapolated_time_h:.9f} h`。",
                ]
            )
    lines.extend(
        [
            "",
            "## Excel 结构验证",
            "",
            f"- ZIP 完整性：`{xlsx_validation['zip_integrity']}`；",
            f"- 工作表范围：`{xlsx_validation['dimension']}`；",
            f"- 行数（含表头）：`{xlsx_validation['rows_including_header']}`；",
            f"- 列数：`{xlsx_validation['columns']}`。",
            "",
            "终止事件及守恒检查均使用未舍入值；四位小数只用于提交表格和 Excel。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_self_tests() -> None:
    solution = solve_tridiagonal(
        [-1.0, -1.0], [2.0, 2.0, 2.0], [-1.0, -1.0], [1.0, 0.0, 1.0]
    )
    if max(abs(value - 1.0) for value in solution) > 1.0e-12:
        raise AssertionError("tridiagonal solver self-test failed")
    parameters = PhysicalParameters()
    expected = (
        2.4e-3 * math.exp(-0.45 / 2.55) * math.exp(-3850.0 / (28.0 + 273.15))
    )
    if abs(parameters.diffusivity(2.55, 28.0 + 273.15) - expected) > 1.0e-20:
        raise AssertionError("diffusivity self-test failed")
    face_value = integrated_face_diffusivity(0.5, 0.5, 323.15, parameters)
    if abs(face_value - parameters.diffusivity(0.5, 323.15)) > 1.0e-20:
        raise AssertionError("face diffusivity self-test failed")
    grid = make_uniform_grid(0.02, 80)
    if abs(sum(radial_storage_weights(grid)) - 0.02**2 / 2.0) > 1.0e-16:
        raise AssertionError("radial volume self-test failed")


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "A题" / "附件" / "附件1.xlsx",
        help="Attachment 1 XLSX path",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=root / "A题" / "附件" / "附件3" / "result3.xlsx",
        help="result3.xlsx template path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "result",
        help="output directory; defaults to submission/result",
    )
    parser.add_argument(
        "--experiment-name",
        help="optional directory name below problem3/experiments",
    )
    parser.add_argument("--radial-cells", type=int, default=2560)
    parser.add_argument("--time-step", type=float, default=0.5, dest="time_step_s")
    parser.add_argument("--maximum-time-h", type=float, default=96.0)
    parser.add_argument("--plateau-start-h", type=float, default=3.0)
    parser.add_argument(
        "--environment-extension",
        choices=("last_hour_mean", "last_point", "fixed"),
        default="fixed",
    )
    parser.add_argument("--fixed-environment-temperature-c", type=float, default=50.0)
    parser.add_argument("--fixed-environment-moisture", type=float, default=0.05)
    parser.add_argument("--skip-convergence", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_arguments(argv)
    run_self_tests()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if not args.template.is_file():
        raise FileNotFoundError(args.template)
    main_config = NumericalConfig(
        radial_cells=args.radial_cells,
        time_step_s=args.time_step_s,
        output_interval_s=60.0,
        maximum_time_h=args.maximum_time_h,
        plateau_start_h=args.plateau_start_h,
        environment_extension=args.environment_extension,
        fixed_environment_temperature_c=args.fixed_environment_temperature_c,
        fixed_environment_moisture_kg_kg=args.fixed_environment_moisture,
    )
    main_config.validate()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = load_environment(
        args.input,
        main_config.plateau_start_h,
        main_config.environment_extension,
        main_config.fixed_environment_temperature_c,
        main_config.fixed_environment_moisture_kg_kg,
    )
    parameters = PhysicalParameters()

    print(
        f"Main solve: N={main_config.radial_cells}, "
        f"dt={main_config.time_step_s:g} s, "
        f"plateau=({environment.plateau_temperature_c:.6f} °C, "
        f"{environment.plateau_moisture_kg_kg:.8f} kg/kg)",
        flush=True,
    )
    main_result = solve_model(environment, parameters, main_config, args.progress)
    print(
        f"Drying time = {main_result.drying_time_s / 3600.0:.9f} h "
        f"({main_result.drying_time_s:.3f} s)",
        flush=True,
    )

    convergence_results: list[SimulationResult] = [main_result]
    if not args.skip_convergence:
        if main_config.radial_cells % 4 != 0:
            raise ValueError("main radial_cells must be divisible by 4 for convergence")
        coarse_configs = [
            NumericalConfig(
                radial_cells=main_config.radial_cells // 4,
                time_step_s=main_config.time_step_s * 4.0,
                output_interval_s=max(60.0, main_config.time_step_s * 4.0),
                maximum_time_h=main_config.maximum_time_h,
                plateau_start_h=main_config.plateau_start_h,
                environment_extension=main_config.environment_extension,
                fixed_environment_temperature_c=main_config.fixed_environment_temperature_c,
                fixed_environment_moisture_kg_kg=main_config.fixed_environment_moisture_kg_kg,
            ),
            NumericalConfig(
                radial_cells=main_config.radial_cells // 2,
                time_step_s=main_config.time_step_s * 2.0,
                output_interval_s=max(60.0, main_config.time_step_s * 2.0),
                maximum_time_h=main_config.maximum_time_h,
                plateau_start_h=main_config.plateau_start_h,
                environment_extension=main_config.environment_extension,
                fixed_environment_temperature_c=main_config.fixed_environment_temperature_c,
                fixed_environment_moisture_kg_kg=main_config.fixed_environment_moisture_kg_kg,
            ),
        ]
        convergence_results = []
        for config in coarse_configs:
            print(
                f"Convergence solve: N={config.radial_cells}, "
                f"dt={config.time_step_s:g} s",
                flush=True,
            )
            result = solve_model(environment, parameters, config, False)
            convergence_results.append(result)
            print(f"  drying time = {result.drying_time_s / 3600.0:.9f} h")
        convergence_results.append(main_result)

    result_xlsx = output_dir / "result3.xlsx"
    write_result3_xlsx(args.template, result_xlsx, main_result)
    xlsx_validation = validate_result3_xlsx(result_xlsx, main_result)
    write_full_precision_csv(
        output_dir / "moisture_full_precision.csv", main_result
    )
    write_table5_csv(output_dir / "table5.csv", main_result)
    write_table5_markdown(output_dir / "table5.md", main_result)
    write_environment_csv(output_dir / "environment_input_used.csv", environment)
    comparisons = write_convergence_csv(
        output_dir / "convergence.csv", convergence_results
    )
    for level_result in convergence_results[:-1]:
        archive_convergence_level(output_dir, level_result)
    write_validation_report(
        output_dir / "validation_report.md",
        main_result,
        environment,
        comparisons,
        xlsx_validation,
    )

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "input_path": str(args.input.resolve()),
        "input_sha256": sha256_file(args.input),
        "template_path": str(args.template.resolve()),
        "template_sha256": sha256_file(args.template),
        "physical_parameters": asdict(parameters),
        "main_config": asdict(main_config),
        "environment": {
            "measured_end_s": environment.times_s[-1],
            "extension_strategy": environment.extension_strategy,
            "plateau_start_s": environment.plateau_start_s,
            "plateau_temperature_c": environment.plateau_temperature_c,
            "plateau_moisture_kg_kg": environment.plateau_moisture_kg_kg,
        },
        "drying_time_s": main_result.drying_time_s,
        "drying_time_h": main_result.drying_time_s / 3600.0,
        "diagnostics": {
            **asdict(main_result.diagnostics),
            "mean_picard_iterations": main_result.diagnostics.mean_picard_iterations,
        },
        "convergence": comparisons,
        "xlsx_validation": xlsx_validation,
        "output_sha256": sha256_file(result_xlsx),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    append_experiment_index(
        output_dir / "experiments_index.csv",
        output_dir,
        main_result,
        environment,
        result_xlsx,
    )
    print(f"Wrote outputs to {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
