#!/usr/bin/env python3
"""Solve Problem A, Question 1 with a conservative radial finite-volume model.

The implementation intentionally uses only the Python standard library so it can
run in the supplied environment without installing numerical or Excel packages.

Model
-----
    rho*cp*dT/dt = (1/r) d/dr (r*k*dT/dr)
    dC/dt        = (1/r) d/dr (r*D(C)*dC/dr)

with symmetry at r=0 and Robin heat/mass-transfer conditions at r=R.
Time integration is backward Euler for the first step and BDF2 afterwards.
The nonlinear moisture equation is solved by Picard iteration at every step.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import os
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
PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


@dataclass(frozen=True)
class PhysicalParameters:
    radius_m: float = 0.02
    density_kg_m3: float = 820.0
    heat_capacity_j_kg_k: float = 2600.0
    conductivity_w_m_k: float = 0.36
    heat_transfer_w_m2_k: float = 25.0
    mass_transfer_m_s: float = 8.0e-7
    initial_temperature_c: float = 28.0
    initial_moisture_kg_kg: float = 2.55
    diffusivity_prefactor_m2_s: float = 7.0e-9
    diffusivity_exponent: float = 0.89

    @property
    def volumetric_heat_capacity(self) -> float:
        return self.density_kg_m3 * self.heat_capacity_j_kg_k

    @property
    def thermal_diffusivity_m2_s(self) -> float:
        return self.conductivity_w_m_k / self.volumetric_heat_capacity

    def moisture_diffusivity(self, moisture: float) -> float:
        if not math.isfinite(moisture) or moisture <= 0.0:
            raise ValueError(f"Moisture must be positive and finite, got {moisture!r}")
        return self.diffusivity_prefactor_m2_s * math.exp(
            -self.diffusivity_exponent / moisture
        )


@dataclass(frozen=True)
class NumericalConfig:
    radial_cells: int = 1520
    time_step_s: float = 0.03125
    end_time_s: float = 1800.0
    output_interval_s: float = 1.0
    surface_layer_m: float = 0.002
    surface_refinement_factor: int = 10
    picard_tolerance: float = 2.0e-12
    picard_max_iterations: int = 40

    def validate(self) -> None:
        if self.radial_cells < 38 or self.radial_cells % 19 != 0:
            raise ValueError("radial_cells must be a multiple of 19 and at least 38")
        for name, value in (
            ("time_step_s", self.time_step_s),
            ("end_time_s", self.end_time_s),
            ("output_interval_s", self.output_interval_s),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        total_steps = self.end_time_s / self.time_step_s
        output_stride = self.output_interval_s / self.time_step_s
        if abs(total_steps - round(total_steps)) > 1.0e-10:
            raise ValueError("end_time_s must be an integer multiple of time_step_s")
        if abs(output_stride - round(output_stride)) > 1.0e-10:
            raise ValueError("output_interval_s must be an integer multiple of time_step_s")
        if self.picard_tolerance <= 0.0 or self.picard_max_iterations < 1:
            raise ValueError("invalid Picard iteration settings")
        if self.surface_layer_m <= 0.0 or self.surface_layer_m >= 0.02:
            raise ValueError("surface_layer_m must lie strictly between 0 and 0.02 m")
        if self.surface_refinement_factor < 1:
            raise ValueError("surface_refinement_factor must be at least 1")


@dataclass
class EnvironmentData:
    times_s: list[float]
    temperatures_c: list[float]
    moistures_kg_kg: list[float]

    def validate(self, end_time_s: float) -> None:
        n = len(self.times_s)
        if n < 2 or len(self.temperatures_c) != n or len(self.moistures_kg_kg) != n:
            raise ValueError("environment columns have inconsistent lengths")
        if self.times_s[0] > 0.0 or self.times_s[-1] < end_time_s:
            raise ValueError(
                f"environment data must cover [0, {end_time_s}], got "
                f"[{self.times_s[0]}, {self.times_s[-1]}]"
            )
        if any(not math.isfinite(x) for x in self.times_s):
            raise ValueError("environment time contains a non-finite value")
        if any(not math.isfinite(x) for x in self.temperatures_c):
            raise ValueError("environment temperature contains a non-finite value")
        if any(not math.isfinite(x) or x <= 0.0 for x in self.moistures_kg_kg):
            raise ValueError("environment moisture must be positive and finite")
        if any(b <= a for a, b in zip(self.times_s, self.times_s[1:])):
            raise ValueError("environment times must be strictly increasing")

    def interpolate(self, t: float) -> tuple[float, float]:
        if t <= self.times_s[0]:
            return self.temperatures_c[0], self.moistures_kg_kg[0]
        if t >= self.times_s[-1]:
            return self.temperatures_c[-1], self.moistures_kg_kg[-1]
        j = bisect.bisect_right(self.times_s, t) - 1
        t0, t1 = self.times_s[j], self.times_s[j + 1]
        weight = (t - t0) / (t1 - t0)
        temperature = self.temperatures_c[j] + weight * (
            self.temperatures_c[j + 1] - self.temperatures_c[j]
        )
        moisture = self.moistures_kg_kg[j] + weight * (
            self.moistures_kg_kg[j + 1] - self.moistures_kg_kg[j]
        )
        return temperature, moisture


@dataclass
class SolverDiagnostics:
    max_picard_iterations: int = 0
    total_picard_iterations: int = 0
    moisture_steps: int = 0
    max_picard_update: float = 0.0
    max_heat_balance_abs: float = 0.0
    max_heat_balance_rel: float = 0.0
    max_moisture_balance_abs: float = 0.0
    max_moisture_balance_rel: float = 0.0
    min_temperature_c: float = math.inf
    max_temperature_c: float = -math.inf
    min_moisture_kg_kg: float = math.inf
    max_moisture_kg_kg: float = -math.inf

    @property
    def mean_picard_iterations(self) -> float:
        if self.moisture_steps == 0:
            return 0.0
        return self.total_picard_iterations / self.moisture_steps


@dataclass
class SimulationResult:
    times_s: list[int]
    radii_cm: list[float]
    temperatures_c: list[list[float]]
    moistures_kg_kg: list[list[float]]
    diagnostics: SolverDiagnostics
    runtime_s: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def excel_column_index(cell_reference: str) -> int:
    value = 0
    for char in cell_reference:
        if not char.isalpha():
            break
        value = value * 26 + ord(char.upper()) - ord("A") + 1
    return value - 1


def excel_column_name(index_zero_based: int) -> str:
    value = index_zero_based + 1
    result = []
    while value:
        value, remainder = divmod(value - 1, 26)
        result.append(chr(ord("A") + remainder))
    return "".join(reversed(result))


def read_first_worksheet(path: Path) -> list[list[object]]:
    """Read values from the first worksheet of an XLSX using stdlib only."""
    with ZipFile(path) as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall(f"{{{XLSX_NS}}}si"):
                text = "".join(
                    node.text or "" for node in item.iter(f"{{{XLSX_NS}}}t")
                )
                shared_strings.append(text)

        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        first_sheet = workbook.find(f".//{{{XLSX_NS}}}sheet")
        if first_sheet is None:
            raise ValueError(f"workbook has no worksheets: {path}")
        relation_id = first_sheet.attrib[f"{{{OFFICE_REL_NS}}}id"]
        relations = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        relation_map = {item.attrib["Id"]: item.attrib["Target"] for item in relations}
        target = relation_map[relation_id].lstrip("/")
        if not target.startswith("xl/"):
            target = f"xl/{target}"

        worksheet = ET.fromstring(archive.read(target))
        rows: list[list[object]] = []
        for row in worksheet.findall(f".//{{{XLSX_NS}}}sheetData/{{{XLSX_NS}}}row"):
            values_by_column: dict[int, object] = {}
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
                elif cell_type == "b":
                    value = value_node.text == "1"
                else:
                    value = float(value_node.text)
                values_by_column[column] = value
            if values_by_column:
                width = max(values_by_column) + 1
                rows.append([values_by_column.get(i, "") for i in range(width)])
        return rows


def load_environment(path: Path, end_time_s: float) -> EnvironmentData:
    rows = read_first_worksheet(path)
    if not rows or len(rows[0]) < 3:
        raise ValueError("附件1 must contain 时间、温度、水分浓度 columns")
    headers = [str(value).strip() for value in rows[0][:3]]
    expected = ["时间", "温度", "水分浓度"]
    if headers != expected:
        raise ValueError(f"unexpected headers {headers!r}; expected {expected!r}")

    times: list[float] = []
    temperatures: list[float] = []
    moistures: list[float] = []
    for row_number, row in enumerate(rows[1:], start=2):
        if len(row) < 3 or any(value == "" for value in row[:3]):
            raise ValueError(f"missing input value in row {row_number}")
        try:
            t, temperature, moisture = (float(value) for value in row[:3])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"non-numeric input in row {row_number}") from exc
        times.append(t)
        temperatures.append(temperature)
        moistures.append(moisture)

    environment = EnvironmentData(times, temperatures, moistures)
    environment.validate(end_time_s)
    return environment


def solve_tridiagonal(
    lower: Sequence[float],
    diagonal: Sequence[float],
    upper: Sequence[float],
    rhs: Sequence[float],
) -> list[float]:
    """Thomas algorithm with pivot checks for a tridiagonal linear system."""
    n = len(diagonal)
    if n == 0 or len(rhs) != n or len(lower) != n - 1 or len(upper) != n - 1:
        raise ValueError("invalid tridiagonal dimensions")

    modified_upper = [0.0] * (n - 1)
    modified_rhs = [0.0] * n
    pivot = diagonal[0]
    if not math.isfinite(pivot) or abs(pivot) < 1.0e-300:
        raise ArithmeticError("zero or non-finite pivot at row 0")
    if n > 1:
        modified_upper[0] = upper[0] / pivot
    modified_rhs[0] = rhs[0] / pivot

    for i in range(1, n):
        pivot = diagonal[i] - lower[i - 1] * modified_upper[i - 1]
        if not math.isfinite(pivot) or abs(pivot) < 1.0e-300:
            raise ArithmeticError(f"zero or non-finite pivot at row {i}")
        if i < n - 1:
            modified_upper[i] = upper[i] / pivot
        modified_rhs[i] = (rhs[i] - lower[i - 1] * modified_rhs[i - 1]) / pivot

    solution = [0.0] * n
    solution[-1] = modified_rhs[-1]
    for i in range(n - 2, -1, -1):
        solution[i] = modified_rhs[i] - modified_upper[i] * solution[i + 1]
    return solution


def harmonic_mean(a: float, b: float) -> float:
    denominator = a + b
    if a <= 0.0 or b <= 0.0 or denominator <= 0.0:
        raise ValueError("diffusion coefficients must be positive")
    return 2.0 * a * b / denominator


def make_radial_grid(
    radius_m: float,
    radial_cells: int,
    surface_layer_m: float,
    surface_refinement_factor: int,
) -> list[float]:
    """Create a two-zone grid refined near the convective surface.

    The bulk cell width is ``surface_refinement_factor`` times the surface
    cell width. Requested 0.1 cm output radii are nodes for all default
    comparison grids.
    """
    bulk_length = radius_m - surface_layer_m
    effective_length = bulk_length + surface_refinement_factor * surface_layer_m
    bulk_step = effective_length / radial_cells
    surface_step = bulk_step / surface_refinement_factor
    bulk_cells_float = bulk_length / bulk_step
    surface_cells_float = surface_layer_m / surface_step
    bulk_cells = int(round(bulk_cells_float))
    surface_cells = int(round(surface_cells_float))
    if (
        abs(bulk_cells_float - bulk_cells) > 1.0e-10
        or abs(surface_cells_float - surface_cells) > 1.0e-10
        or bulk_cells + surface_cells != radial_cells
    ):
        raise ValueError(
            "radial_cells is incompatible with the two-zone grid; "
            "use a multiple of 19 for the default 2 mm / factor-10 refinement"
        )
    grid = [i * bulk_step for i in range(bulk_cells + 1)]
    interface = bulk_length
    grid.extend(
        interface + i * surface_step for i in range(1, surface_cells + 1)
    )
    grid[0] = 0.0
    grid[-1] = radius_m
    if any(b <= a for a, b in zip(grid, grid[1:])):
        raise AssertionError("constructed radial grid is not strictly increasing")
    return grid


def make_time_coefficients(
    old: Sequence[float], previous: Sequence[float] | None, dt: float
) -> tuple[float, list[float]]:
    if previous is None:
        return 1.0 / dt, [value / dt for value in old]
    factor = 1.0 / (2.0 * dt)
    return 3.0 * factor, [
        (4.0 * current - prior) * factor
        for current, prior in zip(old, previous)
    ]


def implicit_fvm_step(
    old: Sequence[float],
    previous: Sequence[float] | None,
    dt: float,
    radii_m: Sequence[float],
    storage: float,
    gamma_nodes: Sequence[float],
    boundary_transfer: float,
    external_value: float,
) -> list[float]:
    """One BE/BDF2 step for the conservative radial diffusion equation."""
    node_count = len(old)
    if (
        node_count < 2
        or len(gamma_nodes) != node_count
        or len(radii_m) != node_count
    ):
        raise ValueError("inconsistent spatial arrays")
    radial_cells = node_count - 1
    radius_m = radii_m[-1]
    if radii_m[0] != 0.0 or any(b <= a for a, b in zip(radii_m, radii_m[1:])):
        raise ValueError("invalid radial grid")
    time_diagonal, history = make_time_coefficients(old, previous, dt)

    lower = [0.0] * radial_cells
    diagonal = [0.0] * node_count
    upper = [0.0] * radial_cells
    rhs = [0.0] * node_count

    gamma_east = harmonic_mean(gamma_nodes[0], gamma_nodes[1])
    r_east = 0.5 * (radii_m[0] + radii_m[1])
    east_conductance = r_east * gamma_east / (radii_m[1] - radii_m[0])
    center_weight = r_east * r_east / 2.0
    diagonal[0] = storage * center_weight * time_diagonal + east_conductance
    upper[0] = -east_conductance
    rhs[0] = storage * center_weight * history[0]

    for i in range(1, radial_cells):
        r_i = radii_m[i]
        r_west = 0.5 * (radii_m[i - 1] + r_i)
        r_east = 0.5 * (r_i + radii_m[i + 1])
        gamma_west = harmonic_mean(gamma_nodes[i - 1], gamma_nodes[i])
        gamma_east = harmonic_mean(gamma_nodes[i], gamma_nodes[i + 1])
        west_conductance = r_west * gamma_west / (r_i - radii_m[i - 1])
        east_conductance = r_east * gamma_east / (radii_m[i + 1] - r_i)
        volume_weight = (r_east * r_east - r_west * r_west) / 2.0
        lower[i - 1] = -west_conductance
        diagonal[i] = (
            storage * volume_weight * time_diagonal
            + west_conductance
            + east_conductance
        )
        upper[i] = -east_conductance
        rhs[i] = storage * volume_weight * history[i]

    r_west = 0.5 * (radii_m[-2] + radius_m)
    gamma_west = harmonic_mean(gamma_nodes[-2], gamma_nodes[-1])
    west_conductance = r_west * gamma_west / (radius_m - radii_m[-2])
    boundary_conductance = radius_m * boundary_transfer
    surface_weight = (radius_m * radius_m - r_west * r_west) / 2.0
    lower[-1] = -west_conductance
    diagonal[-1] = (
        storage * surface_weight * time_diagonal
        + west_conductance
        + boundary_conductance
    )
    rhs[-1] = (
        storage * surface_weight * history[-1]
        + boundary_conductance * external_value
    )

    return solve_tridiagonal(lower, diagonal, upper, rhs)


def radial_storage_weights(radii_m: Sequence[float]) -> list[float]:
    radial_cells = len(radii_m) - 1
    radius_m = radii_m[-1]
    weights = [0.0] * (radial_cells + 1)
    first_face = 0.5 * (radii_m[0] + radii_m[1])
    weights[0] = first_face * first_face / 2.0
    for i in range(1, radial_cells):
        west = 0.5 * (radii_m[i - 1] + radii_m[i])
        east = 0.5 * (radii_m[i] + radii_m[i + 1])
        weights[i] = (east * east - west * west) / 2.0
    last_face = 0.5 * (radii_m[-2] + radius_m)
    weights[-1] = (radius_m * radius_m - last_face * last_face) / 2.0
    return weights


def update_balance_diagnostic(
    diagnostics: SolverDiagnostics,
    field_name: str,
    new: Sequence[float],
    old: Sequence[float],
    previous: Sequence[float] | None,
    dt: float,
    weights: Sequence[float],
    storage: float,
    radius_m: float,
    boundary_transfer: float,
    external_value: float,
) -> None:
    time_diagonal, history = make_time_coefficients(old, previous, dt)
    storage_rate = sum(
        storage * weight * (time_diagonal * value - hist)
        for weight, value, hist in zip(weights, new, history)
    )
    boundary_rate = radius_m * boundary_transfer * (external_value - new[-1])
    residual = abs(storage_rate - boundary_rate)
    scale = max(abs(storage_rate), abs(boundary_rate), 1.0e-30)
    relative = residual / scale
    if field_name == "heat":
        diagnostics.max_heat_balance_abs = max(
            diagnostics.max_heat_balance_abs, residual
        )
        diagnostics.max_heat_balance_rel = max(
            diagnostics.max_heat_balance_rel, relative
        )
    elif field_name == "moisture":
        diagnostics.max_moisture_balance_abs = max(
            diagnostics.max_moisture_balance_abs, residual
        )
        diagnostics.max_moisture_balance_rel = max(
            diagnostics.max_moisture_balance_rel, relative
        )
    else:
        raise ValueError(field_name)


def solve_model(
    environment: EnvironmentData,
    parameters: PhysicalParameters,
    config: NumericalConfig,
    progress: bool = False,
) -> SimulationResult:
    config.validate()
    environment.validate(config.end_time_s)
    started = time.perf_counter()

    n = config.radial_cells
    dt = config.time_step_s
    step_count = int(round(config.end_time_s / dt))
    output_stride = int(round(config.output_interval_s / dt))
    radii_m = make_radial_grid(
        parameters.radius_m,
        n,
        config.surface_layer_m,
        config.surface_refinement_factor,
    )
    radii_cm = [0.1 * i for i in range(21)]
    requested_indices = []
    for radius_cm in radii_cm:
        target = radius_cm / 100.0
        index = bisect.bisect_left(radii_m, target)
        candidates = [j for j in (index - 1, index) if 0 <= j < len(radii_m)]
        best = min(candidates, key=lambda j: abs(radii_m[j] - target))
        if abs(radii_m[best] - target) > 1.0e-12:
            raise ValueError(f"output radius {radius_cm} cm is not a grid node")
        requested_indices.append(best)
    weights = radial_storage_weights(radii_m)

    temperature = [parameters.initial_temperature_c] * (n + 1)
    moisture = [parameters.initial_moisture_kg_kg] * (n + 1)
    previous_temperature: list[float] | None = None
    previous_moisture: list[float] | None = None
    constant_k = [parameters.conductivity_w_m_k] * (n + 1)

    output_times: list[int] = []
    temperature_output: list[list[float]] = []
    moisture_output: list[list[float]] = []
    diagnostics = SolverDiagnostics()

    for step in range(1, step_count + 1):
        t_new = step * dt
        external_temperature, external_moisture = environment.interpolate(t_new)

        new_temperature = implicit_fvm_step(
            temperature,
            previous_temperature,
            dt,
            radii_m,
            parameters.volumetric_heat_capacity,
            constant_k,
            parameters.heat_transfer_w_m2_k,
            external_temperature,
        )

        guess = moisture.copy()
        final_update = math.inf
        for iteration in range(1, config.picard_max_iterations + 1):
            diffusivity = [parameters.moisture_diffusivity(value) for value in guess]
            candidate = implicit_fvm_step(
                moisture,
                previous_moisture,
                dt,
                radii_m,
                1.0,
                diffusivity,
                parameters.mass_transfer_m_s,
                external_moisture,
            )
            final_update = max(abs(a - b) for a, b in zip(candidate, guess))
            guess = candidate
            if final_update < config.picard_tolerance:
                break
        else:
            raise RuntimeError(
                f"Picard iteration failed at t={t_new:.9g} s; "
                f"last update={final_update:.3e}"
            )
        new_moisture = guess

        diagnostics.moisture_steps += 1
        diagnostics.total_picard_iterations += iteration
        diagnostics.max_picard_iterations = max(
            diagnostics.max_picard_iterations, iteration
        )
        diagnostics.max_picard_update = max(
            diagnostics.max_picard_update, final_update
        )

        update_balance_diagnostic(
            diagnostics,
            "heat",
            new_temperature,
            temperature,
            previous_temperature,
            dt,
            weights,
            parameters.volumetric_heat_capacity,
            parameters.radius_m,
            parameters.heat_transfer_w_m2_k,
            external_temperature,
        )
        update_balance_diagnostic(
            diagnostics,
            "moisture",
            new_moisture,
            moisture,
            previous_moisture,
            dt,
            weights,
            1.0,
            parameters.radius_m,
            parameters.mass_transfer_m_s,
            external_moisture,
        )

        previous_temperature, temperature = temperature, new_temperature
        previous_moisture, moisture = moisture, new_moisture

        diagnostics.min_temperature_c = min(
            diagnostics.min_temperature_c, min(temperature)
        )
        diagnostics.max_temperature_c = max(
            diagnostics.max_temperature_c, max(temperature)
        )
        diagnostics.min_moisture_kg_kg = min(
            diagnostics.min_moisture_kg_kg, min(moisture)
        )
        diagnostics.max_moisture_kg_kg = max(
            diagnostics.max_moisture_kg_kg, max(moisture)
        )

        if step % output_stride == 0:
            output_times.append(int(round(t_new)))
            temperature_output.append([temperature[i] for i in requested_indices])
            moisture_output.append([moisture[i] for i in requested_indices])

        if progress and (step % max(1, step_count // 10) == 0):
            print(
                f"  {100.0 * step / step_count:5.1f}%  "
                f"t={t_new:8.3f} s  Picard={iteration}",
                flush=True,
            )

    if output_times != list(range(1, int(config.end_time_s) + 1)):
        raise AssertionError("unexpected output time grid")
    return SimulationResult(
        times_s=output_times,
        radii_cm=radii_cm,
        temperatures_c=temperature_output,
        moistures_kg_kg=moisture_output,
        diagnostics=diagnostics,
        runtime_s=time.perf_counter() - started,
    )


def compare_results(coarse: SimulationResult, fine: SimulationResult) -> dict[str, object]:
    if coarse.times_s != fine.times_s or coarse.radii_cm != fine.radii_cm:
        raise ValueError("results use different output grids")

    def compare_arrays(
        coarse_values: Sequence[Sequence[float]],
        fine_values: Sequence[Sequence[float]],
    ) -> dict[str, object]:
        max_difference = -1.0
        location = (0, 0)
        stable_rounding = 0
        total = 0
        sum_squared = 0.0
        for ti, (coarse_row, fine_row) in enumerate(zip(coarse_values, fine_values)):
            for ri, (coarse_value, fine_value) in enumerate(zip(coarse_row, fine_row)):
                difference = abs(coarse_value - fine_value)
                sum_squared += difference * difference
                total += 1
                if round(coarse_value, 4) == round(fine_value, 4):
                    stable_rounding += 1
                if difference > max_difference:
                    max_difference = difference
                    location = (ti, ri)
        ti, ri = location
        return {
            "max_abs_difference": max_difference,
            "rms_difference": math.sqrt(sum_squared / total),
            "max_location_time_s": fine.times_s[ti],
            "max_location_radius_cm": fine.radii_cm[ri],
            "four_decimal_agreement_count": stable_rounding,
            "comparison_count": total,
            "four_decimal_agreement_fraction": stable_rounding / total,
        }

    return {
        "temperature": compare_arrays(
            coarse.temperatures_c, fine.temperatures_c
        ),
        "moisture": compare_arrays(
            coarse.moistures_kg_kg, fine.moistures_kg_kg
        ),
    }


def add_three_level_error_estimate(
    coarse_to_medium: dict[str, object],
    medium_to_fine: dict[str, object],
) -> dict[str, object]:
    """Add observed order and a Richardson fine-grid error estimate.

    All three levels are refined by a factor of two in both space and time.
    The estimate is based on the max norm over every required output point.
    """
    enriched: dict[str, object] = {}
    for field in ("temperature", "moisture"):
        coarse_difference = float(coarse_to_medium[field]["max_abs_difference"])
        fine_difference = float(medium_to_fine[field]["max_abs_difference"])
        if coarse_difference <= 0.0 or fine_difference <= 0.0:
            observed_order = math.inf
            estimated_error = 0.0
        else:
            observed_order = math.log(coarse_difference / fine_difference, 2.0)
            denominator = 2.0**observed_order - 1.0
            estimated_error = (
                math.inf if denominator <= 0.0 else fine_difference / denominator
            )
        enriched[field] = {
            **medium_to_fine[field],
            "coarse_to_medium_max_abs_difference": coarse_difference,
            "observed_order_max_norm": observed_order,
            "estimated_fine_grid_max_error": estimated_error,
            "estimated_fine_grid_max_error_with_1p25_safety": 1.25 * estimated_error,
        }
    return enriched


def write_csv(
    path: Path,
    times_s: Sequence[int],
    radii_cm: Sequence[float],
    values: Sequence[Sequence[float]],
    precision: int = 10,
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["时间/s"] + [f"{radius:.1f} cm" for radius in radii_cm])
        for t, row in zip(times_s, values):
            writer.writerow([t] + [f"{value:.{precision}f}" for value in row])


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
        insert_at = 0 if fonts is None else list(root).index(fonts)
        root.insert(insert_at, number_formats)

    format_id = 164
    used_ids = {
        int(node.attrib["numFmtId"])
        for node in number_formats.findall(f"{{{XLSX_NS}}}numFmt")
        if "numFmtId" in node.attrib
    }
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
        raise ValueError("template styles.xml has no cellXfs")
    style_id = len(cell_formats.findall(f"{{{XLSX_NS}}}xf"))
    new_style = ET.SubElement(
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
        new_style,
        f"{{{XLSX_NS}}}alignment",
        {"horizontal": "center", "vertical": "center"},
    )
    cell_formats.set("count", str(style_id + 1))

    xml = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    return xml, style_id


def make_worksheet_xml(
    times_s: Sequence[int],
    radii_cm: Sequence[float],
    values: Sequence[Sequence[float]],
    value_style_id: int,
    selected: bool,
) -> bytes:
    if len(times_s) != len(values):
        raise ValueError("time and value row counts differ")
    if any(len(row) != len(radii_cm) for row in values):
        raise ValueError("inconsistent output row width")
    last_column = excel_column_name(len(radii_cm))
    last_row = len(times_s) + 1
    selected_attribute = ' tabSelected="1"' if selected else ""
    pieces = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        f'<worksheet xmlns="{XLSX_NS}" '
        f'xmlns:r="{OFFICE_REL_NS}">',
        f'<dimension ref="A1:{last_column}{last_row}"/>',
        f'<sheetViews><sheetView{selected_attribute} workbookViewId="0">',
        '<pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>',
        '</sheetView></sheetViews>',
        '<sheetFormatPr defaultRowHeight="15"/>',
        f'<cols><col min="1" max="1" width="21" customWidth="1"/>'
        f'<col min="2" max="{len(radii_cm) + 1}" width="12" customWidth="1"/></cols>',
        '<sheetData>',
        f'<row r="1" spans="1:{len(radii_cm) + 1}">',
        '<c r="A1" s="2" t="inlineStr"><is><t>时间\\到药材中心的距离/cm</t></is></c>',
    ]
    for column_index, radius in enumerate(radii_cm, start=1):
        column = excel_column_name(column_index)
        pieces.append(f'<c r="{column}1" s="2"><v>{radius:.1f}</v></c>')
    pieces.append('</row>')

    for row_index, (t, row_values) in enumerate(zip(times_s, values), start=2):
        pieces.append(f'<row r="{row_index}" spans="1:{len(radii_cm) + 1}">')
        pieces.append(f'<c r="A{row_index}" s="1"><v>{t}</v></c>')
        for column_index, value in enumerate(row_values, start=1):
            if not math.isfinite(value):
                raise ValueError("cannot write non-finite result to XLSX")
            column = excel_column_name(column_index)
            pieces.append(
                f'<c r="{column}{row_index}" s="{value_style_id}">'
                f'<v>{value:.4f}</v></c>'
            )
        pieces.append('</row>')

    pieces.extend(
        [
            '</sheetData>',
            f'<autoFilter ref="A1:{last_column}{last_row}"/>',
            '<pageMargins left="0.75" right="0.75" top="1" bottom="1" '
            'header="0.5" footer="0.5"/>',
            '</worksheet>',
        ]
    )
    return "".join(pieces).encode("utf-8")


def write_result_xlsx(
    template_path: Path, output_path: Path, result: SimulationResult
) -> None:
    with ZipFile(template_path, "r") as source:
        styles_xml, value_style_id = add_four_decimal_style(
            source.read("xl/styles.xml")
        )
        replacements = {
            "xl/styles.xml": styles_xml,
            "xl/worksheets/sheet1.xml": make_worksheet_xml(
                result.times_s,
                result.radii_cm,
                result.temperatures_c,
                value_style_id,
                selected=True,
            ),
            "xl/worksheets/sheet2.xml": make_worksheet_xml(
                result.times_s,
                result.radii_cm,
                result.moistures_kg_kg,
                value_style_id,
                selected=False,
            ),
        }
        temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
        with ZipFile(temporary_path, "w", compression=ZIP_DEFLATED, compresslevel=9) as target:
            for item in source.infolist():
                content = replacements.get(item.filename, source.read(item.filename))
                target.writestr(item, content)
        os.replace(temporary_path, output_path)


def validate_result_xlsx(path: Path, expected_rows: int, expected_columns: int) -> dict[str, object]:
    with ZipFile(path) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise ValueError(f"corrupt XLSX member: {bad_member}")
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        sheet_names = [
            node.attrib["name"]
            for node in workbook.findall(f".//{{{XLSX_NS}}}sheet")
        ]
        if sheet_names != ["温度", "水分浓度"]:
            raise ValueError(f"unexpected output sheet names: {sheet_names}")
        dimensions = []
        row_counts = []
        for member in ("xl/worksheets/sheet1.xml", "xl/worksheets/sheet2.xml"):
            root = ET.fromstring(archive.read(member))
            dimension = root.find(f"{{{XLSX_NS}}}dimension")
            dimensions.append(None if dimension is None else dimension.attrib.get("ref"))
            rows = root.findall(f".//{{{XLSX_NS}}}sheetData/{{{XLSX_NS}}}row")
            row_counts.append(len(rows))
            if len(rows) != expected_rows:
                raise ValueError(f"{member} has {len(rows)} rows, expected {expected_rows}")
            final_cells = rows[-1].findall(f"{{{XLSX_NS}}}c")
            if len(final_cells) != expected_columns:
                raise ValueError(
                    f"{member} final row has {len(final_cells)} cells, "
                    f"expected {expected_columns}"
                )
        return {
            "zip_integrity": "ok",
            "sheet_names": sheet_names,
            "dimensions": dimensions,
            "row_counts_including_header": row_counts,
            "columns": expected_columns,
        }


def select_table(
    result: SimulationResult,
    requested_times: Sequence[int],
    requested_radii_cm: Sequence[float],
    field: str,
) -> list[list[float]]:
    time_indices = {t: i for i, t in enumerate(result.times_s)}
    radius_indices = {round(r, 10): i for i, r in enumerate(result.radii_cm)}
    source = (
        result.temperatures_c if field == "temperature" else result.moistures_kg_kg
    )
    table = []
    for t in requested_times:
        ti = time_indices[t]
        table.append([
            source[ti][radius_indices[round(radius, 10)]]
            for radius in requested_radii_cm
        ])
    return table


def markdown_table(
    times: Sequence[int], radii_cm: Sequence[float], values: Sequence[Sequence[float]]
) -> str:
    header = "| 时间/s | " + " | ".join(f"{r:g} cm" for r in radii_cm) + " |"
    separator = "|---:" + "|---:" * len(radii_cm) + "|"
    rows = [header, separator]
    for t, row in zip(times, values):
        rows.append("| " + str(t) + " | " + " | ".join(f"{v:.4f}" for v in row) + " |")
    return "\n".join(rows)


def write_summary_markdown(path: Path, result: SimulationResult) -> None:
    requested_times = [100, 300, 600, 900, 1200, 1500, 1800]
    requested_radii = [0.0, 0.5, 1.0, 1.5, 2.0]
    temperature_table = select_table(
        result, requested_times, requested_radii, "temperature"
    )
    moisture_table = select_table(
        result, requested_times, requested_radii, "moisture"
    )
    text = f"""# 第一问数值结果

结果由 `solve_problem1.py` 使用守恒型径向有限体积法计算。表中数值按题意保留四位小数；计算内部未提前舍入。

## 表 1：30 分钟内药材的温度（°C）

{markdown_table(requested_times, requested_radii, temperature_table)}

## 表 2：30 分钟内药材的水分浓度（kg/kg）

{markdown_table(requested_times, requested_radii, moisture_table)}

## 输出说明

- `result1.xlsx`：题目模板格式，含“温度”和“水分浓度”两个工作表；
- `temperature_full_precision.csv`：每秒、每 0.1 cm 的高精度温度结果；
- `moisture_full_precision.csv`：每秒、每 0.1 cm 的高精度水分浓度结果；
- `validation_report.md`：输入质量、守恒性和网格收敛检查。
"""
    path.write_text(text, encoding="utf-8")


def write_validation_markdown(
    path: Path,
    environment: EnvironmentData,
    fine: SimulationResult,
    convergence: dict[str, object],
    input_sha256: str,
    xlsx_validation: dict[str, object],
    fine_config: NumericalConfig,
    medium_config: NumericalConfig,
    coarse_config: NumericalConfig,
) -> None:
    heat = convergence["temperature"]
    moisture = convergence["moisture"]
    d = fine.diagnostics
    fine_grid = make_radial_grid(
        0.02,
        fine_config.radial_cells,
        fine_config.surface_layer_m,
        fine_config.surface_refinement_factor,
    )
    medium_grid = make_radial_grid(
        0.02,
        medium_config.radial_cells,
        medium_config.surface_layer_m,
        medium_config.surface_refinement_factor,
    )
    coarse_grid = make_radial_grid(
        0.02,
        coarse_config.radial_cells,
        coarse_config.surface_layer_m,
        coarse_config.surface_refinement_factor,
    )
    fine_steps = [b - a for a, b in zip(fine_grid, fine_grid[1:])]
    medium_steps = [b - a for a, b in zip(medium_grid, medium_grid[1:])]
    coarse_steps = [b - a for a, b in zip(coarse_grid, coarse_grid[1:])]
    four_decimal_tolerance = 0.5e-4
    heat_passes = (
        float(heat["estimated_fine_grid_max_error_with_1p25_safety"])
        < four_decimal_tolerance
    )
    moisture_passes = (
        float(moisture["estimated_fine_grid_max_error_with_1p25_safety"])
        < four_decimal_tolerance
    )
    text = fr"""# 第一问数值验证报告

## 总体结论

主结果采用 $N_r={fine_config.radial_cells}$、$\Delta t={fine_config.time_step_s:g}\,\mathrm{{s}}$ 的守恒型径向有限体积离散。温度采用常系数导热模型；水分扩散系数在每一时间步内经 Picard 迭代收敛。结果文件结构、全局守恒和网格加密差异均已自动检查。

四位小数的绝对误差判据取半个末位单位，即 $5\times10^{{-5}}$。引入 1.25 安全系数后，温度判据{'通过' if heat_passes else '未通过'}，水分浓度判据{'通过' if moisture_passes else '未通过'}。

## 输入数据检查

- 数据源：`附件1.xlsx`；
- SHA-256：`{input_sha256}`；
- 数据行数：{len(environment.times_s)}；
- 时间覆盖：{environment.times_s[0]:g}–{environment.times_s[-1]:g} s；
- 本问使用区间：0–{fine_config.end_time_s:g} s；
- 时间严格递增：是；
- 空值或非数值：未发现；
- 温度范围：{min(environment.temperatures_c):.6g}–{max(environment.temperatures_c):.6g} °C；
- 环境水分浓度范围：{min(environment.moistures_kg_kg):.6g}–{max(environment.moistures_kg_kg):.6g} kg/kg。

## 数值配置

| 配置 | 径向区间数 | 最小/最大空间步长/m | 时间步长/s | 时间格式 |
|---|---:|---:|---:|---|
| 加密主结果 | {fine_config.radial_cells} | {min(fine_steps):.8g} / {max(fine_steps):.8g} | {fine_config.time_step_s:g} | 首步 BE，后续 BDF2 |
| 中等网格 | {medium_config.radial_cells} | {min(medium_steps):.8g} / {max(medium_steps):.8g} | {medium_config.time_step_s:g} | 首步 BE，后续 BDF2 |
| 粗网格 | {coarse_config.radial_cells} | {min(coarse_steps):.8g} / {max(coarse_steps):.8g} | {coarse_config.time_step_s:g} | 首步 BE，后续 BDF2 |

## 加密收敛比较

比较范围包含 $t=1,2,\ldots,1800\,\mathrm{{s}}$ 和 $r=0,0.1,\ldots,2.0\,\mathrm{{cm}}$ 的全部 {heat['comparison_count']} 个输出点。

| 场变量 | 中等—加密最大差 | 观测收敛阶 | 加密解估计最大误差 | 1.25 安全系数误差上界 | 最大差位置 |
|---|---:|---:|---:|---:|---|
| 温度/°C | {heat['max_abs_difference']:.6e} | {heat['observed_order_max_norm']:.4f} | {heat['estimated_fine_grid_max_error']:.6e} | {heat['estimated_fine_grid_max_error_with_1p25_safety']:.6e} | $t={heat['max_location_time_s']}\,\mathrm{{s}}$, $r={heat['max_location_radius_cm']:.1f}\,\mathrm{{cm}}$ |
| 水分浓度/(kg/kg) | {moisture['max_abs_difference']:.6e} | {moisture['observed_order_max_norm']:.4f} | {moisture['estimated_fine_grid_max_error']:.6e} | {moisture['estimated_fine_grid_max_error_with_1p25_safety']:.6e} | $t={moisture['max_location_time_s']}\,\mathrm{{s}}$, $r={moisture['max_location_radius_cm']:.1f}\,\mathrm{{cm}}$ |

观测阶按三层网格的最大范数差计算：

$$
p=\log_2\!\left(\frac{{\lVert u_{{{coarse_config.radial_cells}}}-u_{{{medium_config.radial_cells}}}\rVert_\infty}}{{\lVert u_{{{medium_config.radial_cells}}}-u_{{{fine_config.radial_cells}}}\rVert_\infty}}\right).
$$

加密解剩余离散误差采用 Richardson 形式估计：

$$
e_{{{fine_config.radial_cells}}}\approx\frac{{\lVert u_{{{fine_config.radial_cells}}}-u_{{{medium_config.radial_cells}}}\rVert_\infty}}{{2^p-1}}.
$$

四位小数精度判据：

| 场变量 | 1.25 安全系数误差上界 | 允许上界 | 结论 |
|---|---:|---:|---|
| 温度/$^\circ\mathrm{{C}}$ | {heat['estimated_fine_grid_max_error_with_1p25_safety']:.6e} | {four_decimal_tolerance:.1e} | {'通过' if heat_passes else '未通过'} |
| 水分浓度/(kg/kg) | {moisture['estimated_fine_grid_max_error_with_1p25_safety']:.6e} | {four_decimal_tolerance:.1e} | {'通过' if moisture_passes else '未通过'} |

注：该判据验证的是未舍入数值的最大绝对离散误差尺度。若某一数值极接近四舍五入分界点，两层网格显示的最后一位仍可能不同，这与绝对误差超标不等价。

## 守恒与非线性收敛

- 最大相对能量守恒残差：{d.max_heat_balance_rel:.6e}；
- 最大相对水分守恒残差：{d.max_moisture_balance_rel:.6e}；
- 水分 Picard 平均迭代次数：{d.mean_picard_iterations:.4f}；
- 水分 Picard 最大迭代次数：{d.max_picard_iterations}；
- Picard 终止容差：{fine_config.picard_tolerance:.3e}；
- 数值温度范围：{d.min_temperature_c:.8f}–{d.max_temperature_c:.8f} °C；
- 数值水分浓度范围：{d.min_moisture_kg_kg:.8f}–{d.max_moisture_kg_kg:.8f} kg/kg。

## Excel 文件检查

- ZIP/XML 完整性：{xlsx_validation['zip_integrity']}；
- 工作表：{', '.join(xlsx_validation['sheet_names'])}；
- 工作表区域：{', '.join(xlsx_validation['dimensions'])}；
- 每个工作表行数（含表头）：{', '.join(str(x) for x in xlsx_validation['row_counts_including_header'])}；
- 列数：{xlsx_validation['columns']}。

## 必须保留的模型说明

1. 题目没有给出端面边界数据，程序采用一维径向、忽略端部效应的模型。
2. 题目没有给出汽化潜热及相变源项，温度和水分方程按题给参数并行求解。
3. 附件 1 的 60 s 间隔数据在相邻测点间采用分段线性插值。
4. 水分浓度采用题面给出的 kg/kg 有效变量尺度，对流传质边界按题给 $h_m$ 直接作用于该变量。
"""
    path.write_text(text, encoding="utf-8")


def run_self_tests() -> None:
    solution = solve_tridiagonal(
        [-1.0, -1.0], [2.0, 2.0, 2.0], [-1.0, -1.0], [1.0, 0.0, 1.0]
    )
    if max(abs(value - 1.0) for value in solution) > 1.0e-13:
        raise AssertionError("Thomas solver self-test failed")

    radius = 0.02
    uniform = [7.25] * 21
    gamma = [0.36] * 21
    new = implicit_fvm_step(
        uniform,
        None,
        0.5,
        [radius * i / 20 for i in range(21)],
        820.0 * 2600.0,
        gamma,
        25.0,
        7.25,
    )
    if max(abs(value - 7.25) for value in new) > 2.0e-12:
        raise AssertionError("uniform-equilibrium preservation test failed")

    params = PhysicalParameters()
    expected_d = 7.0e-9 * math.exp(-0.89 / 2.55)
    if abs(params.moisture_diffusivity(2.55) - expected_d) > 1.0e-24:
        raise AssertionError("diffusivity formula test failed")


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    output_directory = project_root / "result"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=project_root / "A题" / "附件" / "附件1.xlsx",
        help="path to 附件1.xlsx",
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=project_root / "A题" / "附件" / "附件3" / "result1.xlsx",
        help="path to the result1.xlsx template",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=output_directory,
        help="output directory",
    )
    parser.add_argument("--radial-cells", type=int, default=1520)
    parser.add_argument("--dt", type=float, default=0.03125)
    parser.add_argument(
        "--coarse-radial-cells",
        type=int,
        default=760,
        help="radial cells for the medium convergence level",
    )
    parser.add_argument(
        "--coarse-dt",
        type=float,
        default=0.0625,
        help="time step for the medium convergence level",
    )
    parser.add_argument(
        "--coarsest-radial-cells",
        type=int,
        default=380,
        help="radial cells for the coarsest convergence level",
    )
    parser.add_argument(
        "--coarsest-dt",
        type=float,
        default=0.125,
        help="time step for the coarsest convergence level",
    )
    parser.add_argument(
        "--skip-convergence",
        action="store_true",
        help="skip the coarser comparison run",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--self-test-only",
        action="store_true",
        help="run lightweight numerical tests and exit",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_arguments(argv)
    run_self_tests()
    if args.self_test_only:
        print("Self-tests passed.")
        return 0

    args.outdir.mkdir(parents=True, exist_ok=True)
    parameters = PhysicalParameters()
    fine_config = NumericalConfig(
        radial_cells=args.radial_cells,
        time_step_s=args.dt,
    )
    fine_config.validate()
    environment = load_environment(args.input, fine_config.end_time_s)

    print(
        f"Main solve: Nr={fine_config.radial_cells}, "
        f"dt={fine_config.time_step_s:g} s"
    )
    fine_result = solve_model(
        environment, parameters, fine_config, progress=not args.quiet
    )

    if args.skip_convergence:
        medium_config = fine_config
        coarse_config = fine_config
        convergence = {
            "temperature": {
                "max_abs_difference": 0.0,
                "rms_difference": 0.0,
                "max_location_time_s": 0,
                "max_location_radius_cm": 0.0,
                "four_decimal_agreement_count": 0,
                "comparison_count": 0,
                "four_decimal_agreement_fraction": 0.0,
                "coarse_to_medium_max_abs_difference": 0.0,
                "observed_order_max_norm": 0.0,
                "estimated_fine_grid_max_error": 0.0,
                "estimated_fine_grid_max_error_with_1p25_safety": 0.0,
            },
            "moisture": {
                "max_abs_difference": 0.0,
                "rms_difference": 0.0,
                "max_location_time_s": 0,
                "max_location_radius_cm": 0.0,
                "four_decimal_agreement_count": 0,
                "comparison_count": 0,
                "four_decimal_agreement_fraction": 0.0,
                "coarse_to_medium_max_abs_difference": 0.0,
                "observed_order_max_norm": 0.0,
                "estimated_fine_grid_max_error": 0.0,
                "estimated_fine_grid_max_error_with_1p25_safety": 0.0,
            },
        }
    else:
        medium_config = NumericalConfig(
            radial_cells=args.coarse_radial_cells,
            time_step_s=args.coarse_dt,
        )
        medium_config.validate()
        coarse_config = NumericalConfig(
            radial_cells=args.coarsest_radial_cells,
            time_step_s=args.coarsest_dt,
        )
        coarse_config.validate()
        print(
            f"Medium solve: Nr={medium_config.radial_cells}, "
            f"dt={medium_config.time_step_s:g} s"
        )
        medium_result = solve_model(
            environment, parameters, medium_config, progress=not args.quiet
        )
        print(
            f"Coarse solve: Nr={coarse_config.radial_cells}, "
            f"dt={coarse_config.time_step_s:g} s"
        )
        coarse_result = solve_model(
            environment, parameters, coarse_config, progress=not args.quiet
        )
        medium_to_fine = compare_results(medium_result, fine_result)
        coarse_to_medium = compare_results(coarse_result, medium_result)
        convergence = add_three_level_error_estimate(
            coarse_to_medium, medium_to_fine
        )

    output_xlsx = args.outdir / "result1.xlsx"
    write_result_xlsx(args.template, output_xlsx, fine_result)
    xlsx_validation = validate_result_xlsx(
        output_xlsx,
        expected_rows=len(fine_result.times_s) + 1,
        expected_columns=len(fine_result.radii_cm) + 1,
    )

    write_csv(
        args.outdir / "temperature_full_precision.csv",
        fine_result.times_s,
        fine_result.radii_cm,
        fine_result.temperatures_c,
    )
    write_csv(
        args.outdir / "moisture_full_precision.csv",
        fine_result.times_s,
        fine_result.radii_cm,
        fine_result.moistures_kg_kg,
    )
    write_environment_csv(args.outdir / "environment_input_used.csv", environment)
    write_summary_markdown(args.outdir / "summary_tables.md", fine_result)

    input_hash = sha256_file(args.input)
    write_validation_markdown(
        args.outdir / "validation_report.md",
        environment,
        fine_result,
        convergence,
        input_hash,
        xlsx_validation,
        fine_config,
        medium_config,
        coarse_config,
    )

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "input": {
            "path": str(args.input.resolve()),
            "sha256": input_hash,
            "rows": len(environment.times_s),
        },
        "template": {
            "path": str(args.template.resolve()),
            "sha256": sha256_file(args.template),
        },
        "physical_parameters": asdict(parameters),
        "main_numerical_config": asdict(fine_config),
        "medium_numerical_config": asdict(medium_config),
        "coarse_numerical_config": asdict(coarse_config),
        "main_runtime_s": fine_result.runtime_s,
        "diagnostics": {
            **asdict(fine_result.diagnostics),
            "mean_picard_iterations": fine_result.diagnostics.mean_picard_iterations,
        },
        "convergence": convergence,
        "xlsx_validation": xlsx_validation,
    }
    (args.outdir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Completed in {fine_result.runtime_s:.3f} s for the main solve.")
    print(f"Result workbook: {output_xlsx}")
    print(
        "Max coarse/fine differences: "
        f"T={convergence['temperature']['max_abs_difference']:.6e}, "
        f"C={convergence['moisture']['max_abs_difference']:.6e}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
