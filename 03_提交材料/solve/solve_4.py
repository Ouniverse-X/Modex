#!/usr/bin/env python3
"""Solve Problem A, Question 4 on a shrinking cylindrical domain.

The primary model uses the normalized material coordinate xi=r/R(t).  An
affine radial solid velocity makes the material derivative equal to the time
derivative at fixed xi.  A conservative node-centred finite-volume method is
combined with SciPy's sparse implicit BDF integrator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
import warnings
from copy import copy
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import scipy
from openpyxl import load_workbook
from openpyxl.styles import Alignment
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.sparse import csr_matrix, lil_matrix


@dataclass(frozen=True)
class PhysicalParameters:
    heat_transfer_w_m2_k: float = 25.0
    mass_transfer_m_s: float = 8.0e-7
    initial_temperature_c: float = 28.0
    initial_moisture_kg_kg: float = 2.55
    drying_threshold_kg_kg: float = 0.15
    density_intercept_kg_m3: float = 760.0
    density_moisture_slope_kg_m3: float = 90.0
    heat_capacity_intercept_j_kg_k: float = 1850.0
    heat_capacity_moisture_scale_j_kg_k: float = 2150.0
    conductivity_intercept_w_m_k: float = 0.12
    conductivity_moisture_scale_w_m_k: float = 0.20
    diffusivity_prefactor_m2_s: float = 4.2e-4
    diffusivity_moisture_exponent: float = 0.30
    diffusivity_temperature_exponent_k: float = 3850.0

    def evaluate_properties(
        self, moisture: np.ndarray, temperature_c: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        moisture = np.asarray(moisture, dtype=float)
        temperature_c = np.asarray(temperature_c, dtype=float)
        if moisture.shape != temperature_c.shape:
            raise ValueError("temperature and moisture shapes differ")
        if not np.all(np.isfinite(moisture)) or np.any(moisture <= 0.0):
            raise ValueError("moisture must be positive and finite")
        temperature_k = temperature_c + 273.15
        if not np.all(np.isfinite(temperature_k)) or np.any(temperature_k <= 0.0):
            raise ValueError("absolute temperature must be positive and finite")
        ratio = moisture / (moisture + 1.0)
        density = self.density_intercept_kg_m3 + self.density_moisture_slope_kg_m3 * moisture
        heat_capacity = (
            self.heat_capacity_intercept_j_kg_k
            + self.heat_capacity_moisture_scale_j_kg_k * ratio
        )
        conductivity = (
            self.conductivity_intercept_w_m_k
            + self.conductivity_moisture_scale_w_m_k * ratio
        )
        diffusivity = self.diffusivity_prefactor_m2_s * np.exp(
            -self.diffusivity_moisture_exponent / moisture
            -self.diffusivity_temperature_exponent_k / temperature_k
        )
        arrays = (density, heat_capacity, conductivity, diffusivity)
        if any(np.any(a <= 0.0) or not np.all(np.isfinite(a)) for a in arrays):
            raise ValueError("an Appendix 4 material property is invalid")
        return arrays


@dataclass(frozen=True)
class NumericalConfig:
    radial_cells: int = 760
    output_interval_s: float = 60.0
    reference_surface_layer: float = 0.1
    surface_refinement_factor: int = 10
    relative_tolerance: float = 2.0e-9
    temperature_absolute_tolerance: float = 2.0e-9
    moisture_absolute_tolerance: float = 2.0e-11
    initial_maximum_step_s: float = 5.0
    drying_maximum_step_s: float = 30.0
    first_step_s: float = 1.0e-4
    environment_data_end_s: float = 14400.0
    drying_segment_s: float = 21600.0
    maximum_duration_s: float = 14.0 * 86400.0

    def validate(self) -> None:
        if self.radial_cells < 38 or self.radial_cells % 19 != 0:
            raise ValueError("radial_cells must be a multiple of 19 and at least 38")
        if not 0.0 < self.reference_surface_layer < 1.0:
            raise ValueError("reference_surface_layer must lie in (0,1)")
        if self.surface_refinement_factor < 1:
            raise ValueError("surface_refinement_factor must be positive")
        positive = (
            self.output_interval_s,
            self.relative_tolerance,
            self.temperature_absolute_tolerance,
            self.moisture_absolute_tolerance,
            self.initial_maximum_step_s,
            self.drying_maximum_step_s,
            self.first_step_s,
            self.environment_data_end_s,
            self.drying_segment_s,
            self.maximum_duration_s,
        )
        if any(not math.isfinite(x) or x <= 0.0 for x in positive):
            raise ValueError("all numerical time and tolerance parameters must be positive")
        for value in (
            self.environment_data_end_s,
            self.drying_segment_s,
            self.maximum_duration_s,
        ):
            q = value / self.output_interval_s
            if abs(q - round(q)) > 1.0e-12:
                raise ValueError("segment times must be divisible by output_interval_s")


@dataclass(frozen=True)
class EnvironmentData:
    times_s: np.ndarray
    temperatures_c: np.ndarray
    moistures_kg_kg: np.ndarray
    extension_strategy: str = "last_point"
    extension_temperature_c: float | None = None
    extension_moisture_kg_kg: float | None = None

    def validate(self) -> None:
        if self.times_s.ndim != 1 or self.times_s.size < 2:
            raise ValueError("environment needs at least two rows")
        if not (
            self.times_s.shape
            == self.temperatures_c.shape
            == self.moistures_kg_kg.shape
        ):
            raise ValueError("environment columns have inconsistent shapes")
        if not all(
            np.all(np.isfinite(x))
            for x in (self.times_s, self.temperatures_c, self.moistures_kg_kg)
        ):
            raise ValueError("environment contains non-finite values")
        if np.any(np.diff(self.times_s) <= 0.0):
            raise ValueError("environment times must increase strictly")
        if self.times_s[0] != 0.0 or np.any(self.moistures_kg_kg <= 0.0):
            raise ValueError("invalid environment initial time or moisture")
        if self.extension_strategy not in {"last_point", "fixed"}:
            raise ValueError("unknown environment extension strategy")
        if self.extension_strategy == "fixed":
            if self.extension_temperature_c is None or not math.isfinite(
                self.extension_temperature_c
            ):
                raise ValueError("fixed environment temperature must be finite")
            if (
                self.extension_moisture_kg_kg is None
                or not math.isfinite(self.extension_moisture_kg_kg)
                or self.extension_moisture_kg_kg <= 0.0
            ):
                raise ValueError("fixed environment moisture must be positive and finite")

    def interpolate(self, t_s: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        temperature = np.interp(
            t_s,
            self.times_s,
            self.temperatures_c,
            left=self.temperatures_c[0],
            right=self.temperatures_c[-1],
        )
        moisture = np.interp(
            t_s,
            self.times_s,
            self.moistures_kg_kg,
            left=self.moistures_kg_kg[0],
            right=self.moistures_kg_kg[-1],
        )
        if self.extension_strategy == "fixed":
            after_measurement = np.asarray(t_s) >= self.times_s[-1]
            temperature = np.where(
                after_measurement, self.extension_temperature_c, temperature
            )
            moisture = np.where(
                after_measurement, self.extension_moisture_kg_kg, moisture
            )
        return np.asarray(temperature), np.asarray(moisture)


@dataclass(frozen=True)
class RadiusData:
    times_s: np.ndarray
    radii_m: np.ndarray

    def validate(self) -> None:
        if self.times_s.ndim != 1 or self.times_s.size < 2:
            raise ValueError("radius data need at least two rows")
        if self.times_s.shape != self.radii_m.shape:
            raise ValueError("radius columns have inconsistent shapes")
        if not np.all(np.isfinite(self.times_s)) or not np.all(np.isfinite(self.radii_m)):
            raise ValueError("radius data contain non-finite values")
        if self.times_s[0] != 0.0 or np.any(np.diff(self.times_s) <= 0.0):
            raise ValueError("radius times must start at zero and increase strictly")
        if np.any(self.radii_m <= 0.0) or np.any(np.diff(self.radii_m) > 1.0e-14):
            raise ValueError("radii must be positive and non-increasing")

    def interpolate(self, t_s: float | np.ndarray) -> np.ndarray:
        return np.asarray(
            np.interp(
                t_s,
                self.times_s,
                self.radii_m,
                left=self.radii_m[0],
                right=self.radii_m[-1],
            )
        )


@dataclass
class SolverDiagnostics:
    runtime_s: float
    function_evaluations: int
    jacobian_evaluations: int
    lu_decompositions: int
    integration_segments: int
    returned_time_points: int
    min_temperature_c: float
    max_temperature_c: float
    min_moisture_kg_kg: float
    max_moisture_kg_kg: float
    min_density_kg_m3: float
    max_density_kg_m3: float
    min_heat_capacity_j_kg_k: float
    max_heat_capacity_j_kg_k: float
    min_conductivity_w_m_k: float
    max_conductivity_w_m_k: float
    min_diffusivity_m2_s: float
    max_diffusivity_m2_s: float
    max_output_interval_balance_abs: float
    max_output_interval_balance_rel: float
    maximum_location_was_center: bool


@dataclass
class SimulationResult:
    times_s: np.ndarray
    xi_samples: np.ndarray
    temperatures_c: np.ndarray
    moistures_kg_kg: np.ndarray
    physical_radii_cm: np.ndarray
    physical_moistures_kg_kg: np.ndarray
    surface_moistures_kg_kg: np.ndarray
    maximum_moistures_kg_kg: np.ndarray
    radii_m: np.ndarray
    drying_crossing_s: float
    first_strict_output_s: int
    diagnostics: SolverDiagnostics
    config: NumericalConfig


def density_shrinkage_consistency_from_field(
    xi: np.ndarray,
    moisture_kg_kg: np.ndarray,
    initial_radius_m: float,
    current_radius_m: float,
    parameters: PhysicalParameters,
    evaluation_time_s: float,
    cylinder_length_m: float = 0.25,
) -> dict[str, float | str]:
    """Test one optional physical interpretation of Appendix 4 density.

    The primary PDE treats dry-basis moisture as a material scalar and uses
    Appendix 4 density only in the thermal storage coefficient. This
    diagnostic asks a separate counterfactual question: if that density were
    instead the actual wet bulk density, would rho/(1+C) and the prescribed
    affine radius conserve dry-solid mass? A non-zero discrepancy is a model
    structure inconsistency under that interpretation, not a discretization
    residual of the solved PDE.
    """

    xi = np.asarray(xi, dtype=float)
    moisture = np.asarray(moisture_kg_kg, dtype=float)
    if xi.ndim != 1 or moisture.shape != xi.shape or xi.size < 2:
        raise ValueError("xi and moisture must be one-dimensional arrays of equal size")
    if not np.all(np.isfinite(xi)) or not np.all(np.isfinite(moisture)):
        raise ValueError("density-shrinkage diagnostic inputs must be finite")
    if abs(float(xi[0])) > 1.0e-14 or abs(float(xi[-1]) - 1.0) > 1.0e-14:
        raise ValueError("xi must span the closed material interval [0,1]")
    if np.any(np.diff(xi) <= 0.0) or np.any(moisture <= 0.0):
        raise ValueError("xi must increase strictly and moisture must be positive")
    if initial_radius_m <= 0.0 or current_radius_m <= 0.0:
        raise ValueError("radii must be positive")
    if cylinder_length_m <= 0.0:
        raise ValueError("cylinder length must be positive")

    initial_moisture = parameters.initial_moisture_kg_kg
    initial_wet_density = (
        parameters.density_intercept_kg_m3
        + parameters.density_moisture_slope_kg_m3 * initial_moisture
    )
    initial_dry_density = initial_wet_density / (1.0 + initial_moisture)
    wet_density = (
        parameters.density_intercept_kg_m3
        + parameters.density_moisture_slope_kg_m3 * moisture
    )
    dry_density = wet_density / (1.0 + moisture)

    # For a cylinder, the volume average of f(xi) is
    # 2*integral_0^1 f(xi)*xi dxi. Write the trapezoidal rule explicitly to
    # support both NumPy 1.x and 2.x without relying on a renamed API.
    weighted = dry_density * xi
    volume_averaged_dry_density = 2.0 * float(
        np.sum(0.5 * (weighted[:-1] + weighted[1:]) * np.diff(xi))
    )
    initial_volume = math.pi * initial_radius_m**2 * cylinder_length_m
    current_volume = math.pi * current_radius_m**2 * cylinder_length_m
    initial_dry_mass = initial_volume * initial_dry_density
    current_dry_mass = current_volume * volume_averaged_dry_density
    dry_mass_ratio = current_dry_mass / initial_dry_mass
    jacobian = (current_radius_m / initial_radius_m) ** 2
    local_reference_mass_ratio = jacobian * dry_density / initial_dry_density
    dry_mass_conserving_radius = initial_radius_m * math.sqrt(
        initial_dry_density / volume_averaged_dry_density
    )
    relative_change = dry_mass_ratio - 1.0
    tolerance = 1.0e-6

    return {
        "interpretation": "appendix_4_density_as_actual_wet_bulk_density",
        "status": (
            "consistent_within_tolerance"
            if abs(relative_change) <= tolerance
            else "inconsistent_with_prescribed_affine_shrinkage"
        ),
        "evaluation_time_s": float(evaluation_time_s),
        "cylinder_length_m": float(cylinder_length_m),
        "initial_radius_m": float(initial_radius_m),
        "current_radius_m": float(current_radius_m),
        "volume_jacobian": float(jacobian),
        "initial_wet_bulk_density_kg_m3": float(initial_wet_density),
        "initial_dry_solid_density_proxy_kg_m3": float(initial_dry_density),
        "current_volume_averaged_dry_solid_density_proxy_kg_m3": float(
            volume_averaged_dry_density
        ),
        "initial_dry_mass_proxy_kg": float(initial_dry_mass),
        "current_dry_mass_proxy_kg": float(current_dry_mass),
        "dry_mass_proxy_ratio": float(dry_mass_ratio),
        "dry_mass_proxy_relative_change": float(relative_change),
        "dry_mass_proxy_drop_percent": float(-100.0 * relative_change),
        "local_dry_mass_jacobian_ratio_min": float(
            np.min(local_reference_mass_ratio)
        ),
        "local_dry_mass_jacobian_ratio_max": float(
            np.max(local_reference_mass_ratio)
        ),
        "dry_mass_conserving_radius_m": float(dry_mass_conserving_radius),
        "radius_difference_actual_minus_consistent_m": float(
            current_radius_m - dry_mass_conserving_radius
        ),
    }


def density_shrinkage_consistency_at_time(
    result: SimulationResult,
    parameters: PhysicalParameters,
    evaluation_time_s: int,
    cylinder_length_m: float = 0.25,
) -> dict[str, float | str]:
    """Evaluate the wet-density consistency diagnostic at an output time."""

    matches = np.flatnonzero(result.times_s == evaluation_time_s)
    if matches.size != 1:
        raise ValueError("density-shrinkage diagnostic time is absent or duplicated")
    initial_matches = np.flatnonzero(result.times_s == 0)
    if initial_matches.size != 1:
        raise ValueError("simulation result does not contain exactly one initial time")
    index = int(matches[0])
    initial_index = int(initial_matches[0])
    return density_shrinkage_consistency_from_field(
        result.xi_samples,
        result.moistures_kg_kg[index],
        float(result.radii_m[initial_index]),
        float(result.radii_m[index]),
        parameters,
        float(evaluation_time_s),
        cylinder_length_m,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_three_columns(path: Path) -> np.ndarray:
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        rows = list(workbook.active.iter_rows(values_only=True))
    finally:
        workbook.close()
    values = []
    for row_number, row in enumerate(rows[1:], start=2):
        if len(row) < 3 or any(x is None for x in row[:3]):
            raise ValueError(f"missing input value in {path}, row {row_number}")
        values.append(tuple(float(x) for x in row[:3]))
    return np.asarray(values, dtype=float)


def load_environment(path: Path) -> EnvironmentData:
    values = _read_three_columns(path)
    data = EnvironmentData(values[:, 0], values[:, 1], values[:, 2])
    data.validate()
    return data


def load_radius(path: Path) -> RadiusData:
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        rows = list(workbook.active.iter_rows(values_only=True))
    finally:
        workbook.close()
    values = []
    for row_number, row in enumerate(rows[1:], start=2):
        if len(row) < 2 or any(x is None for x in row[:2]):
            raise ValueError(f"missing radius value in row {row_number}")
        values.append((float(row[0]), 0.01 * float(row[1])))
    array = np.asarray(values, dtype=float)
    data = RadiusData(array[:, 0], array[:, 1])
    data.validate()
    return data


def make_reference_grid(
    radial_cells: int,
    surface_layer: float = 0.1,
    refinement_factor: int = 10,
) -> np.ndarray:
    bulk_length = 1.0 - surface_layer
    effective_length = bulk_length + refinement_factor * surface_layer
    bulk_step = effective_length / radial_cells
    surface_step = bulk_step / refinement_factor
    bulk_cells_float = bulk_length / bulk_step
    surface_cells_float = surface_layer / surface_step
    bulk_cells = int(round(bulk_cells_float))
    surface_cells = int(round(surface_cells_float))
    if (
        abs(bulk_cells_float - bulk_cells) > 1.0e-10
        or abs(surface_cells_float - surface_cells) > 1.0e-10
        or bulk_cells + surface_cells != radial_cells
    ):
        raise ValueError("radial_cells is incompatible with the two-zone grid")
    bulk = np.arange(bulk_cells + 1, dtype=float) * bulk_step
    surface = bulk_length + np.arange(1, surface_cells + 1, dtype=float) * surface_step
    xi = np.concatenate((bulk, surface))
    xi[0] = 0.0
    xi[-1] = 1.0
    if xi.size != radial_cells + 1 or np.any(np.diff(xi) <= 0.0):
        raise AssertionError("invalid reference grid")
    return xi


def radial_storage_weights(xi: np.ndarray) -> np.ndarray:
    faces = 0.5 * (xi[:-1] + xi[1:])
    weights = np.empty_like(xi)
    weights[0] = 0.5 * faces[0] ** 2
    weights[1:-1] = 0.5 * (faces[1:] ** 2 - faces[:-1] ** 2)
    weights[-1] = 0.5 * (1.0 - faces[-1] ** 2)
    return weights


def harmonic_mean(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if np.any(left <= 0.0) or np.any(right <= 0.0):
        raise ValueError("transport coefficient is non-positive")
    return 2.0 * left * right / (left + right)


def make_jacobian_sparsity(node_count: int) -> csr_matrix:
    """Jacobian pattern for T, C and an auxiliary boundary-flux integral."""

    state_size = 2 * node_count + 1
    pattern = lil_matrix((state_size, state_size), dtype=np.int8)
    for i in range(node_count):
        for j in range(max(0, i - 1), min(node_count, i + 2)):
            pattern[i, j] = 1
            pattern[i, node_count + j] = 1
            pattern[node_count + i, j] = 1
            pattern[node_count + i, node_count + j] = 1
    pattern[-1, 2 * node_count - 1] = 1
    return pattern.tocsr()


class MovingBoundarySystem:
    def __init__(
        self,
        environment: EnvironmentData,
        radius_data: RadiusData,
        parameters: PhysicalParameters,
        xi: np.ndarray,
    ) -> None:
        self.environment = environment
        self.radius_data = radius_data
        self.parameters = parameters
        self.xi = xi
        self.node_count = xi.size
        self.faces = 0.5 * (xi[:-1] + xi[1:])
        self.spacing = np.diff(xi)
        self.weights = radial_storage_weights(xi)

    def transport_net(
        self,
        field: np.ndarray,
        gamma: np.ndarray,
        radius_m: float,
        boundary_transfer: float,
        external_value: float,
    ) -> np.ndarray:
        gamma_faces = harmonic_mean(gamma[:-1], gamma[1:])
        internal_flux = self.faces * gamma_faces * np.diff(field) / self.spacing
        net = np.empty_like(field)
        net[0] = internal_flux[0]
        net[1:-1] = internal_flux[1:] - internal_flux[:-1]
        net[-1] = (
            radius_m * boundary_transfer * (external_value - field[-1])
            - internal_flux[-1]
        )
        return net

    def __call__(self, t_s: float, state: np.ndarray) -> np.ndarray:
        n = self.node_count
        temperature = state[:n]
        moisture = state[n : 2 * n]
        density, heat_capacity, conductivity, diffusivity = (
            self.parameters.evaluate_properties(moisture, temperature)
        )
        ambient_temperature, ambient_moisture = self.environment.interpolate(t_s)
        radius_m = float(self.radius_data.interpolate(t_s))
        heat_net = self.transport_net(
            temperature,
            conductivity,
            radius_m,
            self.parameters.heat_transfer_w_m2_k,
            float(ambient_temperature),
        )
        moisture_net = self.transport_net(
            moisture,
            diffusivity,
            radius_m,
            self.parameters.mass_transfer_m_s,
            float(ambient_moisture),
        )
        radius_squared = radius_m * radius_m
        temperature_rate = heat_net / (
            radius_squared * self.weights * density * heat_capacity
        )
        moisture_rate = moisture_net / (radius_squared * self.weights)
        boundary_inventory_rate = (
            self.parameters.mass_transfer_m_s
            / radius_m
            * (float(ambient_moisture) - moisture[-1])
        )
        return np.concatenate(
            (temperature_rate, moisture_rate, np.asarray([boundary_inventory_rate]))
        )


def interpolate_rows(
    source_x: np.ndarray, rows: np.ndarray, target_x: np.ndarray
) -> np.ndarray:
    output = np.empty((rows.shape[0], target_x.size), dtype=float)
    for i, row in enumerate(rows):
        output[i] = np.interp(target_x, source_x, row)
    return output


def extract_physical_moistures(
    xi: np.ndarray,
    moisture_rows: np.ndarray,
    radii_m: np.ndarray,
    fixed_radii_cm: np.ndarray,
) -> np.ndarray:
    output = np.full((moisture_rows.shape[0], fixed_radii_cm.size), np.nan)
    fixed_m = fixed_radii_cm * 0.01
    for i, (row, radius_m) in enumerate(zip(moisture_rows, radii_m)):
        valid = fixed_m < radius_m - 1.0e-13
        output[i, valid] = np.interp(fixed_m[valid] / radius_m, xi, row)
    return output


def solve_until_dry(
    environment: EnvironmentData,
    radius_data: RadiusData,
    parameters: PhysicalParameters,
    config: NumericalConfig,
    progress_label: str | None = None,
    integration_method: str = "BDF",
) -> SimulationResult:
    if integration_method not in {"BDF", "Radau"}:
        raise ValueError("integration_method must be 'BDF' or 'Radau'")
    config.validate()
    xi = make_reference_grid(
        config.radial_cells,
        config.reference_surface_layer,
        config.surface_refinement_factor,
    )
    system = MovingBoundarySystem(environment, radius_data, parameters, xi)
    node_count = xi.size
    # The sparsity graph depends only on the fixed reference grid.  Construct
    # it once; rebuilding the same matrix in every six-hour integration
    # segment is pure overhead and changes no solver semantics.
    jacobian_sparsity = make_jacobian_sparsity(node_count)
    state = np.concatenate(
        (
            np.full(node_count, parameters.initial_temperature_c),
            np.full(node_count, parameters.initial_moisture_kg_kg),
            np.asarray([0.0]),
        )
    )
    absolute_tolerance = np.concatenate(
        (
            np.full(node_count, config.temperature_absolute_tolerance),
            np.full(node_count, config.moisture_absolute_tolerance),
            np.asarray([config.moisture_absolute_tolerance]),
        )
    )
    xi_samples = np.linspace(0.0, 1.0, 101)
    physical_radii_cm = np.arange(20, dtype=float) * 0.1
    time_chunks: list[np.ndarray] = []
    temperature_chunks: list[np.ndarray] = []
    moisture_chunks: list[np.ndarray] = []
    physical_chunks: list[np.ndarray] = []
    surface_chunks: list[np.ndarray] = []
    maximum_chunks: list[np.ndarray] = []
    radius_chunks: list[np.ndarray] = []
    inventory_chunks: list[np.ndarray] = []
    boundary_integral_chunks: list[np.ndarray] = []

    minima = np.full(6, np.inf)
    maxima = np.full(6, -np.inf)
    nfev = njev = nlu = segments = 0
    maximum_at_center = True
    crossing_s: float | None = None
    started = time.perf_counter()
    current_time = 0.0

    while current_time < config.maximum_duration_s and crossing_s is None:
        if current_time < config.environment_data_end_s:
            segment_end = min(config.environment_data_end_s, config.maximum_duration_s)
            maximum_step = config.initial_maximum_step_s
        else:
            segment_end = min(
                current_time + config.drying_segment_s,
                config.maximum_duration_s,
            )
            maximum_step = config.drying_maximum_step_s
        eval_times = np.arange(
            current_time,
            segment_end + 0.5 * config.output_interval_s,
            config.output_interval_s,
        )
        if progress_label:
            print(
                f"[{progress_label}] {current_time / 3600:.1f}–"
                f"{segment_end / 3600:.1f} h, max_step={maximum_step:g} s",
                flush=True,
            )
        # The last state is a passive flux integral. Its Jacobian column is
        # identically zero, so SciPy's finite-difference step controller can
        # harmlessly overflow its unused perturbation factor. Suppress only
        # those two narrowly scoped internal warnings.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="overflow encountered in multiply",
                category=RuntimeWarning,
                module=r"scipy\.integrate\._ivp\.common",
            )
            warnings.filterwarnings(
                "ignore",
                message="invalid value encountered in multiply",
                category=RuntimeWarning,
                module=r"scipy\.integrate\._ivp\.common",
            )
            solution = solve_ivp(
                system,
                (current_time, segment_end),
                state,
                method=integration_method,
                t_eval=eval_times,
                dense_output=True,
                rtol=config.relative_tolerance,
                atol=absolute_tolerance,
                jac_sparsity=jacobian_sparsity,
                max_step=maximum_step,
                first_step=min(config.first_step_s, segment_end - current_time),
            )
        if not solution.success:
            raise RuntimeError(
                f"{integration_method} integration failed: {solution.message}"
            )
        if not np.all(np.isfinite(solution.y)):
            raise RuntimeError(
                f"{integration_method} solution contains non-finite values"
            )
        temperatures_nodes = solution.y[:node_count].T
        moistures_nodes = solution.y[node_count : 2 * node_count].T
        boundary_integral = solution.y[2 * node_count]
        if np.min(moistures_nodes) <= 0.0:
            raise RuntimeError("computed non-positive moisture")

        densities, heat_capacities, conductivities, diffusivities = (
            parameters.evaluate_properties(
                moistures_nodes.ravel(), temperatures_nodes.ravel()
            )
        )
        tracked = (
            temperatures_nodes,
            moistures_nodes,
            densities,
            heat_capacities,
            conductivities,
            diffusivities,
        )
        for j, values in enumerate(tracked):
            minima[j] = min(minima[j], float(np.min(values)))
            maxima[j] = max(maxima[j], float(np.max(values)))

        local_maximum = np.max(moistures_nodes, axis=1)
        maximum_at_center = maximum_at_center and bool(
            np.all(local_maximum - moistures_nodes[:, 0] <= 1.0e-10)
        )
        crossing_candidates = np.flatnonzero(
            (local_maximum[:-1] > parameters.drying_threshold_kg_kg)
            & (local_maximum[1:] <= parameters.drying_threshold_kg_kg)
        )
        if crossing_candidates.size:
            j = int(crossing_candidates[0])

            def event_value(t_s: float) -> float:
                dense_state = solution.sol(t_s)
                return (
                    float(np.max(dense_state[node_count : 2 * node_count]))
                    - parameters.drying_threshold_kg_kg
                )

            crossing_s = float(
                brentq(
                    event_value,
                    float(solution.t[j]),
                    float(solution.t[j + 1]),
                    xtol=1.0e-7,
                    rtol=1.0e-13,
                )
            )

        radii = radius_data.interpolate(solution.t)
        sampled_temperature = interpolate_rows(xi, temperatures_nodes, xi_samples)
        sampled_moisture = interpolate_rows(xi, moistures_nodes, xi_samples)
        physical_moisture = extract_physical_moistures(
            xi, moistures_nodes, radii, physical_radii_cm
        )
        inventory = moistures_nodes @ system.weights
        start_index = 0 if segments == 0 else 1
        sl = slice(start_index, None)
        time_chunks.append(solution.t[sl].copy())
        temperature_chunks.append(sampled_temperature[sl].copy())
        moisture_chunks.append(sampled_moisture[sl].copy())
        physical_chunks.append(physical_moisture[sl].copy())
        surface_chunks.append(moistures_nodes[sl, -1].copy())
        maximum_chunks.append(local_maximum[sl].copy())
        radius_chunks.append(radii[sl].copy())
        inventory_chunks.append(inventory[sl].copy())
        boundary_integral_chunks.append(boundary_integral[sl].copy())

        nfev += int(solution.nfev)
        njev += int(solution.njev)
        nlu += int(solution.nlu)
        segments += 1
        state = solution.y[:, -1].copy()
        current_time = segment_end

    runtime_s = time.perf_counter() - started
    if crossing_s is None:
        raise RuntimeError(
            f"drying threshold was not reached within {config.maximum_duration_s / 3600:g} h"
        )

    times = np.concatenate(time_chunks)
    temperatures = np.vstack(temperature_chunks)
    moistures = np.vstack(moisture_chunks)
    physical_moistures = np.vstack(physical_chunks)
    surface_moistures = np.concatenate(surface_chunks)
    maximum_moistures = np.concatenate(maximum_chunks)
    radii = np.concatenate(radius_chunks)
    inventories = np.concatenate(inventory_chunks)
    boundary_integrals = np.concatenate(boundary_integral_chunks)
    strict_indices = np.flatnonzero(
        (times > 0.0)
        & (maximum_moistures < parameters.drying_threshold_kg_kg)
    )
    if strict_indices.size == 0:
        raise RuntimeError("post-crossing segment did not contain a strict output time")
    first_strict_output_s = int(round(float(times[int(strict_indices[0])])))

    residual = inventories - inventories[0] - boundary_integrals
    residual_scale = np.maximum.reduce(
        (
            np.abs(inventories - inventories[0]),
            np.abs(boundary_integrals),
            np.full_like(residual, 1.0e-30),
        )
    )

    diagnostics = SolverDiagnostics(
        runtime_s=runtime_s,
        function_evaluations=nfev,
        jacobian_evaluations=njev,
        lu_decompositions=nlu,
        integration_segments=segments,
        returned_time_points=int(times.size),
        min_temperature_c=minima[0],
        max_temperature_c=maxima[0],
        min_moisture_kg_kg=minima[1],
        max_moisture_kg_kg=maxima[1],
        min_density_kg_m3=minima[2],
        max_density_kg_m3=maxima[2],
        min_heat_capacity_j_kg_k=minima[3],
        max_heat_capacity_j_kg_k=maxima[3],
        min_conductivity_w_m_k=minima[4],
        max_conductivity_w_m_k=maxima[4],
        min_diffusivity_m2_s=minima[5],
        max_diffusivity_m2_s=maxima[5],
        max_output_interval_balance_abs=float(np.max(np.abs(residual))),
        max_output_interval_balance_rel=float(
            np.max(np.abs(residual[1:]) / residual_scale[1:])
        ),
        maximum_location_was_center=maximum_at_center,
    )
    if progress_label:
        print(
            f"[{progress_label}] crossing={crossing_s / 3600:.6f} h, "
            f"runtime={runtime_s:.2f} s, nfev={nfev}, nlu={nlu}",
            flush=True,
        )
    return SimulationResult(
        times_s=np.rint(times).astype(int),
        xi_samples=xi_samples,
        temperatures_c=temperatures,
        moistures_kg_kg=moistures,
        physical_radii_cm=physical_radii_cm,
        physical_moistures_kg_kg=physical_moistures,
        surface_moistures_kg_kg=surface_moistures,
        maximum_moistures_kg_kg=maximum_moistures,
        radii_m=radii,
        drying_crossing_s=crossing_s,
        first_strict_output_s=first_strict_output_s,
        diagnostics=diagnostics,
        config=config,
    )


def common_field_comparison(a: SimulationResult, b: SimulationResult) -> dict:
    common_end = min(int(a.times_s[-1]), int(b.times_s[-1]))
    count = common_end // int(a.config.output_interval_s) + 1
    if not np.array_equal(a.times_s[:count], b.times_s[:count]):
        raise ValueError("comparison time grids do not align")
    difference = np.abs(a.moistures_kg_kg[:count] - b.moistures_kg_kg[:count])
    flat_index = int(np.argmax(difference))
    time_index, space_index = np.unravel_index(flat_index, difference.shape)
    return {
        "max_abs_difference": float(difference[time_index, space_index]),
        "rms_difference": float(np.sqrt(np.mean(difference * difference))),
        "max_location_time_s": int(a.times_s[time_index]),
        "max_location_xi": float(a.xi_samples[space_index]),
        "common_end_time_s": common_end,
        "drying_crossing_difference_s": abs(a.drying_crossing_s - b.drying_crossing_s),
    }


def spatial_convergence(
    coarse: SimulationResult,
    medium: SimulationResult,
    fine: SimulationResult,
) -> dict:
    cm = common_field_comparison(coarse, medium)
    mf = common_field_comparison(medium, fine)
    if cm["max_abs_difference"] > 0.0 and mf["max_abs_difference"] > 0.0:
        order = math.log(cm["max_abs_difference"] / mf["max_abs_difference"], 2.)
        if order > 0.0:
            estimate = mf["max_abs_difference"] / (2.0**order - 1.0)
        else:
            estimate = math.inf
    else:
        order = math.nan
        estimate = math.nan
    return {
        "coarse_to_medium": cm,
        "medium_to_fine": mf,
        "observed_order": order,
        "estimated_fine_grid_max_error": estimate,
        "estimated_fine_grid_max_error_with_1p25_safety": 1.25 * estimate,
        "crossing_times_s": {
            "coarse": coarse.drying_crossing_s,
            "medium": medium.drying_crossing_s,
            "fine": fine.drying_crossing_s,
        },
    }


def choose_safe_end_time(
    fine: SimulationResult,
    numerical_error_bound: float,
    threshold: float,
) -> int:
    # The extra half unit in the fourth decimal place ensures the submitted
    # table visibly rounds below 0.1500 as well as satisfying the full-precision
    # physical threshold after the numerical error allowance is added.
    displayed_threshold = threshold - 0.5e-4
    candidates = np.flatnonzero(
        (fine.times_s > 0)
        & (
            fine.maximum_moistures_kg_kg + numerical_error_bound
            < displayed_threshold
        )
    )
    if candidates.size == 0:
        raise RuntimeError("stored post-crossing interval is too short for safety margin")
    return int(fine.times_s[int(candidates[0])])


def write_result_xlsx(
    template_path: Path,
    output_path: Path,
    result: SimulationResult,
    end_time_s: int,
) -> None:
    workbook = load_workbook(template_path)
    try:
        worksheet = workbook.active
        header_style = copy(worksheet["B1"]._style)
        surface_header_style = copy(worksheet["F1"]._style)
        time_style = copy(worksheet["A2"]._style)
        value_style = copy(worksheet["B2"]._style)
        if worksheet.max_row > 1:
            worksheet.delete_rows(2, worksheet.max_row - 1)
        if worksheet.max_column > 1:
            worksheet.delete_cols(2, worksheet.max_column - 1)
        worksheet["A1"] = "时间\\到药材中心的距离"
        for column, radius_cm in enumerate(result.physical_radii_cm, start=2):
            cell = worksheet.cell(1, column, round(float(radius_cm), 1))
            cell._style = copy(header_style)
        surface_column = 2 + result.physical_radii_cm.size
        cell = worksheet.cell(1, surface_column, "药材表面")
        cell._style = copy(surface_header_style)

        selected = np.flatnonzero((result.times_s > 0) & (result.times_s <= end_time_s))
        for output_row, source_index in enumerate(selected, start=2):
            time_cell = worksheet.cell(output_row, 1, int(result.times_s[source_index]))
            time_cell._style = copy(time_style)
            for j, value in enumerate(result.physical_moistures_kg_kg[source_index], start=2):
                cell = worksheet.cell(
                    output_row,
                    j,
                    None if np.isnan(value) else round(float(value), 4),
                )
                cell._style = copy(value_style)
                cell.number_format = "0.0000"
                cell.alignment = Alignment(horizontal="center", vertical="center")
            cell = worksheet.cell(
                output_row,
                surface_column,
                round(float(result.surface_moistures_kg_kg[source_index]), 4),
            )
            cell._style = copy(value_style)
            cell.number_format = "0.0000"
            cell.alignment = Alignment(horizontal="center", vertical="center")
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = f"A1:V{selected.size + 1}"
        worksheet.column_dimensions["A"].width = 25
        for column in range(2, surface_column + 1):
            letter = worksheet.cell(1, column).column_letter
            worksheet.column_dimensions[letter].width = 12
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        workbook.save(temporary)
        os.replace(temporary, output_path)
    finally:
        workbook.close()


def validate_result_xlsx(
    path: Path,
    expected: SimulationResult,
    end_time_s: int,
) -> dict:
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        worksheet = workbook.active
        expected_rows = end_time_s // 60 + 1
        if worksheet.max_row != expected_rows or worksheet.max_column != 22:
            raise ValueError(
                f"result4 dimensions {worksheet.max_row}x{worksheet.max_column}, "
                f"expected {expected_rows}x22"
            )
        headers = [worksheet.cell(1, c).value for c in range(2, 22)]
        if any(abs(float(a) - 0.1 * i) > 1.0e-12 for i, a in enumerate(headers)):
            raise ValueError("unexpected fixed-radius headers")
        if worksheet.cell(1, 22).value != "药材表面":
            raise ValueError("missing moving-surface column")
        max_error = 0.0
        checked = 0
        for row_index, row in enumerate(
            worksheet.iter_rows(min_row=2, values_only=True), start=1
        ):
            if int(row[0]) != 60 * row_index:
                raise ValueError(f"unexpected time in row {row_index + 1}")
            source = int(np.where(expected.times_s == 60 * row_index)[0][0])
            wanted = expected.physical_moistures_kg_kg[source]
            for value, target in zip(row[1:21], wanted):
                if np.isnan(target):
                    if value is not None:
                        raise ValueError("an outside-domain cell is not blank")
                else:
                    max_error = max(max_error, abs(float(value) - round(float(target), 4)))
                    checked += 1
            surface_error = abs(
                float(row[21]) - round(float(expected.surface_moistures_kg_kg[source]), 4)
            )
            max_error = max(max_error, surface_error)
            checked += 1
        if max_error > 5.0e-13:
            raise ValueError(f"Excel write mismatch: {max_error:.3e}")
        return {
            "sheet_name": worksheet.title,
            "dimension": worksheet.calculate_dimension(),
            "checked_numerical_values": checked,
            "max_error_against_rounded_solver_output": max_error,
        }
    finally:
        workbook.close()


def write_full_precision_csv(
    path: Path,
    result: SimulationResult,
    end_time_s: int,
    material_coordinates: bool,
) -> None:
    selected = np.flatnonzero((result.times_s > 0) & (result.times_s <= end_time_s))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        if material_coordinates:
            writer.writerow(["time_s"] + [f"xi={x:.2f}" for x in result.xi_samples])
            for i in selected:
                writer.writerow(
                    [int(result.times_s[i])]
                    + [f"{value:.12f}" for value in result.moistures_kg_kg[i]]
                )
        else:
            writer.writerow(
                ["time_s"]
                + [f"r={x:.1f}cm" for x in result.physical_radii_cm]
                + ["surface", "radius_cm"]
            )
            for i in selected:
                values = [
                    "" if np.isnan(value) else f"{value:.12f}"
                    for value in result.physical_moistures_kg_kg[i]
                ]
                writer.writerow(
                    [int(result.times_s[i])]
                    + values
                    + [
                        f"{result.surface_moistures_kg_kg[i]:.12f}",
                        f"{100.0 * result.radii_m[i]:.12f}",
                    ]
                )


def markdown_value(value: float) -> str:
    return "—" if np.isnan(value) else f"{value:.4f}"


def write_summary_markdown(
    path: Path,
    result: SimulationResult,
    crossing_s: float,
    safe_end_s: int,
    numerical_error_bound: float,
) -> None:
    regular_times = list(range(21600, safe_end_s, 21600))
    table_times = regular_times + [safe_end_s]
    header = "| 时间/h | 0 cm | 0.5 cm | 1.0 cm | 药材表面 | 当前半径/cm |"
    separator = "|---:|---:|---:|---:|---:|---:|"
    rows = [header, separator]
    for t_s in table_times:
        index = int(np.where(result.times_s == t_s)[0][0])
        values = result.physical_moistures_kg_kg[index]
        label = f"{t_s / 3600:.4f}" if t_s == safe_end_s else f"{t_s / 3600:.0f}"
        rows.append(
            f"| {label} | {markdown_value(values[0])} | "
            f"{markdown_value(values[5])} | {markdown_value(values[10])} | "
            f"{result.surface_moistures_kg_kg[index]:.4f} | "
            f"{100.0 * result.radii_m[index]:.4f} |"
        )
    text = rf"""# 第四问计算结果

连续模型达到临界含水率的时刻为

$$
t_*= {crossing_s / 3600:.10f}\ \mathrm{{h}}.
$$

计入数值误差安全上界、60 s 输出间隔和四位小数显示裕量后，建议烘干时长为

$$
t_{{\mathrm{{end}}}}= {safe_end_s / 3600:.10f}\ \mathrm{{h}}
= {safe_end_s}\ \mathrm{{s}}.
$$

采用的含水率数值误差安全上界为 `{numerical_error_bound:.6e} kg/kg`。

## 表 6：药材烘干过程的水分浓度

{chr(10).join(rows)}

表中“—”表示该固定物理位置已经位于收缩后的药材外部；移动表面值单独列出。所有表格值保留四位小数，临界时刻由未舍入结果确定。
"""
    path.write_text(text, encoding="utf-8")


def write_validation_markdown(
    path: Path,
    fine: SimulationResult,
    safe_end_s: int,
    numerical_error_bound: float,
    spatial: dict | None,
    temporal: dict | None,
    excel_validation: dict,
    environment: EnvironmentData,
    radius_data: RadiusData,
    density_consistency: dict[str, float | str] | None = None,
) -> None:
    d = fine.diagnostics
    lines = [
        "# 第四问数值验证报告",
        "",
        "## 最终时长",
        "",
        f"- 连续临界时刻：{fine.drying_crossing_s / 3600:.10f} h；",
        f"- 计入数值误差、60 s 输出间隔和四位小数显示裕量后的安全时长：{safe_end_s / 3600:.10f} h（{safe_end_s} s）；",
        f"- 含水率数值误差安全上界：{numerical_error_bound:.6e} kg/kg；",
        f"- 安全时刻的最大含水率：{fine.maximum_moistures_kg_kg[np.where(fine.times_s == safe_end_s)[0][0]]:.12f} kg/kg。",
        "",
        "## 数据与延拓",
        "",
        (
            f"- 附件 1：{environment.times_s.size} 个测点，覆盖 0–{environment.times_s[-1]:g} s；"
            + (
                f"之后固定为 {environment.extension_temperature_c:.6f} °C、"
                f"{environment.extension_moisture_kg_kg:.8f} kg/kg；"
                if environment.extension_strategy == "fixed"
                else "之后保持末值；"
            )
        ),
        f"- 附件 2：{radius_data.times_s.size} 个测点，覆盖 0–{radius_data.times_s[-1]:g} s；之后保持 {100 * radius_data.radii_m[-1]:.4f} cm；",
        f"- 本次安全结束时刻为 {safe_end_s:g} s，早于附件 2 末时刻，因此半径常值延拓未实际参与主结果；",
        "- 半径和数据覆盖范围内的环境均采用分段线性插值；",
        "- 主模型采用均匀仿射收缩的材料坐标，不重复加入网格运动项。",
        "",
        "## 主计算配置与诊断",
        "",
        f"- 径向区间数：{fine.config.radial_cells}；",
        f"- BDF 相对容差：{fine.config.relative_tolerance:.3e}；",
        f"- 温度/含水率绝对容差：{fine.config.temperature_absolute_tolerance:.3e} / {fine.config.moisture_absolute_tolerance:.3e}；",
        f"- 前 4 h/其后最大内部时间步：{fine.config.initial_maximum_step_s:g} / {fine.config.drying_maximum_step_s:g} s；",
        f"- 主计算运行时间：{d.runtime_s:.3f} s；函数计算 {d.function_evaluations} 次；稀疏 LU 分解 {d.lu_decompositions} 次；",
        f"- 温度范围：{d.min_temperature_c:.10f}–{d.max_temperature_c:.10f} °C；",
        f"- 含水率范围：{d.min_moisture_kg_kg:.10f}–{d.max_moisture_kg_kg:.10f} kg/kg；",
        f"- 扩散系数范围：{d.min_diffusivity_m2_s:.10e}–{d.max_diffusivity_m2_s:.10e} m²/s；",
        f"- 最大含水率在全部输出时刻均位于中心（数值容差内）：{'是' if d.maximum_location_was_center else '否'}；",
        f"- 材料坐标归一化水分库存与边界通量的最大平衡绝对残差：{d.max_output_interval_balance_abs:.6e}；",
        f"- 对应最大相对残差：{d.max_output_interval_balance_rel:.6e}。该残差验证离散 PDE，不代表题面密度与收缩数据已经满足干固体质量守恒。",
        "",
    ]
    if density_consistency is not None:
        c = density_consistency
        lines.extend(
            [
                "## 密度—收缩结构一致性诊断",
                "",
                "以下诊断仅检验一种附加解释：把附录 4 的 $\\rho(C)$ 当作实际湿物料体积密度，进而令干固体体积密度为 $\\rho(C)/(1+C)$。它不是当前 PDE 的数值误差。",
                "",
                f"- 诊断时刻：{float(c['evaluation_time_s']):.0f} s；当前半径：{100.0 * float(c['current_radius_m']):.6f} cm；",
                f"- 初始/当前干固体质量代理量：{float(c['initial_dry_mass_proxy_kg']):.12f} / {float(c['current_dry_mass_proxy_kg']):.12f} kg；",
                f"- 干固体质量代理量相对变化：{100.0 * float(c['dry_mass_proxy_relative_change']):.6f}%；",
                f"- 局部 $J\\rho_d/\\rho_{{d0}}$ 范围：{float(c['local_dry_mass_jacobian_ratio_min']):.9f}–{float(c['local_dry_mass_jacobian_ratio_max']):.9f}，严格守恒时应恒为 1；",
                f"- 在相同含水率场下维持全局干质量守恒所需半径：{100.0 * float(c['dry_mass_conserving_radius_m']):.6f} cm；",
                "- 结论：题面经验密度、附件半径和均匀仿射材料映射不能同时按实际湿密度解释并严格满足干固体质量守恒。主模型因此把题面密度限定为热方程中的有效热储存密度，干质量则在参考材料坐标中定义；该取舍属于模型结构不确定性。",
                "",
            ]
        )
    if spatial is not None:
        mf = spatial["medium_to_fine"]
        fine_cells = fine.config.radial_cells
        medium_cells = fine_cells // 2
        coarse_cells = fine_cells // 4
        lines.extend(
            [
                "## 三层空间网格收敛",
                "",
                f"- {coarse_cells}–{medium_cells} 最大差：{spatial['coarse_to_medium']['max_abs_difference']:.6e} kg/kg；",
                f"- {medium_cells}–{fine_cells} 最大差：{mf['max_abs_difference']:.6e} kg/kg；",
                f"- 观测收敛阶：{spatial['observed_order']:.6f}；",
                f"- 细网格 Richardson 误差的 1.25 安全上界：{spatial['estimated_fine_grid_max_error_with_1p25_safety']:.6e} kg/kg；",
                f"- {coarse_cells}/{medium_cells}/{fine_cells} 临界时刻：{spatial['crossing_times_s']['coarse']:.6f} / {spatial['crossing_times_s']['medium']:.6f} / {spatial['crossing_times_s']['fine']:.6f} s。",
                "",
            ]
        )
    if temporal is not None:
        lines.extend(
            [
                "## 时间积分加严复核",
                "",
                f"- 同网格最大含水率差：{temporal['max_abs_difference']:.6e} kg/kg；",
                f"- RMS 差：{temporal['rms_difference']:.6e} kg/kg；",
                f"- 临界时刻差：{temporal['drying_crossing_difference_s']:.6e} s。",
                "",
            ]
        )
    lines.extend(
        [
            "## Excel 检查",
            "",
            f"- 工作表范围：{excel_validation['dimension']}；",
            f"- 已核验数值单元格：{excel_validation['checked_numerical_values']:,}；",
            f"- 与内存结果四舍五入后的最大差：{excel_validation['max_error_against_rounded_solver_output']:.3e}。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_self_tests() -> None:
    parameters = PhysicalParameters()
    moisture = np.asarray([2.55])
    temperature = np.asarray([28.0])
    rho, cp, k, diffusivity = parameters.evaluate_properties(moisture, temperature)
    expected = 4.2e-4 * math.exp(-0.30 / 2.55) * math.exp(-3850.0 / 301.15)
    if abs(float(rho[0]) - (760.0 + 90.0 * 2.55)) > 1.0e-12:
        raise AssertionError("density self-test failed")
    if abs(float(diffusivity[0]) - expected) > 1.0e-22:
        raise AssertionError("diffusivity self-test failed")
    if cp[0] <= 0.0 or k[0] <= 0.0:
        raise AssertionError("property positivity self-test failed")
    xi = make_reference_grid(190)
    if abs(float(np.sum(radial_storage_weights(xi))) - 0.5) > 1.0e-15:
        raise AssertionError("control-volume weight self-test failed")


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", type=Path, default=root / "A题" / "附件" / "附件1.xlsx")
    parser.add_argument("--radius", type=Path, default=root / "A题" / "附件" / "附件2.xlsx")
    parser.add_argument("--template", type=Path, default=root / "A题" / "附件" / "附件3" / "result4.xlsx")
    parser.add_argument("--outdir", type=Path, default=root / "result")
    parser.add_argument("--radial-cells", type=int, default=6080)
    parser.add_argument("--rtol", type=float, default=5.0e-12)
    parser.add_argument("--initial-max-step", type=float, default=0.5)
    parser.add_argument("--drying-max-step", type=float, default=2.5)
    parser.add_argument("--first-step", type=float, default=2.5e-5)
    parser.add_argument("--post-environment-temperature-c", type=float, default=50.0)
    parser.add_argument("--post-environment-moisture", type=float, default=0.05)
    parser.add_argument("--skip-convergence", action="store_true")
    parser.add_argument("--skip-temporal-check", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_arguments(argv)
    run_self_tests()
    environment = load_environment(args.environment)
    fixed_environment_values = (
        args.post_environment_temperature_c,
        args.post_environment_moisture,
    )
    if (fixed_environment_values[0] is None) != (fixed_environment_values[1] is None):
        raise ValueError(
            "post-environment temperature and moisture must be specified together"
        )
    if fixed_environment_values[0] is not None:
        environment = replace(
            environment,
            extension_strategy="fixed",
            extension_temperature_c=fixed_environment_values[0],
            extension_moisture_kg_kg=fixed_environment_values[1],
        )
        environment.validate()
    radius_data = load_radius(args.radius)
    parameters = PhysicalParameters()
    main_config = NumericalConfig(
        radial_cells=args.radial_cells,
        relative_tolerance=args.rtol,
        temperature_absolute_tolerance=args.rtol,
        moisture_absolute_tolerance=args.rtol * 0.01,
        initial_maximum_step_s=args.initial_max_step,
        drying_maximum_step_s=args.drying_max_step,
        first_step_s=args.first_step,
    )
    label = None if args.quiet else "fine/main"
    fine = solve_until_dry(environment, radius_data, parameters, main_config, label)
    run_configs: dict[str, NumericalConfig] = {"fine_main": main_config}

    spatial = None
    if not args.skip_convergence:
        if args.radial_cells % 4 != 0:
            raise ValueError("full convergence requires radial_cells divisible by four")
        medium_config = replace(main_config, radial_cells=args.radial_cells // 2)
        coarse_config = replace(main_config, radial_cells=args.radial_cells // 4)
        medium = solve_until_dry(
            environment,
            radius_data,
            parameters,
            medium_config,
            None if args.quiet else "medium",
        )
        coarse = solve_until_dry(
            environment,
            radius_data,
            parameters,
            coarse_config,
            None if args.quiet else "coarse",
        )
        spatial = spatial_convergence(coarse, medium, fine)
        run_configs["medium"] = medium_config
        run_configs["coarse"] = coarse_config

    temporal = None
    if not args.skip_temporal_check:
        tight_config = replace(
            main_config,
            relative_tolerance=main_config.relative_tolerance / 4.0,
            temperature_absolute_tolerance=main_config.temperature_absolute_tolerance / 4.0,
            moisture_absolute_tolerance=main_config.moisture_absolute_tolerance / 4.0,
            initial_maximum_step_s=main_config.initial_maximum_step_s / 2.0,
            drying_maximum_step_s=main_config.drying_maximum_step_s / 2.0,
            first_step_s=main_config.first_step_s / 2.0,
        )
        tight = solve_until_dry(
            environment,
            radius_data,
            parameters,
            tight_config,
            None if args.quiet else "fine/tight-time",
        )
        temporal = common_field_comparison(fine, tight)
        run_configs["fine_tight_time"] = tight_config

    spatial_bound = (
        0.0
        if spatial is None
        else float(spatial["estimated_fine_grid_max_error_with_1p25_safety"])
    )
    temporal_bound = 0.0 if temporal is None else 1.25 * temporal["max_abs_difference"]
    numerical_error_bound = spatial_bound + temporal_bound
    safe_end_s = choose_safe_end_time(
        fine,
        numerical_error_bound,
        parameters.drying_threshold_kg_kg,
    )
    density_consistency = density_shrinkage_consistency_at_time(
        fine, parameters, safe_end_s
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    result_xlsx = args.outdir / "result4.xlsx"
    write_result_xlsx(args.template, result_xlsx, fine, safe_end_s)
    excel_validation = validate_result_xlsx(result_xlsx, fine, safe_end_s)
    write_full_precision_csv(
        args.outdir / "moisture_physical_full_precision.csv",
        fine,
        safe_end_s,
        material_coordinates=False,
    )
    write_full_precision_csv(
        args.outdir / "moisture_material_coordinates_full_precision.csv",
        fine,
        safe_end_s,
        material_coordinates=True,
    )
    write_summary_markdown(
        args.outdir / "summary_tables.md",
        fine,
        fine.drying_crossing_s,
        safe_end_s,
        numerical_error_bound,
    )
    write_validation_markdown(
        args.outdir / "validation_report.md",
        fine,
        safe_end_s,
        numerical_error_bound,
        spatial,
        temporal,
        excel_validation,
        environment,
        radius_data,
        density_consistency,
    )
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "inputs": {
            "environment": {"path": str(args.environment.resolve()), "sha256": sha256_file(args.environment)},
            "radius": {"path": str(args.radius.resolve()), "sha256": sha256_file(args.radius)},
            "template": {"path": str(args.template.resolve()), "sha256": sha256_file(args.template)},
        },
        "environment_extension": {
            "strategy": environment.extension_strategy,
            "temperature_c": (
                environment.temperatures_c[-1]
                if environment.extension_strategy == "last_point"
                else environment.extension_temperature_c
            ),
            "moisture_kg_kg": (
                environment.moistures_kg_kg[-1]
                if environment.extension_strategy == "last_point"
                else environment.extension_moisture_kg_kg
            ),
        },
        "physical_parameters": asdict(parameters),
        "run_configs": {name: asdict(config) for name, config in run_configs.items()},
        "main_diagnostics": asdict(fine.diagnostics),
        "drying_crossing_s": fine.drying_crossing_s,
        "safe_end_s": safe_end_s,
        "numerical_error_bound_kg_kg": numerical_error_bound,
        "density_shrinkage_consistency_if_rho_is_wet_bulk": density_consistency,
        "spatial_convergence": spatial,
        "temporal_check": temporal,
        "excel_validation": excel_validation,
        "outputs": {"result4_xlsx_sha256": sha256_file(result_xlsx)},
    }
    (args.outdir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not args.quiet:
        print(f"drying crossing: {fine.drying_crossing_s / 3600:.10f} h")
        print(f"safe 60-s drying duration: {safe_end_s / 3600:.10f} h")
        print(
            "dry-mass proxy change if rho is wet bulk density: "
            f"{100.0 * float(density_consistency['dry_mass_proxy_relative_change']):.6f}%"
        )
        print(f"wrote {result_xlsx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
