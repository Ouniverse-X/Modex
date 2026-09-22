"""Fixed-length, true-density, dry-mass-conserving free-radius model.

The measured radius from Attachment 2 is deliberately excluded from the
forward model.  It is used only after the solve as an independent validation
series.  The material coordinate ``mu`` is cumulative dry-solid mass fraction.
With fixed cylinder length, local dry-mass conservation uniquely determines
the non-affine radial mapping and hence the outer radius.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import njit


HERE = Path(__file__).resolve().parent

# Reuse the previously verified finite-volume linear algebra, interpolation,
# grid generator and Appendix-4 property implementation.  The free-boundary
# geometry and time-integration kernel below are new and live in this file.
from visualize_4_2_model import (  # noqa: E402
    _center_value,
    _conductances,
    _implicit_step,
    _properties,
    _sample_profile,
    _sample_radii,
    appendix4_properties,
    load_environment,
    load_radius,
    make_mass_faces,
)


@dataclass(frozen=True)
class NumericalConfig:
    radial_cells: int = 760
    time_step_s: float = 2.0
    output_interval_s: float = 60.0
    maximum_time_h: float = 72.0
    surface_layer: float = 0.1
    surface_refinement_factor: int = 10
    picard_tolerance: float = 1.0e-10
    picard_max_iterations: int = 40
    drying_threshold_kg_kg: float = 0.15

    def validate(self) -> None:
        if self.radial_cells < 38 or self.radial_cells % 19 != 0:
            raise ValueError("radial_cells must be a multiple of 19 and at least 38")
        if self.time_step_s <= 0.0 or self.output_interval_s <= 0.0:
            raise ValueError("time intervals must be positive")
        stride = self.output_interval_s / self.time_step_s
        if abs(stride - round(stride)) > 1.0e-12:
            raise ValueError("output interval must be divisible by the time step")
        total_steps = self.maximum_time_h * 3600.0 / self.time_step_s
        if abs(total_steps - round(total_steps)) > 1.0e-10:
            raise ValueError("maximum time must be divisible by the time step")
        if self.maximum_time_h <= 4.0:
            raise ValueError("maximum time must exceed the measured environment")
        if not 0.0 < self.surface_layer < 1.0:
            raise ValueError("surface_layer must lie in (0,1)")
        if self.surface_refinement_factor < 1:
            raise ValueError("surface_refinement_factor must be positive")
        if self.picard_tolerance <= 0.0 or self.picard_max_iterations < 1:
            raise ValueError("invalid Picard settings")


@dataclass
class SimulationResult:
    times_s: np.ndarray
    dry_mass_coordinates: np.ndarray
    temperatures_c: np.ndarray
    moistures_kg_kg: np.ndarray
    material_radii_m: np.ndarray
    predicted_radii_m: np.ndarray
    surface_temperatures_c: np.ndarray
    surface_moistures_kg_kg: np.ndarray
    maximum_moistures_kg_kg: np.ndarray
    drying_crossing_s: float
    runtime_s: float
    diagnostics: dict[str, float | int | bool]
    config: NumericalConfig


def fixed_length_geometry(
    moisture: np.ndarray,
    mass_faces: np.ndarray,
    initial_radius_m: float = 0.02,
    fixed_length_m: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Compute the radial map implied by local dry-mass conservation.

    ``fixed_length_m`` is accepted for interface clarity.  It cancels from the
    normalized radius formula because the initial and current lengths are the
    same, while it remains part of the total dry mass in the PDE solver.
    """
    del fixed_length_m
    moisture = np.asarray(moisture, dtype=float)
    mass_faces = np.asarray(mass_faces, dtype=float)
    widths = np.diff(mass_faces)
    if moisture.shape != widths.shape:
        raise ValueError("moisture and dry-mass grid shapes differ")
    _, dry_density, _, _, _ = appendix4_properties(
        moisture, np.full_like(moisture, 28.0)
    )
    initial_dry_density = (760.0 + 90.0 * 2.55) / (1.0 + 2.55)
    radius_squared_faces = np.concatenate(
        ([0.0], initial_radius_m**2 * initial_dry_density * np.cumsum(widths / dry_density))
    )
    radius_centers = np.sqrt(
        0.5 * (radius_squared_faces[:-1] + radius_squared_faces[1:])
    )
    local_ratios = (
        dry_density * np.diff(radius_squared_faces)
        / (initial_radius_m**2 * initial_dry_density * widths)
    )
    return (
        np.sqrt(radius_squared_faces),
        radius_centers,
        float(np.sqrt(radius_squared_faces[-1])),
        local_ratios,
    )


@njit(cache=True)
def _fixed_length_geometry(
    dry_density,
    mass_widths,
    initial_radius,
    initial_dry_density,
    radius_squared_faces,
    radius_centers,
):
    radius_squared_faces[0] = 0.0
    coefficient = initial_radius * initial_radius * initial_dry_density
    cumulative = 0.0
    for i in range(dry_density.size):
        cumulative += mass_widths[i] / dry_density[i]
        radius_squared_faces[i + 1] = coefficient * cumulative
        radius_centers[i] = math.sqrt(
            0.5 * (radius_squared_faces[i] + radius_squared_faces[i + 1])
        )
    return math.sqrt(radius_squared_faces[dry_density.size])


@njit(cache=True)
def _kernel(
    mass_faces,
    mass_centers,
    mass_widths,
    sample_coordinates,
    time_step,
    output_stride,
    external_temperatures,
    external_moistures,
    maximum_steps,
    picard_tolerance,
    picard_max_iterations,
    drying_threshold,
    progress_stride,
):
    n = mass_widths.size
    inverse_dt = 1.0 / time_step
    initial_radius = 0.02
    fixed_length = 0.25
    initial_moisture = 2.55
    initial_wet_density = 760.0 + 90.0 * initial_moisture
    initial_dry_density = initial_wet_density / (1.0 + initial_moisture)
    total_dry_mass = (
        math.pi * fixed_length * initial_radius * initial_radius * initial_dry_density
    )

    temperature = np.full(n, 28.0)
    moisture = np.full(n, initial_moisture)
    previous_temperature = np.empty(n)
    previous_moisture = np.empty(n)
    guess_temperature = np.empty(n)
    guess_moisture = np.empty(n)
    candidate_temperature = np.empty(n)
    candidate_moisture = np.empty(n)
    wet_density = np.empty(n)
    dry_density = np.empty(n)
    heat_capacity = np.empty(n)
    conductivity = np.empty(n)
    diffusivity = np.empty(n)
    heat_storage = np.empty(n)
    moisture_storage = mass_widths.copy()
    radius_squared_faces = np.empty(n + 1)
    radius_centers = np.empty(n)
    heat_conductance = np.empty(n - 1)
    moisture_conductance = np.empty(n - 1)

    lower = np.empty(n - 1)
    diagonal = np.empty(n)
    upper = np.empty(n - 1)
    rhs = np.empty(n)
    cprime = np.empty(n - 1)
    dprime = np.empty(n)

    maximum_outputs = maximum_steps // output_stride + 3
    output_times = np.empty(maximum_outputs)
    output_temperatures = np.empty((maximum_outputs, sample_coordinates.size))
    output_moistures = np.empty((maximum_outputs, sample_coordinates.size))
    output_radii = np.empty((maximum_outputs, sample_coordinates.size))
    output_predicted_radius = np.empty(maximum_outputs)
    output_surface_temperature = np.empty(maximum_outputs)
    output_surface_moisture = np.empty(maximum_outputs)
    output_maximum_moisture = np.empty(maximum_outputs)
    output_count = 0

    max_picard = 0
    total_picard = 0
    max_water_balance_abs = 0.0
    max_water_balance_rel = 0.0
    max_local_dry_mass_error = 0.0
    max_radius_increase_m = 0.0
    min_predicted_radius = initial_radius
    max_predicted_radius = initial_radius
    has_previous = False
    previous_gap = initial_moisture - drying_threshold
    drying_time = -1.0
    previous_outer_radius = initial_radius

    # Initial state and output.
    _properties(
        moisture, temperature, wet_density, dry_density, heat_capacity,
        conductivity, diffusivity,
    )
    outer_radius = _fixed_length_geometry(
        dry_density, mass_widths, initial_radius, initial_dry_density,
        radius_squared_faces, radius_centers,
    )
    heat_boundary, moisture_boundary, heat_external, moisture_external = _conductances(
        mass_faces, mass_centers, dry_density, conductivity, diffusivity,
        radius_squared_faces, outer_radius, fixed_length, total_dry_mass,
        25.0, 8.0e-7, heat_conductance, moisture_conductance,
    )
    heat_flux = heat_boundary * (external_temperatures[0] - temperature[n - 1])
    moisture_flux = moisture_boundary * (external_moistures[0] - moisture[n - 1])
    surface_temperature = external_temperatures[0] - heat_flux / heat_external
    surface_moisture = external_moistures[0] - moisture_flux / moisture_external
    output_times[0] = 0.0
    _sample_profile(
        mass_centers, temperature, surface_temperature, sample_coordinates,
        output_temperatures[0],
    )
    _sample_profile(
        mass_centers, moisture, surface_moisture, sample_coordinates,
        output_moistures[0],
    )
    _sample_radii(
        mass_faces, radius_squared_faces, sample_coordinates, output_radii[0]
    )
    output_predicted_radius[0] = outer_radius
    output_surface_temperature[0] = surface_temperature
    output_surface_moisture[0] = surface_moisture
    output_maximum_moisture[0] = initial_moisture
    output_count = 1

    for step in range(1, maximum_steps + 1):
        for i in range(n):
            guess_temperature[i] = temperature[i]
            guess_moisture[i] = moisture[i]

        converged = False
        iteration_used = 0
        heat_boundary = 0.0
        moisture_boundary = 0.0
        heat_external = 0.0
        moisture_external = 0.0
        outer_radius = previous_outer_radius
        for iteration in range(1, picard_max_iterations + 1):
            _properties(
                guess_moisture, guess_temperature, wet_density, dry_density,
                heat_capacity, conductivity, diffusivity,
            )
            outer_radius = _fixed_length_geometry(
                dry_density, mass_widths, initial_radius, initial_dry_density,
                radius_squared_faces, radius_centers,
            )
            heat_boundary, _, heat_external, _ = _conductances(
                mass_faces, mass_centers, dry_density, conductivity, diffusivity,
                radius_squared_faces, outer_radius, fixed_length, total_dry_mass,
                25.0, 8.0e-7, heat_conductance, moisture_conductance,
            )
            for i in range(n):
                heat_storage[i] = (
                    mass_widths[i]
                    * (1.0 + guess_moisture[i])
                    * heat_capacity[i]
                )
            _implicit_step(
                temperature, previous_temperature, has_previous, inverse_dt,
                heat_storage, heat_conductance, heat_boundary,
                external_temperatures[step], lower, diagonal, upper, rhs,
                cprime, dprime, candidate_temperature,
            )

            _properties(
                guess_moisture, candidate_temperature, wet_density, dry_density,
                heat_capacity, conductivity, diffusivity,
            )
            outer_radius = _fixed_length_geometry(
                dry_density, mass_widths, initial_radius, initial_dry_density,
                radius_squared_faces, radius_centers,
            )
            _, moisture_boundary, _, moisture_external = _conductances(
                mass_faces, mass_centers, dry_density, conductivity, diffusivity,
                radius_squared_faces, outer_radius, fixed_length, total_dry_mass,
                25.0, 8.0e-7, heat_conductance, moisture_conductance,
            )
            _implicit_step(
                moisture, previous_moisture, has_previous, inverse_dt,
                moisture_storage, moisture_conductance, moisture_boundary,
                external_moistures[step], lower, diagonal, upper, rhs,
                cprime, dprime, candidate_moisture,
            )

            maximum_update = 0.0
            for i in range(n):
                maximum_update = max(
                    maximum_update,
                    abs(candidate_temperature[i] - guess_temperature[i]) / 50.0,
                    abs(candidate_moisture[i] - guess_moisture[i]),
                )
                guess_temperature[i] = candidate_temperature[i]
                guess_moisture[i] = candidate_moisture[i]
            iteration_used = iteration
            if maximum_update < picard_tolerance:
                converged = True
                break
        if not converged:
            diagnostics = np.asarray([1.0, step])
            return (
                output_times[:output_count], output_temperatures[:output_count],
                output_moistures[:output_count], output_radii[:output_count],
                output_predicted_radius[:output_count],
                output_surface_temperature[:output_count],
                output_surface_moisture[:output_count],
                output_maximum_moisture[:output_count], drying_time, diagnostics,
            )

        _properties(
            candidate_moisture, candidate_temperature, wet_density, dry_density,
            heat_capacity, conductivity, diffusivity,
        )
        outer_radius = _fixed_length_geometry(
            dry_density, mass_widths, initial_radius, initial_dry_density,
            radius_squared_faces, radius_centers,
        )
        heat_boundary, moisture_boundary, heat_external, moisture_external = _conductances(
            mass_faces, mass_centers, dry_density, conductivity, diffusivity,
            radius_squared_faces, outer_radius, fixed_length, total_dry_mass,
            25.0, 8.0e-7, heat_conductance, moisture_conductance,
        )

        storage_rate = 0.0
        for i in range(n):
            if has_previous:
                time_term = (
                    1.5 * candidate_moisture[i]
                    - 2.0 * moisture[i]
                    + 0.5 * previous_moisture[i]
                ) * inverse_dt
            else:
                time_term = (candidate_moisture[i] - moisture[i]) * inverse_dt
            storage_rate += mass_widths[i] * time_term
        boundary_rate = moisture_boundary * (
            external_moistures[step] - candidate_moisture[n - 1]
        )
        balance_abs = abs(storage_rate - boundary_rate)
        balance_scale = max(abs(storage_rate), abs(boundary_rate), 1.0e-16)
        max_water_balance_abs = max(max_water_balance_abs, balance_abs)
        max_water_balance_rel = max(max_water_balance_rel, balance_abs / balance_scale)

        coefficient = initial_radius * initial_radius * initial_dry_density
        for i in range(n):
            local_ratio = (
                dry_density[i]
                * (radius_squared_faces[i + 1] - radius_squared_faces[i])
                / (coefficient * mass_widths[i])
            )
            max_local_dry_mass_error = max(
                max_local_dry_mass_error, abs(local_ratio - 1.0)
            )
        max_radius_increase_m = max(
            max_radius_increase_m, outer_radius - previous_outer_radius
        )
        min_predicted_radius = min(min_predicted_radius, outer_radius)
        max_predicted_radius = max(max_predicted_radius, outer_radius)
        max_picard = max(max_picard, iteration_used)
        total_picard += iteration_used

        heat_flux = heat_boundary * (
            external_temperatures[step] - candidate_temperature[n - 1]
        )
        moisture_flux = moisture_boundary * (
            external_moistures[step] - candidate_moisture[n - 1]
        )
        surface_temperature = external_temperatures[step] - heat_flux / heat_external
        surface_moisture = external_moistures[step] - moisture_flux / moisture_external
        center_moisture = _center_value(mass_centers, candidate_moisture)
        new_maximum = max(center_moisture, np.max(candidate_moisture), surface_moisture)
        new_gap = new_maximum - drying_threshold
        if drying_time < 0.0 and new_gap < 0.0 <= previous_gap:
            fraction = previous_gap / (previous_gap - new_gap)
            fraction = min(1.0, max(0.0, fraction))
            drying_time = (step - 1 + fraction) * time_step

        if step % output_stride == 0 or step == maximum_steps:
            output_times[output_count] = step * time_step
            _sample_profile(
                mass_centers, candidate_temperature, surface_temperature,
                sample_coordinates, output_temperatures[output_count],
            )
            _sample_profile(
                mass_centers, candidate_moisture, surface_moisture,
                sample_coordinates, output_moistures[output_count],
            )
            _sample_radii(
                mass_faces, radius_squared_faces, sample_coordinates,
                output_radii[output_count],
            )
            output_predicted_radius[output_count] = outer_radius
            output_surface_temperature[output_count] = surface_temperature
            output_surface_moisture[output_count] = surface_moisture
            output_maximum_moisture[output_count] = new_maximum
            output_count += 1

        for i in range(n):
            previous_temperature[i] = temperature[i]
            temperature[i] = candidate_temperature[i]
            previous_moisture[i] = moisture[i]
            moisture[i] = candidate_moisture[i]
        has_previous = True
        previous_gap = new_gap
        previous_outer_radius = outer_radius
        if progress_stride > 0 and step % progress_stride == 0:
            print(
                "progress_h", step * time_step / 3600.0,
                "Cmax", new_maximum, "R_cm", 100.0 * outer_radius,
            )

    diagnostics = np.asarray(
        [
            0.0, maximum_steps, max_picard, total_picard,
            max_water_balance_abs, max_water_balance_rel,
            max_local_dry_mass_error, max_radius_increase_m,
            min_predicted_radius, max_predicted_radius,
        ]
    )
    return (
        output_times[:output_count], output_temperatures[:output_count],
        output_moistures[:output_count], output_radii[:output_count],
        output_predicted_radius[:output_count],
        output_surface_temperature[:output_count],
        output_surface_moisture[:output_count],
        output_maximum_moisture[:output_count], drying_time, diagnostics,
    )


def solve_model(
    environment: np.ndarray,
    config: NumericalConfig,
    progress: bool = False,
) -> SimulationResult:
    """Integrate the forward model; no measured-radius input is accepted."""
    config.validate()
    mass_faces = make_mass_faces(
        config.radial_cells,
        config.surface_layer,
        config.surface_refinement_factor,
    )
    mass_centers = 0.5 * (mass_faces[:-1] + mass_faces[1:])
    mass_widths = np.diff(mass_faces)
    sample_coordinates = np.linspace(0.0, 1.0, 21)
    maximum_steps = int(round(config.maximum_time_h * 3600.0 / config.time_step_s))
    times = np.arange(maximum_steps + 1, dtype=float) * config.time_step_s
    external_temperature = np.interp(times, environment[:, 0], environment[:, 1])
    external_moisture = np.interp(times, environment[:, 0], environment[:, 2])
    after_environment = times >= environment[-1, 0]
    external_temperature[after_environment] = 50.0
    external_moisture[after_environment] = 0.05

    started = time.perf_counter()
    values = _kernel(
        mass_faces,
        mass_centers,
        mass_widths,
        sample_coordinates,
        config.time_step_s,
        int(round(config.output_interval_s / config.time_step_s)),
        external_temperature,
        external_moisture,
        maximum_steps,
        config.picard_tolerance,
        config.picard_max_iterations,
        config.drying_threshold_kg_kg,
        int(round(21600.0 / config.time_step_s)) if progress else 0,
    )
    (
        output_times,
        output_temperatures,
        output_moistures,
        output_radii,
        output_predicted_radius,
        output_surface_temperature,
        output_surface_moisture,
        output_maximum_moisture,
        drying_time,
        diagnostic_values,
    ) = values
    if diagnostic_values[0] != 0.0:
        raise RuntimeError(
            f"Picard iteration failed; diagnostic step={int(diagnostic_values[1])}"
        )
    diagnostics: dict[str, float | int | bool] = {
        "steps": int(diagnostic_values[1]),
        "max_picard_iterations": int(diagnostic_values[2]),
        "mean_picard_iterations": float(
            diagnostic_values[3] / diagnostic_values[1]
        ),
        "max_water_balance_abs": float(diagnostic_values[4]),
        "max_water_balance_rel": float(diagnostic_values[5]),
        "max_local_dry_mass_relative_error": float(diagnostic_values[6]),
        "max_single_step_radius_increase_m": float(diagnostic_values[7]),
        "minimum_predicted_radius_m": float(diagnostic_values[8]),
        "maximum_predicted_radius_m": float(diagnostic_values[9]),
        "maximum_moisture_at_center": bool(
            np.all(
                output_moistures[:, 0]
                >= np.max(output_moistures[:, 1:], axis=1) - 1.0e-10
            )
        ),
    }
    return SimulationResult(
        times_s=output_times,
        dry_mass_coordinates=sample_coordinates,
        temperatures_c=output_temperatures,
        moistures_kg_kg=output_moistures,
        material_radii_m=output_radii,
        predicted_radii_m=output_predicted_radius,
        surface_temperatures_c=output_surface_temperature,
        surface_moistures_kg_kg=output_surface_moisture,
        maximum_moistures_kg_kg=output_maximum_moisture,
        drying_crossing_s=float(drying_time),
        runtime_s=time.perf_counter() - started,
        diagnostics=diagnostics,
        config=config,
    )


__all__ = [
    "NumericalConfig",
    "SimulationResult",
    "appendix4_properties",
    "fixed_length_geometry",
    "load_environment",
    "load_radius",
    "make_mass_faces",
    "solve_model",
]
