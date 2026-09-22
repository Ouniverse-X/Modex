"""Numba-accelerated, numerically equivalent kernel for Question 3.

The physical model, BE/BDF2 time formula, Picard stopping rule, four-point
Kirchhoff face average, event interpolation, and requested output grid are the
same as :mod:`problem3.solve_problem3`.  Only loop execution and storage reuse
are changed: geometry and time-independent coefficients are precomputed once,
and all hot loops run in compiled code with reusable work arrays.
"""

from __future__ import annotations

import math
import time

import numpy as np
from numba import njit

from problem3.solve_problem3 import (
    EnvironmentData,
    NumericalConfig,
    PhysicalParameters,
    SimulationResult,
    SolverDiagnostics,
)


@njit(cache=True)
def _solve_tridiagonal_reuse(lower, diagonal, upper, rhs, cprime, dprime, out):
    n = diagonal.size
    pivot = diagonal[0]
    if n > 1:
        cprime[0] = upper[0] / pivot
    dprime[0] = rhs[0] / pivot
    for i in range(1, n):
        pivot = diagonal[i] - lower[i - 1] * cprime[i - 1]
        if i < n - 1:
            cprime[i] = upper[i] / pivot
        dprime[i] = (rhs[i] - lower[i - 1] * dprime[i - 1]) / pivot
    out[n - 1] = dprime[n - 1]
    for i in range(n - 2, -1, -1):
        out[i] = dprime[i] - cprime[i] * out[i + 1]


@njit(cache=True)
def _implicit_step_reuse(
    old,
    previous,
    has_previous,
    inverse_dt,
    weights,
    face_geometry,
    storage,
    gamma_faces,
    boundary_conductance,
    external_value,
    lower,
    diagonal,
    upper,
    rhs,
    cprime,
    dprime,
    out,
):
    n = old.size
    if has_previous:
        time_diagonal = 1.5 * inverse_dt
    else:
        time_diagonal = inverse_dt

    for i in range(n - 1):
        conductance = face_geometry[i] * gamma_faces[i]
        upper[i] = -conductance
        lower[i] = -conductance

    for i in range(n):
        if has_previous:
            history = (4.0 * old[i] - previous[i]) * (0.5 * inverse_dt)
        else:
            history = old[i] * inverse_dt
        west = 0.0 if i == 0 else -lower[i - 1]
        east = 0.0 if i == n - 1 else -upper[i]
        diagonal[i] = storage[i] * weights[i] * time_diagonal + west + east
        rhs[i] = storage[i] * weights[i] * history

    diagonal[n - 1] += boundary_conductance
    rhs[n - 1] += boundary_conductance * external_value
    _solve_tridiagonal_reuse(lower, diagonal, upper, rhs, cprime, dprime, out)


@njit(cache=True)
def _balance_residual(
    new,
    old,
    previous,
    has_previous,
    inverse_dt,
    weights,
    storage,
    boundary_conductance,
    external_value,
    relative_scale_floor,
):
    storage_rate = 0.0
    for i in range(new.size):
        if has_previous:
            history = (4.0 * old[i] - previous[i]) * (0.5 * inverse_dt)
            time_term = 1.5 * inverse_dt * new[i] - history
        else:
            time_term = inverse_dt * (new[i] - old[i])
        storage_rate += storage[i] * weights[i] * time_term
    boundary_rate = boundary_conductance * (external_value - new[new.size - 1])
    absolute = abs(storage_rate - boundary_rate)
    scale = max(abs(storage_rate), abs(boundary_rate))
    relative = 0.0 if scale < relative_scale_floor else absolute / scale
    return absolute, relative


@njit(cache=True)
def _kernel(
    radial_cells,
    dt_s,
    output_stride,
    maximum_steps,
    external_temperature,
    external_moisture,
    weights,
    face_geometry,
    output_indices,
    picard_tolerance,
    picard_max_iterations,
    progress_stride,
    moisture_threshold,
):
    n = radial_cells + 1
    inverse_dt = 1.0 / dt_s
    radius_m = 0.02
    heat_boundary = radius_m * 25.0
    moisture_boundary = radius_m * 8.0e-7

    temperature = np.full(n, 28.0)
    moisture = np.full(n, 2.55)
    previous_temperature = np.empty(n)
    previous_moisture = np.empty(n)
    guess_temperature = np.empty(n)
    guess_moisture = np.empty(n)
    candidate_temperature = np.empty(n)
    candidate_moisture = np.empty(n)
    heat_storage = np.empty(n)
    unit_storage = np.ones(n)
    heat_faces = np.empty(radial_cells)
    moisture_faces = np.empty(radial_cells)

    lower = np.empty(radial_cells)
    diagonal = np.empty(n)
    upper = np.empty(radial_cells)
    rhs = np.empty(n)
    cprime = np.empty(radial_cells)
    dprime = np.empty(n)

    max_outputs = maximum_steps // output_stride + 2
    output_times = np.empty(max_outputs)
    output_values = np.empty((max_outputs, output_indices.size))
    output_count = 0

    steps = 0
    max_picard = 0
    total_picard = 0
    final_update = 0.0
    max_heat_abs = 0.0
    max_heat_rel = 0.0
    max_moist_abs = 0.0
    max_moist_rel = 0.0
    max_monotonicity = 0.0
    min_temperature = 1.0e300
    max_temperature = -1.0e300
    min_moisture = 1.0e300
    max_moisture = -1.0e300
    previous_gap = 2.55 - moisture_threshold
    drying_time = -1.0
    drying_values = np.empty(output_indices.size)
    has_previous = False

    gx0 = -0.8611363115940526
    gx1 = -0.3399810435848563
    gx2 = 0.3399810435848563
    gx3 = 0.8611363115940526
    gw0 = 0.3478548451374538
    gw1 = 0.6521451548625461
    gw2 = 0.6521451548625461
    gw3 = 0.3478548451374538

    for step in range(1, maximum_steps + 1):
        for i in range(n):
            guess_temperature[i] = temperature[i]
            guess_moisture[i] = moisture[i]

        converged = False
        combined_update = 1.0e300
        iteration_used = 0
        for iteration in range(1, picard_max_iterations + 1):
            for i in range(n):
                c = guess_moisture[i]
                density = 650.0 + 128.0 * c
                heat_capacity = 1450.0 + 2736.0 * c / (c + 1.0)
                heat_storage[i] = density * heat_capacity
            for i in range(radial_cells):
                cface = 0.5 * (guess_moisture[i] + guess_moisture[i + 1])
                heat_faces[i] = 0.21 + 0.38 * cface / (cface + 1.0)

            _implicit_step_reuse(
                temperature,
                previous_temperature,
                has_previous,
                inverse_dt,
                weights,
                face_geometry,
                heat_storage,
                heat_faces,
                heat_boundary,
                external_temperature[step],
                lower,
                diagonal,
                upper,
                rhs,
                cprime,
                dprime,
                candidate_temperature,
            )

            for i in range(radial_cells):
                left = guess_moisture[i]
                right = guess_moisture[i + 1]
                midpoint = 0.5 * (left + right)
                half_span = 0.5 * (right - left)
                face_temperature_k = (
                    0.5 * (candidate_temperature[i] + candidate_temperature[i + 1])
                    + 273.15
                )
                thermal = 2.4e-3 * math.exp(-3850.0 / face_temperature_k)
                q0 = math.exp(-0.45 / (midpoint + half_span * gx0))
                q1 = math.exp(-0.45 / (midpoint + half_span * gx1))
                q2 = math.exp(-0.45 / (midpoint + half_span * gx2))
                q3 = math.exp(-0.45 / (midpoint + half_span * gx3))
                moisture_faces[i] = 0.5 * thermal * (
                    gw0 * q0 + gw1 * q1 + gw2 * q2 + gw3 * q3
                )

            _implicit_step_reuse(
                moisture,
                previous_moisture,
                has_previous,
                inverse_dt,
                weights,
                face_geometry,
                unit_storage,
                moisture_faces,
                moisture_boundary,
                external_moisture[step],
                lower,
                diagonal,
                upper,
                rhs,
                cprime,
                dprime,
                candidate_moisture,
            )

            temperature_update = 0.0
            moisture_update = 0.0
            for i in range(n):
                temperature_update = max(
                    temperature_update,
                    abs(candidate_temperature[i] - guess_temperature[i]),
                )
                moisture_update = max(
                    moisture_update,
                    abs(candidate_moisture[i] - guess_moisture[i]),
                )
                guess_temperature[i] = candidate_temperature[i]
                guess_moisture[i] = candidate_moisture[i]
            combined_update = max(temperature_update / 50.0, moisture_update)
            iteration_used = iteration
            if combined_update < picard_tolerance:
                converged = True
                break
        if not converged:
            return (output_times[:0], output_values[:0], drying_values, -1.0, np.asarray([1.0]))

        for i in range(n):
            c = candidate_moisture[i]
            heat_storage[i] = (650.0 + 128.0 * c) * (
                1450.0 + 2736.0 * c / (c + 1.0)
            )
        heat_abs, heat_rel = _balance_residual(
            candidate_temperature,
            temperature,
            previous_temperature,
            has_previous,
            inverse_dt,
            weights,
            heat_storage,
            heat_boundary,
            external_temperature[step],
            1.0e-8,
        )
        moisture_abs, moisture_rel = _balance_residual(
            candidate_moisture,
            moisture,
            previous_moisture,
            has_previous,
            inverse_dt,
            weights,
            unit_storage,
            moisture_boundary,
            external_moisture[step],
            1.0e-14,
        )

        steps += 1
        total_picard += iteration_used
        max_picard = max(max_picard, iteration_used)
        final_update = combined_update
        max_heat_abs = max(max_heat_abs, heat_abs)
        max_heat_rel = max(max_heat_rel, heat_rel)
        max_moist_abs = max(max_moist_abs, moisture_abs)
        max_moist_rel = max(max_moist_rel, moisture_rel)

        new_maximum = candidate_moisture[0]
        for i in range(n):
            min_temperature = min(min_temperature, candidate_temperature[i])
            max_temperature = max(max_temperature, candidate_temperature[i])
            min_moisture = min(min_moisture, candidate_moisture[i])
            max_moisture = max(max_moisture, candidate_moisture[i])
            new_maximum = max(new_maximum, candidate_moisture[i])
            if i < n - 1:
                max_monotonicity = max(
                    max_monotonicity,
                    candidate_moisture[i + 1] - candidate_moisture[i],
                )

        new_gap = new_maximum - moisture_threshold
        if new_gap < 0.0 <= previous_gap:
            fraction = previous_gap / (previous_gap - new_gap)
            fraction = min(1.0, max(0.0, fraction))
            drying_time = step * dt_s - dt_s + fraction * dt_s
            for j in range(output_indices.size):
                i = output_indices[j]
                drying_values[j] = moisture[i] + fraction * (
                    candidate_moisture[i] - moisture[i]
                )
            if output_count == 0 or abs(output_times[output_count - 1] - drying_time) > 1.0e-9:
                output_times[output_count] = drying_time
                for j in range(output_indices.size):
                    output_values[output_count, j] = drying_values[j]
                output_count += 1
            diagnostics = np.asarray(
                [
                    0.0,
                    steps,
                    max_picard,
                    total_picard,
                    final_update,
                    max_heat_abs,
                    max_heat_rel,
                    max_moist_abs,
                    max_moist_rel,
                    max_monotonicity,
                    min_temperature,
                    max_temperature,
                    min_moisture,
                    max_moisture,
                ]
            )
            return (
                output_times[:output_count],
                output_values[:output_count],
                drying_values,
                drying_time,
                diagnostics,
            )

        if step % output_stride == 0:
            output_times[output_count] = step * dt_s
            for j in range(output_indices.size):
                output_values[output_count, j] = candidate_moisture[output_indices[j]]
            output_count += 1

        for i in range(n):
            previous_temperature[i] = temperature[i]
            temperature[i] = candidate_temperature[i]
            previous_moisture[i] = moisture[i]
            moisture[i] = candidate_moisture[i]
        has_previous = True
        previous_gap = new_gap

        if progress_stride > 0 and step % progress_stride == 0:
            print("progress_h", step * dt_s / 3600.0, "Cmax", new_maximum)

    return (output_times[:0], output_values[:0], drying_values, -1.0, np.asarray([2.0]))


def solve_model_accelerated(
    environment: EnvironmentData,
    parameters: PhysicalParameters,
    config: NumericalConfig,
    progress: bool = False,
) -> SimulationResult:
    """Run the original numerical scheme through the compiled hot-loop kernel."""
    config.validate()
    environment.validate()
    standard = PhysicalParameters()
    if (
        parameters.radius_m != standard.radius_m
        or parameters.initial_temperature_c != standard.initial_temperature_c
        or parameters.initial_moisture_kg_kg != standard.initial_moisture_kg_kg
        or parameters.heat_transfer_w_m2_k != standard.heat_transfer_w_m2_k
        or parameters.mass_transfer_m_s != standard.mass_transfer_m_s
    ):
        raise ValueError(
            "accelerated kernel permits a custom moisture threshold only; "
            "all other Q3 physical parameters must remain standard"
        )
    if not (
        environment.plateau_moisture_kg_kg
        < parameters.moisture_threshold_kg_kg
        < parameters.initial_moisture_kg_kg
    ):
        raise ValueError(
            "moisture threshold must lie strictly between the extended "
            "environment moisture and the initial moisture"
        )

    n = config.radial_cells + 1
    radii = np.linspace(0.0, parameters.radius_m, n)
    faces = 0.5 * (radii[:-1] + radii[1:])
    weights = np.empty(n)
    weights[0] = 0.5 * faces[0] ** 2
    weights[1:-1] = 0.5 * (faces[1:] ** 2 - faces[:-1] ** 2)
    weights[-1] = 0.5 * (parameters.radius_m**2 - faces[-1] ** 2)
    face_geometry = faces / np.diff(radii)
    output_indices = np.arange(21, dtype=np.int64) * (config.radial_cells // 20)
    output_stride = int(round(config.output_interval_s / config.time_step_s))
    maximum_steps = int(math.ceil(config.maximum_time_h * 3600.0 / config.time_step_s))

    times = np.arange(maximum_steps + 1, dtype=float) * config.time_step_s
    source_times = np.asarray(environment.times_s, dtype=float)
    external_temperature = np.interp(
        times,
        source_times,
        np.asarray(environment.temperatures_c, dtype=float),
    )
    external_moisture = np.interp(
        times,
        source_times,
        np.asarray(environment.moistures_kg_kg, dtype=float),
    )
    extension_mask = times >= source_times[-1]
    external_temperature[extension_mask] = environment.plateau_temperature_c
    external_moisture[extension_mask] = environment.plateau_moisture_kg_kg

    started = time.perf_counter()
    output_times, output_values, drying_values, drying_time, diag = _kernel(
        config.radial_cells,
        config.time_step_s,
        output_stride,
        maximum_steps,
        external_temperature,
        external_moisture,
        weights,
        face_geometry,
        output_indices,
        config.picard_tolerance,
        config.picard_max_iterations,
        int(round(21600.0 / config.time_step_s)) if progress else 0,
        parameters.moisture_threshold_kg_kg,
    )
    runtime_s = time.perf_counter() - started
    if drying_time < 0.0:
        reason = "Picard iteration failed" if diag[0] == 1.0 else "drying threshold not reached"
        raise RuntimeError(reason)

    diagnostics = SolverDiagnostics(
        steps=int(diag[1]),
        max_picard_iterations=int(diag[2]),
        total_picard_iterations=int(diag[3]),
        final_picard_update=float(diag[4]),
        max_heat_balance_abs=float(diag[5]),
        max_heat_balance_rel=float(diag[6]),
        max_moisture_balance_abs=float(diag[7]),
        max_moisture_balance_rel=float(diag[8]),
        max_radial_monotonicity_violation=float(diag[9]),
        min_temperature_c=float(diag[10]),
        max_temperature_c=float(diag[11]),
        min_moisture_kg_kg=float(diag[12]),
        max_moisture_kg_kg=float(diag[13]),
    )
    return SimulationResult(
        times_s=output_times.tolist(),
        radii_cm=[0.1 * i for i in range(21)],
        moistures_kg_kg=output_values.tolist(),
        drying_time_s=float(drying_time),
        drying_moistures_kg_kg=drying_values.tolist(),
        diagnostics=diagnostics,
        runtime_s=runtime_s,
        config=config,
    )
