"""Core routines for Questions 1--4 (I/O, logging and plotting omitted)."""

from __future__ import annotations

from collections.abc import Callable
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


HEAT_TRANSFER = 25.0       # W/(m^2 K)
MASS_TRANSFER = 8.0e-7     # m/s
INITIAL_T = 28.0           # deg C
INITIAL_C = 2.55           # kg/kg, dry basis
DRY_LIMIT = 0.15           # kg/kg, dry basis


def material_properties(
    moisture: np.ndarray, temperature_c: np.ndarray, model: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return rho, cp, k and D for the selected appendix model."""
    c = np.maximum(np.asarray(moisture, dtype=float), 1.0e-12)
    tk = np.asarray(temperature_c, dtype=float) + 273.15
    ratio = c / (1.0 + c)

    if model == 1:  # Question 1: constant thermal properties
        rho = np.full_like(c, 820.0)
        cp = np.full_like(c, 2600.0)
        k = np.full_like(c, 0.36)
        diffusivity = 7.0e-9 * np.exp(-0.89 / c)
    elif model == 3:  # Attachment 3, Questions 2 and 3
        rho = 650.0 + 128.0 * c
        cp = 1450.0 + 2736.0 * ratio
        k = 0.21 + 0.38 * ratio
        diffusivity = 2.4e-3 * np.exp(-0.45 / c - 3850.0 / tk)
    elif model == 4:  # Attachment 4, Question 4
        rho = 760.0 + 90.0 * c
        cp = 1850.0 + 2150.0 * ratio
        k = 0.12 + 0.20 * ratio
        diffusivity = 4.2e-4 * np.exp(-0.30 / c - 3850.0 / tk)
    else:
        raise ValueError("model must be 1, 3 or 4")
    return rho, cp, k, diffusivity


def refined_grid(intervals: int, surface_fraction: float = 0.1) -> np.ndarray:
    """Grid on x in [0,1], with a 10:1 refinement in the surface layer."""
    ratio = 10
    bulk_fraction = 1.0 - surface_fraction
    bulk_intervals = round(
        intervals * bulk_fraction / (bulk_fraction + ratio * surface_fraction)
    )
    surface_intervals = intervals - bulk_intervals
    bulk = np.linspace(0.0, bulk_fraction, bulk_intervals + 1)
    surface = np.linspace(
        bulk_fraction, 1.0, surface_intervals + 1
    )[1:]
    return np.concatenate((bulk, surface))


def storage_weights(x: np.ndarray) -> np.ndarray:
    """Integral of x dx over every node-centred cylindrical control volume."""
    faces = 0.5 * (x[:-1] + x[1:])
    weights = np.empty_like(x)
    weights[0] = 0.5 * faces[0] ** 2
    weights[1:-1] = 0.5 * (faces[1:] ** 2 - faces[:-1] ** 2)
    weights[-1] = 0.5 * (1.0 - faces[-1] ** 2)
    return weights


def harmonic_mean(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return 2.0 * left * right / (left + right)


def thomas_solve(
    lower: np.ndarray, diagonal: np.ndarray,
    upper: np.ndarray, right: np.ndarray,
) -> np.ndarray:
    """Solve a tridiagonal system (Thomas algorithm)."""
    b, d = diagonal.copy(), right.copy()
    for i in range(1, b.size):
        factor = lower[i - 1] / b[i - 1]
        b[i] -= factor * upper[i - 1]
        d[i] -= factor * d[i - 1]
    answer = np.empty_like(d)
    answer[-1] = d[-1] / b[-1]
    for i in range(b.size - 2, -1, -1):
        answer[i] = (d[i] - upper[i] * answer[i + 1]) / b[i]
    return answer


def bdf_history(
    old: np.ndarray,
    previous: np.ndarray | None,
    step_s: float,
    previous_step_s: float | None = None,
) -> tuple[float, np.ndarray]:
    """Return a0 and the history term for BE/variable-step BDF2."""
    if previous is None:
        return 1.0 / step_s, old / step_s
    previous_step_s = step_s if previous_step_s is None else previous_step_s
    q = step_s / previous_step_s
    a0 = (1.0 + 2.0 * q) / ((1.0 + q) * step_s)
    history = ((1.0 + q) * old - q**2 * previous / (1.0 + q)) / step_s
    return a0, history


def implicit_fvm_step(
    old: np.ndarray,
    previous: np.ndarray | None,
    step_s: float,
    previous_step_s: float | None,
    x: np.ndarray,
    radius: float,
    storage: np.ndarray,
    coefficient: np.ndarray,
    transfer: float,
    ambient: float,
) -> np.ndarray:
    """One conservative BE/BDF2 step on the fixed reference interval."""
    a0, history = bdf_history(old, previous, step_s, previous_step_s)
    faces = 0.5 * (x[:-1] + x[1:])
    conductance = (
        faces * harmonic_mean(coefficient[:-1], coefficient[1:]) / np.diff(x)
    )
    capacity = radius**2 * storage_weights(x) * storage
    diagonal = a0 * capacity
    right = capacity * history
    diagonal[:-1] += conductance
    diagonal[1:] += conductance
    diagonal[-1] += radius * transfer
    right[-1] += radius * transfer * ambient
    return thomas_solve(-conductance, diagonal, -conductance, right)


def picard_bdf2_step(
    old_t: np.ndarray,
    old_c: np.ndarray,
    previous_t: np.ndarray | None,
    previous_c: np.ndarray | None,
    step_s: float,
    previous_step_s: float | None,
    x: np.ndarray,
    radius: float,
    model: int,
    ambient: tuple[float, float],
    tolerance: float = 1.0e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Coupled Picard update used by the fixed-step solvers in Q1 and Q3."""
    temperature, moisture = old_t.copy(), old_c.copy()
    for _ in range(50):
        rho, cp, k, _ = material_properties(moisture, temperature, model)
        new_t = implicit_fvm_step(
            old_t, previous_t, step_s, previous_step_s, x, radius,
            rho * cp, k, HEAT_TRANSFER, ambient[0],
        )
        _, _, _, diffusivity = material_properties(moisture, new_t, model)
        new_c = implicit_fvm_step(
            old_c, previous_c, step_s, previous_step_s, x, radius,
            np.ones_like(moisture), diffusivity, MASS_TRANSFER, ambient[1],
        )
        change = max(
            np.max(np.abs(new_t - temperature)),
            np.max(np.abs(new_c - moisture)),
        )
        temperature, moisture = new_t, new_c
        if change < tolerance:
            return temperature, moisture
    raise RuntimeError("Picard iteration did not converge")


def transport_balance(
    field: np.ndarray,
    coefficient: np.ndarray,
    x: np.ndarray,
    radius: float,
    transfer: float,
    ambient: float,
) -> np.ndarray:
    """Conservative internal-face fluxes and the surface Robin flux."""
    faces = 0.5 * (x[:-1] + x[1:])
    face_coefficient = harmonic_mean(coefficient[:-1], coefficient[1:])
    flux = faces * face_coefficient * np.diff(field) / np.diff(x)
    balance = np.empty_like(field)
    balance[0] = flux[0]
    balance[1:-1] = flux[1:] - flux[:-1]
    balance[-1] = radius * transfer * (ambient - field[-1]) - flux[-1]
    return balance


def build_rhs(
    x: np.ndarray,
    model: int,
    radius_at: Callable[[float], float],
    ambient_at: Callable[[float], tuple[float, float]],
) -> Callable[[float, np.ndarray], np.ndarray]:
    """Build the method-of-lines system for a fixed or shrinking cylinder."""
    weights = storage_weights(x)
    n = x.size

    def rhs(time_s: float, state: np.ndarray) -> np.ndarray:
        temperature = state[:n]
        moisture = state[n:]
        radius = float(radius_at(time_s))
        ambient_t, ambient_c = ambient_at(time_s)
        rho, cp, k, diffusivity = material_properties(
            moisture, temperature, model
        )
        heat = transport_balance(
            temperature, k, x, radius, HEAT_TRANSFER, ambient_t
        )
        mass = transport_balance(
            moisture, diffusivity, x, radius, MASS_TRANSFER, ambient_c
        )
        scale = radius**2 * weights
        return np.concatenate((heat / (scale * rho * cp), mass / scale))

    return rhs


def solve_adaptive_bdf(
    intervals: int,
    end_time_s: float,
    model: int,
    radius_at: Callable[[float], float],
    ambient_at: Callable[[float], tuple[float, float]],
    rtol: float = 1.0e-10,
    atol_t: float = 1.0e-10,
    atol_c: float = 1.0e-12,
    max_step_s: float = 2.0,
) -> tuple[object, float | None]:
    """Adaptive BDF driver used by Q2/Q4; locate the drying event."""
    x = refined_grid(intervals)
    n = x.size
    initial = np.r_[np.full(n, INITIAL_T), np.full(n, INITIAL_C)]
    solution = solve_ivp(
        build_rhs(x, model, radius_at, ambient_at),
        (0.0, end_time_s),
        initial,
        method="BDF",
        rtol=rtol,
        atol=np.r_[np.full(n, atol_t), np.full(n, atol_c)],
        dense_output=True,
        max_step=max_step_s,
    )
    if not solution.success:
        raise RuntimeError(solution.message)

    maximum_c = np.max(solution.y[n:], axis=0)
    crossing = np.flatnonzero(
        (maximum_c[:-1] > DRY_LIMIT) & (maximum_c[1:] <= DRY_LIMIT)
    )
    if crossing.size == 0:
        return solution, None
    j = int(crossing[0])
    event = lambda t: float(np.max(solution.sol(t)[n:]) - DRY_LIMIT)
    dry_time = brentq(event, solution.t[j], solution.t[j + 1])
    return solution, float(dry_time)


def richardson_estimate(values: np.ndarray, safety: float = 1.5) -> tuple[float, float]:
    """Observed order and fine-grid remainder for three 2:1 refinements."""
    coarse, medium, fine = np.asarray(values, dtype=float)
    order = np.log2(abs((coarse - medium) / (medium - fine)))
    error = safety * abs(medium - fine) / (2.0**order - 1.0)
    return float(order), float(error)


def dry_mass_ratio(
    x: np.ndarray,
    moisture: np.ndarray,
    radius: float,
    initial_radius: float = 0.02,
) -> float:
    """Question 4 posterior dry-solid conservation diagnostic."""
    weights = storage_weights(x)
    dry_density = (760.0 + 90.0 * moisture) / (1.0 + moisture)
    initial_dry_density = (760.0 + 90.0 * INITIAL_C) / (1.0 + INITIAL_C)
    current = radius**2 * np.sum(weights * dry_density)
    initial = initial_radius**2 * 0.5 * initial_dry_density
    return float(current / initial)
