#!/usr/bin/env python3
"""Unit tests for the Question 3 numerical experiment."""

from __future__ import annotations

import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from problem3.accelerated_problem3 import solve_model_accelerated
from problem3.solve_problem3 import (
    NumericalConfig,
    PhysicalParameters,
    implicit_radial_step,
    integrated_face_diffusivity,
    load_environment,
    make_unique_experiment_directory,
    make_uniform_grid,
    radial_storage_weights,
    solve_tridiagonal,
)


class LinearAlgebraTests(unittest.TestCase):
    def test_tridiagonal_known_solution(self) -> None:
        result = solve_tridiagonal(
            [-1.0, -1.0],
            [2.0, 2.0, 2.0],
            [-1.0, -1.0],
            [1.0, 0.0, 1.0],
        )
        self.assertLess(max(abs(value - 1.0) for value in result), 1.0e-12)


class ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parameters = PhysicalParameters()

    def test_appendix3_properties(self) -> None:
        moisture = 2.55
        self.assertAlmostEqual(self.parameters.density(moisture), 650 + 128 * moisture)
        self.assertAlmostEqual(
            self.parameters.heat_capacity(moisture),
            1450 + 2736 * moisture / (moisture + 1),
        )
        self.assertAlmostEqual(
            self.parameters.conductivity(moisture),
            0.21 + 0.38 * moisture / (moisture + 1),
        )
        expected_d = (
            2.4e-3
            * math.exp(-0.45 / moisture)
            * math.exp(-3850 / (28 + 273.15))
        )
        self.assertAlmostEqual(
            self.parameters.diffusivity(moisture, 28 + 273.15), expected_d
        )

    def test_integrated_face_diffusivity_constant_state(self) -> None:
        expected = self.parameters.diffusivity(0.5, 323.15)
        actual = integrated_face_diffusivity(
            0.5, 0.5, 323.15, self.parameters
        )
        self.assertAlmostEqual(actual, expected, places=18)

    def test_radial_weights_integrate_area(self) -> None:
        grid = make_uniform_grid(0.02, 80)
        self.assertAlmostEqual(sum(radial_storage_weights(grid)), 0.02**2 / 2)

    def test_uniform_equilibrium_is_preserved(self) -> None:
        grid = make_uniform_grid(0.02, 20)
        uniform = [0.5] * len(grid)
        result = implicit_radial_step(
            old=uniform,
            previous=None,
            dt_s=60.0,
            radii_m=grid,
            storage_nodes=[1.0] * len(grid),
            gamma_faces=[1.0e-9] * (len(grid) - 1),
            boundary_transfer=8.0e-7,
            external_value=0.5,
        )
        self.assertLess(max(abs(value - 0.5) for value in result), 1.0e-13)

    def test_attachment1_plateau_mean(self) -> None:
        root = Path(__file__).resolve().parent.parent
        environment = load_environment(
            root / "A题" / "附件" / "附件1.xlsx",
            plateau_start_h=3.0,
            extension_strategy="last_hour_mean",
        )
        self.assertEqual(len(environment.times_s), 241)
        self.assertAlmostEqual(environment.plateau_temperature_c, 49.99893442622951)
        self.assertAlmostEqual(environment.plateau_moisture_kg_kg, 0.04998754098360656)
        self.assertEqual(environment.value(0.0), (28.0, 0.01963))
        self.assertEqual(
            environment.value(20000.0),
            (
                environment.plateau_temperature_c,
                environment.plateau_moisture_kg_kg,
            ),
        )

    def test_fixed_environment_extension(self) -> None:
        root = Path(__file__).resolve().parent.parent
        environment = load_environment(
            root / "A题" / "附件" / "附件1.xlsx",
            plateau_start_h=3.0,
            extension_strategy="fixed",
            fixed_temperature_c=50.0,
            fixed_moisture_kg_kg=0.05,
        )
        self.assertEqual(environment.value(20000.0), (50.0, 0.05))

    def test_accelerated_solver_honors_custom_threshold(self) -> None:
        root = Path(__file__).resolve().parent.parent
        environment = load_environment(
            root / "A题" / "附件" / "附件1.xlsx",
            plateau_start_h=3.0,
            extension_strategy="fixed",
            fixed_temperature_c=50.0,
            fixed_moisture_kg_kg=0.05,
        )
        config = NumericalConfig(
            radial_cells=20,
            time_step_s=60.0,
            output_interval_s=60.0,
            maximum_time_h=96.0,
            environment_extension="fixed",
        )
        default = solve_model_accelerated(
            environment, PhysicalParameters(), config
        )
        strict_parameters = replace(
            PhysicalParameters(), moisture_threshold_kg_kg=0.14994
        )
        strict = solve_model_accelerated(
            environment, strict_parameters, config
        )
        self.assertGreater(strict.drying_time_s, default.drying_time_s)
        self.assertAlmostEqual(
            max(strict.drying_moistures_kg_kg), 0.14994, places=10
        )

    def test_experiment_directory_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = NumericalConfig(radial_cells=160, time_step_s=5.0)
            first = make_unique_experiment_directory(root, "trial", config)
            second = make_unique_experiment_directory(root, "trial", config)
            self.assertEqual(first.name, "trial")
            self.assertEqual(second.name, "trial_02")


if __name__ == "__main__":
    unittest.main(verbosity=2)
