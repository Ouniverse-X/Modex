#!/usr/bin/env python3
"""Unit tests for the Question 1 solver."""

from __future__ import annotations

import math
import unittest
from pathlib import Path

from problem1.solve_problem1 import (
    NumericalConfig,
    PhysicalParameters,
    implicit_fvm_step,
    load_environment,
    make_radial_grid,
    radial_storage_weights,
    solve_tridiagonal,
)


class TridiagonalTests(unittest.TestCase):
    def test_known_solution(self) -> None:
        result = solve_tridiagonal(
            [-1.0, -1.0],
            [2.0, 2.0, 2.0],
            [-1.0, -1.0],
            [1.0, 0.0, 1.0],
        )
        self.assertLess(max(abs(value - 1.0) for value in result), 1.0e-13)


class ModelTests(unittest.TestCase):
    def test_thermal_diffusivity(self) -> None:
        p = PhysicalParameters()
        self.assertAlmostEqual(p.thermal_diffusivity_m2_s, 1.6885553470919323e-7)

    def test_moisture_diffusivity(self) -> None:
        p = PhysicalParameters()
        expected = 7.0e-9 * math.exp(-0.89 / 2.55)
        self.assertAlmostEqual(p.moisture_diffusivity(2.55), expected)

    def test_uniform_equilibrium_is_preserved(self) -> None:
        uniform = [10.0] * 41
        result = implicit_fvm_step(
            uniform,
            None,
            0.25,
            [0.02 * i / 40 for i in range(41)],
            820.0 * 2600.0,
            [0.36] * 41,
            25.0,
            10.0,
        )
        self.assertLess(max(abs(value - 10.0) for value in result), 5.0e-12)

    def test_default_grid_alignment(self) -> None:
        config = NumericalConfig()
        config.validate()
        grid = make_radial_grid(
            0.02,
            config.radial_cells,
            config.surface_layer_m,
            config.surface_refinement_factor,
        )
        for i in range(21):
            self.assertLess(min(abs(r - i * 0.001) for r in grid), 1.0e-14)
        self.assertAlmostEqual(sum(radial_storage_weights(grid)), 0.02**2 / 2.0)

    def test_input_data(self) -> None:
        root = Path(__file__).resolve().parent.parent
        path = root / "A题" / "附件" / "附件1.xlsx"
        environment = load_environment(path, 1800.0)
        self.assertEqual(len(environment.times_s), 241)
        self.assertEqual(environment.times_s[0], 0.0)
        self.assertEqual(environment.times_s[-1], 14400.0)
        self.assertEqual(environment.interpolate(0.0), (28.0, 0.01963))
        self.assertEqual(environment.interpolate(1800.0), (41.513, 0.03307))


if __name__ == "__main__":
    unittest.main(verbosity=2)
