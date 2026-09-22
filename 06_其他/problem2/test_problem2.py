#!/usr/bin/env python3
"""Unit tests for the Question 2 coupled solver."""

from __future__ import annotations

import math
import unittest
from pathlib import Path

import numpy as np

from problem2.solve_problem2 import (
    CoupledRadialSystem,
    EnvironmentData,
    NumericalConfig,
    PhysicalParameters,
    load_environment,
    make_jacobian_sparsity,
    make_radial_grid,
    radial_storage_weights,
    requested_output_indices,
    solve_model,
)


class PropertyTests(unittest.TestCase):
    def test_appendix_3_properties(self) -> None:
        parameters = PhysicalParameters()
        moisture = np.asarray([2.55])
        temperature = np.asarray([28.0])
        density, heat_capacity, conductivity, diffusivity = (
            parameters.evaluate_properties(moisture, temperature)
        )
        self.assertAlmostEqual(float(density[0]), 650.0 + 128.0 * 2.55)
        self.assertAlmostEqual(
            float(heat_capacity[0]), 1450.0 + 2736.0 * 2.55 / 3.55
        )
        self.assertAlmostEqual(
            float(conductivity[0]), 0.21 + 0.38 * 2.55 / 3.55
        )
        expected_d = (
            2.4e-3 * math.exp(-0.45 / 2.55) * math.exp(-3850.0 / 301.15)
        )
        self.assertAlmostEqual(float(diffusivity[0]), expected_d)


class GridAndSystemTests(unittest.TestCase):
    def test_grid_alignment_and_volume(self) -> None:
        grid = make_radial_grid(0.02, 190, 0.002, 10)
        radii_cm, indices = requested_output_indices(grid)
        self.assertEqual(len(indices), 21)
        self.assertTrue(np.allclose(grid[indices] * 100.0, radii_cm))
        self.assertAlmostEqual(
            float(np.sum(radial_storage_weights(grid))), 0.02**2 / 2.0
        )

    def test_uniform_equilibrium_is_preserved(self) -> None:
        grid = make_radial_grid(0.02, 38, 0.002, 10)
        environment = EnvironmentData(
            np.asarray([0.0, 1.0]),
            np.asarray([28.0, 28.0]),
            np.asarray([2.55, 2.55]),
        )
        system = CoupledRadialSystem(environment, PhysicalParameters(), grid)
        state = np.concatenate((np.full(grid.size, 28.0), np.full(grid.size, 2.55)))
        rate = system(0.5, state)
        self.assertLess(float(np.max(np.abs(rate))), 1.0e-12)

    def test_sparse_jacobian_shape(self) -> None:
        pattern = make_jacobian_sparsity(41)
        self.assertEqual(pattern.shape, (82, 82))
        self.assertGreater(pattern.nnz, 0)

    def test_short_transient_solve(self) -> None:
        environment = EnvironmentData(
            np.asarray([0.0, 2.0]),
            np.asarray([40.0, 40.0]),
            np.asarray([0.02, 0.02]),
        )
        config = NumericalConfig(
            radial_cells=38,
            end_time_s=2.0,
            output_interval_s=1.0,
            relative_tolerance=1.0e-8,
            temperature_absolute_tolerance=1.0e-9,
            moisture_absolute_tolerance=1.0e-11,
            maximum_step_s=0.1,
            first_step_s=1.0e-4,
        )
        result = solve_model(environment, PhysicalParameters(), config)
        self.assertEqual(result.temperatures_c.shape, (2, 21))
        self.assertEqual(result.moistures_kg_kg.shape, (2, 21))
        self.assertGreater(result.temperatures_c[-1, -1], result.temperatures_c[-1, 0])
        self.assertLess(result.moistures_kg_kg[-1, -1], result.moistures_kg_kg[-1, 0])


class InputTests(unittest.TestCase):
    def test_attachment_1(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        environment = load_environment(
            project_root / "A题" / "附件" / "附件1.xlsx", 10800.0
        )
        self.assertEqual(environment.times_s.size, 241)
        self.assertEqual(environment.interpolate(0.0), (28.0, 0.01963))
        self.assertEqual(environment.interpolate(10800.0), (50.195, 0.04977))


if __name__ == "__main__":
    unittest.main(verbosity=2)
