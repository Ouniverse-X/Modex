#!/usr/bin/env python3
"""Targeted tests for the Question 4 moving-boundary solver."""

from __future__ import annotations

import math
import unittest
from pathlib import Path

import numpy as np

from problem4.solve_problem4 import (
    EnvironmentData,
    MovingBoundarySystem,
    PhysicalParameters,
    RadiusData,
    density_shrinkage_consistency_from_field,
    extract_physical_moistures,
    load_environment,
    load_radius,
    make_jacobian_sparsity,
    make_reference_grid,
    radial_storage_weights,
)


class PropertyTests(unittest.TestCase):
    def test_appendix_4_properties(self) -> None:
        p = PhysicalParameters()
        c = np.asarray([2.55])
        t = np.asarray([28.0])
        rho, cp, k, diffusivity = p.evaluate_properties(c, t)
        self.assertAlmostEqual(float(rho[0]), 760.0 + 90.0 * 2.55)
        self.assertAlmostEqual(float(cp[0]), 1850.0 + 2150.0 * 2.55 / 3.55)
        self.assertAlmostEqual(float(k[0]), 0.12 + 0.20 * 2.55 / 3.55)
        expected = 4.2e-4 * math.exp(-0.30 / 2.55) * math.exp(-3850.0 / 301.15)
        self.assertAlmostEqual(float(diffusivity[0]), expected)


class GridAndSystemTests(unittest.TestCase):
    def test_two_zone_grid_and_weight(self) -> None:
        xi = make_reference_grid(190)
        self.assertEqual(xi.size, 191)
        self.assertAlmostEqual(float(np.sum(radial_storage_weights(xi))), 0.5)
        self.assertAlmostEqual(float(np.max(np.diff(xi)) / np.min(np.diff(xi))), 10.0)

    def test_uniform_equilibrium_during_shrinkage(self) -> None:
        environment = EnvironmentData(
            np.asarray([0.0, 10.0]),
            np.asarray([28.0, 28.0]),
            np.asarray([2.55, 2.55]),
        )
        radius = RadiusData(
            np.asarray([0.0, 10.0]),
            np.asarray([0.02, 0.018]),
        )
        xi = make_reference_grid(38)
        system = MovingBoundarySystem(environment, radius, PhysicalParameters(), xi)
        state = np.concatenate(
            (np.full(xi.size, 28.0), np.full(xi.size, 2.55), np.asarray([0.0]))
        )
        self.assertLess(float(np.max(np.abs(system(5.0, state)))), 1.0e-12)

    def test_jacobian_pattern(self) -> None:
        pattern = make_jacobian_sparsity(41)
        self.assertEqual(pattern.shape, (83, 83))
        self.assertGreater(pattern.nnz, 0)

    def test_physical_extraction_leaves_outside_blank(self) -> None:
        xi = np.linspace(0.0, 1.0, 11)
        rows = np.asarray([xi])
        values = extract_physical_moistures(
            xi,
            rows,
            np.asarray([0.012]),
            np.arange(20) * 0.1,
        )[0]
        self.assertTrue(np.all(np.isfinite(values[:12])))
        self.assertTrue(np.all(np.isnan(values[12:])))

    def test_density_shrinkage_consistency_identity(self) -> None:
        xi = np.linspace(0.0, 1.0, 101)
        parameters = PhysicalParameters()
        diagnostic = density_shrinkage_consistency_from_field(
            xi,
            np.full_like(xi, parameters.initial_moisture_kg_kg),
            0.02,
            0.02,
            parameters,
            0.0,
        )
        self.assertAlmostEqual(float(diagnostic["dry_mass_proxy_ratio"]), 1.0)
        self.assertAlmostEqual(
            float(diagnostic["dry_mass_conserving_radius_m"]), 0.02
        )
        self.assertEqual(diagnostic["status"], "consistent_within_tolerance")

    def test_density_shrinkage_consistency_detects_unbalanced_radius(self) -> None:
        xi = np.linspace(0.0, 1.0, 101)
        parameters = PhysicalParameters()
        diagnostic = density_shrinkage_consistency_from_field(
            xi,
            np.full_like(xi, parameters.initial_moisture_kg_kg),
            0.02,
            0.012,
            parameters,
            1.0,
        )
        self.assertAlmostEqual(float(diagnostic["dry_mass_proxy_ratio"]), 0.36)
        self.assertAlmostEqual(float(diagnostic["volume_jacobian"]), 0.36)
        self.assertEqual(
            diagnostic["status"], "inconsistent_with_prescribed_affine_shrinkage"
        )


class InputTests(unittest.TestCase):
    def test_attachments(self) -> None:
        root = Path(__file__).resolve().parent.parent
        environment = load_environment(root / "A题" / "附件" / "附件1.xlsx")
        radius = load_radius(root / "A题" / "附件" / "附件2.xlsx")
        self.assertEqual(environment.times_s.size, 241)
        self.assertEqual(radius.times_s.size, 145)
        self.assertAlmostEqual(float(radius.radii_m[0]), 0.02)
        self.assertAlmostEqual(float(radius.radii_m[-1]), 0.01198)
        self.assertTrue(np.all(np.diff(radius.radii_m) <= 0.0))

    def test_fixed_environment_extension(self) -> None:
        root = Path(__file__).resolve().parent.parent
        environment = load_environment(root / "A题" / "附件" / "附件1.xlsx")
        fixed = EnvironmentData(
            environment.times_s,
            environment.temperatures_c,
            environment.moistures_kg_kg,
            extension_strategy="fixed",
            extension_temperature_c=50.0,
            extension_moisture_kg_kg=0.05,
        )
        fixed.validate()
        temperature, moisture = fixed.interpolate(20000.0)
        self.assertEqual(float(temperature), 50.0)
        self.assertEqual(float(moisture), 0.05)


if __name__ == "__main__":
    unittest.main(verbosity=2)
