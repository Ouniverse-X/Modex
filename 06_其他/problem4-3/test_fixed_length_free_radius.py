"""Regression tests for the fixed-length/free-radius Q4-3 model."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from fixed_length_free_radius import (
    NumericalConfig,
    appendix4_properties,
    fixed_length_geometry,
    load_environment,
    make_mass_faces,
    solve_model,
)


ROOT = Path(__file__).resolve().parent.parent


class GeometryTests(unittest.TestCase):
    def test_initial_uniform_state_recovers_initial_radius(self) -> None:
        faces = make_mass_faces(190, 0.1, 10)
        radius_faces, _, outer_radius, ratios = fixed_length_geometry(
            np.full(190, 2.55), faces
        )
        self.assertAlmostEqual(outer_radius, 0.02, places=14)
        self.assertAlmostEqual(radius_faces[0], 0.0, places=15)
        self.assertLess(float(np.max(np.abs(ratios - 1.0))), 5.0e-13)

    def test_uniform_state_matches_closed_form(self) -> None:
        faces = make_mass_faces(190, 0.1, 10)
        moisture = 0.15
        _, _, outer_radius, ratios = fixed_length_geometry(
            np.full(190, moisture), faces
        )
        initial_rho_d = appendix4_properties(
            np.array([2.55]), np.array([28.0])
        )[1][0]
        current_rho_d = appendix4_properties(
            np.array([moisture]), np.array([28.0])
        )[1][0]
        expected = 0.02 * np.sqrt(initial_rho_d / current_rho_d)
        self.assertAlmostEqual(outer_radius, expected, places=14)
        self.assertLess(float(np.max(np.abs(ratios - 1.0))), 5.0e-13)

    def test_nonuniform_state_preserves_each_dry_mass_cell(self) -> None:
        faces = make_mass_faces(380, 0.1, 10)
        moisture = np.linspace(2.0, 0.08, 380)
        radius_faces, _, outer_radius, ratios = fixed_length_geometry(
            moisture, faces
        )
        self.assertTrue(np.all(np.diff(radius_faces) > 0.0))
        self.assertLess(outer_radius, 0.02)
        self.assertLess(float(np.max(np.abs(ratios - 1.0))), 1.0e-12)


class SolverTests(unittest.TestCase):
    def test_short_run_is_finite_conservative_and_shrinking(self) -> None:
        environment = load_environment(ROOT / "A题" / "附件" / "附件1.xlsx")
        result = solve_model(
            environment,
            NumericalConfig(
                radial_cells=38,
                time_step_s=60.0,
                output_interval_s=600.0,
                maximum_time_h=4.1,
            ),
        )
        self.assertTrue(np.all(np.isfinite(result.predicted_radii_m)))
        self.assertAlmostEqual(result.times_s[-1], 4.1 * 3600.0, places=10)
        self.assertTrue(np.all(np.diff(result.predicted_radii_m) <= 1.0e-14))
        self.assertLess(
            result.diagnostics["max_local_dry_mass_relative_error"], 1.0e-11
        )
        self.assertLess(result.predicted_radii_m[-1], result.predicted_radii_m[0])


if __name__ == "__main__":
    unittest.main()
