from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from true_density_nonaffine import (
    _center_value,
    appendix4_properties,
    conservative_geometry,
    make_mass_faces,
)


class TrueDensityNonAffineTests(unittest.TestCase):
    def test_initial_properties(self) -> None:
        wet, dry, cp, k, diffusivity = appendix4_properties(
            np.asarray([2.55]), np.asarray([28.0])
        )
        self.assertAlmostEqual(float(wet[0]), 989.5)
        self.assertAlmostEqual(float(dry[0]), 989.5 / 3.55)
        self.assertGreater(float(cp[0]), 0.0)
        self.assertGreater(float(k[0]), 0.0)
        self.assertGreater(float(diffusivity[0]), 0.0)

    def test_mass_grid(self) -> None:
        faces = make_mass_faces(190, 0.1, 10)
        self.assertEqual(faces.size, 191)
        self.assertEqual(float(faces[0]), 0.0)
        self.assertEqual(float(faces[-1]), 1.0)
        self.assertTrue(np.all(np.diff(faces) > 0.0))

    def test_uniform_initial_state_recovers_initial_geometry(self) -> None:
        faces = make_mass_faces(190, 0.1, 10)
        moisture = np.full(190, 2.55)
        radius_faces, _, length, ratios = conservative_geometry(
            moisture, faces, 0.02
        )
        self.assertAlmostEqual(float(radius_faces[-1]), 0.02, places=14)
        self.assertAlmostEqual(length, 0.25, places=13)
        self.assertLess(float(np.max(np.abs(ratios - 1.0))), 1.0e-12)

    def test_nonuniform_moisture_produces_nonaffine_mapping_and_conserves_mass(self) -> None:
        faces = make_mass_faces(190, 0.1, 10)
        centers = 0.5 * (faces[:-1] + faces[1:])
        moisture = 0.1 + 0.2 * (1.0 - centers)
        radius_faces, _, _, ratios = conservative_geometry(
            moisture, faces, 0.012
        )
        affine = 0.012 * np.sqrt(faces)
        self.assertGreater(float(np.max(np.abs(radius_faces - affine))), 1.0e-7)
        self.assertLess(float(np.max(np.abs(ratios - 1.0))), 1.0e-12)

    def test_axis_reconstruction_is_exact_for_linear_function_of_mass_coordinate(
        self,
    ) -> None:
        centers = np.asarray([0.005, 0.015, 0.025])
        cell_values = 2.0 + 3.0 * centers
        reconstructed = _center_value(centers, cell_values)
        self.assertAlmostEqual(float(reconstructed), 2.0, places=14)


if __name__ == "__main__":
    unittest.main(verbosity=2)
