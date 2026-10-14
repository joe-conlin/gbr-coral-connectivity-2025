"""Small checks for the biological movement rules."""

import unittest

import numpy as np

from model.millepora import (
    age_limited_windage,
    empirical_target_fraction,
    surface_bounded_position,
    surface_velocity,
    target_depth,
)


class MovementRuleTests(unittest.TestCase):
    def test_surface_bound(self):
        moved = surface_bounded_position(np.array([-0.1]), np.array([0.001]), 180)
        self.assertEqual(float(moved[0]), 0.0)

    def test_surface_phase_ends_at_eight_hours(self):
        settings = {"surface_phase_hours": 8.0, "buoyancy_velocity_m_s": 0.0005}
        result = surface_velocity(np.array([0, 8 * 3600]), settings)
        np.testing.assert_allclose(result, [0.0005, 0.0])

    def test_windage_ends_at_eight_hours(self):
        settings = {"maximum_age_hours": 8.0, "wind_drift_factor": 0.03}
        result = age_limited_windage(np.array([0, 8 * 3600]), settings)
        np.testing.assert_allclose(result, [0.03, 0.0])

    def test_source_depth_limits_target(self):
        full = target_depth(100.0, 0.5, False, 20.0)
        limited = target_depth(100.0, 0.5, True, 20.0)
        self.assertEqual(float(full), 50.0)
        self.assertEqual(float(limited), 10.0)

    def test_single_bin_distribution(self):
        settings = {
            "minimum_depth_fraction": 0.05,
            "maximum_depth_fraction": 0.95,
            "distribution_anchors": [
                {"age_hours": 8.0, "probabilities": [1.0] + [0.0] * 9},
                {"age_hours": 59.5, "probabilities": [1.0] + [0.0] * 9},
            ],
        }
        result = empirical_target_fraction(
            np.array([8 * 3600, 59.5 * 3600]),
            np.array([0.1, 0.9]),
            settings,
        )
        np.testing.assert_allclose(result, [0.05, 0.05])


if __name__ == "__main__":
    unittest.main()
