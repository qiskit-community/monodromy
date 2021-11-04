"""
test/test_haar.py

Tests for monodromy/haar.py .
"""

import ddt
import unittest

from fractions import Fraction

from monodromy.coordinates import positive_canonical_alcove_c2
from monodromy.coverage import build_coverage_set, CircuitPolytope
from monodromy.haar import *
from monodromy.static.examples import exactly

epsilon = 0.001


@ddt.ddt
class TestMonodromyHaar(unittest.TestCase):
    """Check various Haar volume routines."""

    def test_alcove_haar_volume(self):
        """Check that the alcove has unit Haar volume."""
        self.assertEqual(1.0, haar_volume(positive_canonical_alcove_c2))

    def test_expected_interaction_strength(self):
        """Check that the Haar-random expected interaction strength is 3/2."""
        left_expected_interaction_strength = haar_volume(
            positive_canonical_alcove_c2.intersect(make_convex_polytope([
                [1, -2, 0, 0]
            ])),
            Polynomial.from_linear_list([0, 1, 1, 1])
        )
        right_expected_interaction_strength = haar_volume(
            positive_canonical_alcove_c2.intersect(make_convex_polytope([
                [-1, 2, 0, 0]
            ])),
            Polynomial.from_linear_list([np.pi, -1, 1, 1])
        )

        self.assertAlmostEqual(
            left_expected_interaction_strength,
            right_expected_interaction_strength,
            delta=epsilon
        )

        self.assertAlmostEqual(
            left_expected_interaction_strength + right_expected_interaction_strength,
            (np.pi / 2) * 3 / 2
        )

    def test_cost_statistics(self):
        """
        Check that Haar-random cost statistics are computed correctly in a known
        example.
        """
        cx_polytope = CircuitPolytope(
            convex_subpolytopes=exactly(
                Fraction(1, 4), Fraction(1, 4), Fraction(-1, 4)
            ).convex_subpolytopes,
            cost=1,
            operations=["CX"]
        )
        coverage_set = build_coverage_set([cx_polytope])
        statistics = cost_statistics(coverage_set, 0, 1)
        expected = {
            "average_cost": 3.0,
            "average_overshot": 1.5,
            "sigma_cost": 0.0,
            "sigma_overshot": 0.28527,
        }
        for k in statistics.keys():
            self.assertAlmostEqual(
                statistics[k], expected[k], delta=epsilon
            )
