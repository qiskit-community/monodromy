"""
test/test_elimination.py

Tests for monodromy/elimination.py .
"""

import qiskit

import ddt
import unittest

from monodromy.coordinates import monodromy_alcove_c2_pcs
from monodromy.elimination import *
from monodromy.polytopes import make_convex_polytope


@ddt.ddt
class TestMonodromyElimination(unittest.TestCase):
    """Check various elimination routines."""

    def test_cube_from_cylinders(self):
        """Build a cube out of cylinderized intervals."""

        interval = make_convex_polytope([
            [0, 1], [1, -1]
        ])

        cube = cylinderize(interval, [0, 1], 4) \
            .intersect(cylinderize(interval, [0, 2], 4)) \
            .intersect(cylinderize(interval, [0, 3], 4))

        expected = make_convex_polytope([
            [0, 1, 0, 0], [1, -1,  0,  0],
            [0, 0, 1, 0], [1,  0, -1,  0],
            [0, 0, 0, 1], [1,  0,  0, -1],
        ])

        self.assertTrue(cube.contains(expected))
        self.assertTrue(expected.contains(cube))

    def test_project_cylinderize_inverses(self):
        """Test that projection after cylinderization is a NOP."""
        original = monodromy_alcove_c2_pcs
        cylinderized = cylinderize(original, [0, 1, 2, 3, 4], 5)
        projected = project(cylinderized, 4)
        self.assertTrue(projected.contains(original))
        self.assertTrue(original.contains(projected))
