"""
test/test_volume.py

Tests for monodromy/volume.py .
"""

import qiskit

import ddt
import unittest

from monodromy.polytopes import make_convex_polytope
from monodromy.volume import *

epsilon = 0.001


@ddt.ddt
class TestMonodromyVolume(unittest.TestCase):
    """Check various volume routines."""

    def volume_fn(self, dim):
        counter = 0

        def volume_fn(convex_polytope):
            nonlocal counter
            counter += 1
            if convex_polytope.volume.dimension == dim:
                return convex_polytope.volume.volume
            elif convex_polytope.volume.dimension < dim:
                return 0
            else:
                raise ValueError("Unexpectedly large volume.")

        def get_counter():
            nonlocal counter
            return counter

        return volume_fn, get_counter

    def test_null_efficiency(self):
        """Test that empty polytopes have skipped children"""
        polytope = make_convex_polytope([
            [0,  1], [1, -1],
        ]).union(make_convex_polytope([
            [0,  1], [-1, -1],
        ]))

        volume_fn, get_counter = self.volume_fn(1)
        alternating_sum(polytope, volume_fn)
        naive_count = 2 ** len(polytope.convex_subpolytopes) - 1
        self.assertLess(get_counter(), naive_count)

    def test_duplication_efficiency(self):
        """Test that equal polytopes have skipped children."""
        polytope = make_convex_polytope([
            [0,  1], [1, -1],  # [0, 1]
        ]).union(make_convex_polytope([
            [0,  1], [2, -1],  # [0, 2]
        ])).union(make_convex_polytope([
            [-1, 1], [1, -1]   # [-1, 1]
        ]))

        volume_fn, get_counter = self.volume_fn(1)
        alternating_sum(polytope, volume_fn)
        naive_count = 2 ** len(polytope.convex_subpolytopes) - 1
        self.assertLess(get_counter(), naive_count)
