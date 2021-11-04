"""
test/test_coordinates.py

Tests for monodromy/coordinates.py .
"""

from fractions import Fraction
import unittest

import ddt
import qiskit

from monodromy.coordinates import *
from monodromy.static.examples import exactly

epsilon = 0.001


@ddt.ddt
class TestMonodromyCoordinates(unittest.TestCase):
    """Check various coordinate routines."""

    def assertApproximate(self, a, b):
        self.assertTrue(np.all(np.abs(np.array(a) - np.array(b)) < epsilon),
                        msg=f"a: {a}\nb: {b}\n")

    @ddt.data((qiskit.circuit.library.SwapGate().to_matrix(),
               (1/4, 1/4, 1/4, -3/4)),
              (qiskit.circuit.library.CXGate().to_matrix(),
               (1/4, 1/4, -1/4, -1/4)),)
    @ddt.unpack
    def test_unitary_to_monodromy_coordinate(self, matrix, target):
        """Check the monodromy coordinates of some operators."""
        self.assertApproximate(
            np.array(target),
            np.array(unitary_to_monodromy_coordinate(matrix))
        )

    @ddt.data(
        ((1/4, 1/4, -1/4), (np.pi / 4, 0, 0)),  # CZ
        ((1/2, 0, 0), (np.pi/4, np.pi/4, 0))  # ISWAP
    )
    @ddt.unpack
    def test_monodromy_to_positive_canonical_coordinate(
            self, in_coord, out_coord
    ):
        """Check the conversion from monodromy to positive canonical coords."""
        self.assertApproximate(
            out_coord,
            monodromy_to_positive_canonical_coordinate(*in_coord)
        )

    @ddt.data(
        ((np.pi / 4, 0, 0), (1 / 4, 1 / 4, -1 / 4)),  # CZ
        ((np.pi / 4, np.pi / 4, 0), (1 / 2, 0, 0))  # ISWAP
    )
    @ddt.unpack
    def test_positive_canonical_to_monodromy_coordinate(
            self, in_coord, out_coord
    ):
        """Check the conversion from positive canonical to monodromy coords."""
        self.assertApproximate(
            out_coord,
            positive_canonical_to_monodromy_coordinate(*in_coord)
        )

    @ddt.data(
        (monodromy_alcove_c2, positive_canonical_alcove_c2),
        (monodromy_alcove, positive_canonical_alcove),
        (exactly(Fraction(1, 4), Fraction(1, 4), -Fraction(1, 4)),
         exactly(Fraction(1, 2), Fraction(0), Fraction(0))),  # CZ
    )
    @ddt.unpack
    def test_monodromy_to_positive_canonical_polytope(self, input, expected):
        """Check the conversion of _polytopes_ from monodromy to positive canonical coords."""
        result = monodromy_to_positive_canonical_polytope(input)
        self.assertTrue(expected.contains(result) and
                        result.contains(expected))
