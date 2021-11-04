"""
test/test_coverage.py

Tests for monodromy/coverage.py .
"""

import ddt
import unittest

from monodromy.polytopes import ConvexPolytope, PolytopeVolume
from monodromy.coverage import *
from monodromy.static.examples import exactly, identity_polytope


@ddt.ddt
class TestMonodromyCoverage(unittest.TestCase):
    """Check various coverage set routines."""

    def test_cx_coverage(self):
        """Test Section III of quant-ph/0308045 ."""
        cx_polytope = CircuitPolytope(
            convex_subpolytopes=exactly(
                Fraction(1, 4), Fraction(1, 4), Fraction(-1, 4)
            ).convex_subpolytopes,
            cost=1,
            operations=["CX"]
        )
        coverage_set = build_coverage_set([cx_polytope])
        self.assertEqual(
            {(), ("CX",), ("CX", "CX"), ("CX", "CX", "CX")},
            {tuple(item.operations) for item in coverage_set}
        )
        cx0_polytope = next(p for p in coverage_set if p.operations == [])
        cx1_polytope = next(p for p in coverage_set if p.operations == ["CX"])
        cx2_polytope = next(p for p in coverage_set if p.operations == ["CX", "CX"])
        cx3_polytope = next(p for p in coverage_set if p.operations == ["CX", "CX", "CX"])
        self.assertTrue(cx3_polytope.contains(cx2_polytope))
        self.assertFalse(cx2_polytope.contains(cx3_polytope))
        self.assertTrue(cx2_polytope.contains(cx1_polytope))
        self.assertFalse(cx1_polytope.contains(cx2_polytope))
        self.assertFalse(cx1_polytope.contains(cx0_polytope))
        self.assertFalse(cx0_polytope.contains(cx1_polytope))
        self.assertTrue(identity_polytope.contains(cx0_polytope))
        self.assertTrue(cx3_polytope.contains(monodromy_alcove_c2))
        self.assertTrue(cx_polytope.contains(cx1_polytope))

        expected_cx2 = Polytope(convex_subpolytopes=[
            ConvexPolytope.convex_hull(
                [[Fraction(0), Fraction(0), Fraction(0)],
                 [Fraction(1, 2), Fraction(1, 2), -Fraction(1, 2)],
                 [Fraction(1, 2), Fraction(0), Fraction(0)]]
            )
        ])
        self.assertTrue(expected_cx2.contains(cx2_polytope))
        self.assertTrue(cx2_polytope.contains(expected_cx2))

    def test_sqrtcx_coverage(self):
        """Test Example 49 of Peterson-Crooks-Smith."""
        sqrtcx_polytope = CircuitPolytope(
            convex_subpolytopes=exactly(
                Fraction(1, 8), Fraction(1, 8), Fraction(-1, 8)
            ).convex_subpolytopes,
            cost=1/2 + 1e-10,
            operations=["sqrtCX"]
        )
        coverage_set = build_coverage_set([sqrtcx_polytope])

        self.assertEqual(
            {tuple(x.operations) for x in coverage_set},
            {tuple("sqrtCX" for _ in range(0, j)) for j in range(0, 6 + 1)}
        )

        expected_volumes = {
            ('sqrtCX',) * 0: PolytopeVolume(dimension=0, volume=Fraction(1, 1)),
            ('sqrtCX',) * 1: PolytopeVolume(dimension=0, volume=Fraction(2, 1)),
            ('sqrtCX',) * 2: PolytopeVolume(dimension=2, volume=Fraction(1, 16)),
            ('sqrtCX',) * 3: PolytopeVolume(dimension=3, volume=Fraction(1, 96)),
            ('sqrtCX',) * 4: PolytopeVolume(dimension=3, volume=Fraction(5, 288)),
            ('sqrtCX',) * 5: PolytopeVolume(dimension=3, volume=Fraction(47, 2304)),
            ('sqrtCX',) * 6: PolytopeVolume(dimension=3, volume=Fraction(1, 48))
        }

        for x in coverage_set:
            self.assertEqual(x.volume, expected_volumes[tuple(x.operations)])

    def test_cx_iswap_cphase_comparison(self):
        """Test Lemma 46 of Peterson-Crooks-Smith."""
        cx_polytope = CircuitPolytope(
            convex_subpolytopes=exactly(
                Fraction(1, 4), Fraction(1, 4), Fraction(-1, 4)
            ).convex_subpolytopes,
            cost=1,
            operations=["CX"]
        )
        iswap_polytope = CircuitPolytope(
            convex_subpolytopes=exactly(
                Fraction(1, 2), Fraction(0), Fraction(0),
            ).convex_subpolytopes,
            cost=1,
            operations=["ISWAP"]
        )
        cphase_polytope = CircuitPolytope(
            convex_subpolytopes=[ConvexPolytope(
                inequalities=[[0, 1, 0, 0], [1, -2, 0, 0]],
                equalities=[[0, 1, -1, 0], [0, 1, 0, 1]]
            )],
            cost=1,
            operations=["CPHASE"],
        )
        cx_coverage_set = build_coverage_set([cx_polytope])
        iswap_coverage_set = build_coverage_set([iswap_polytope])
        cphase_coverage_set = build_coverage_set([cphase_polytope])

        # compare depth 2 polytopes
        cx2_polytope = next(p for p in cx_coverage_set if 2 == len(p.operations))
        iswap2_polytope = next(p for p in iswap_coverage_set if 2 == len(p.operations))
        cphase2_polytope = next(p for p in cphase_coverage_set if 2 == len(p.operations))
        self.assertTrue(cx2_polytope.contains(iswap2_polytope))
        self.assertTrue(iswap2_polytope.contains(cphase2_polytope))
        self.assertTrue(cphase2_polytope.contains(cx2_polytope))

        # compare depth 3 polytopes
        cx3_polytope = next(p for p in cx_coverage_set if 3 == len(p.operations))
        iswap3_polytope = next(p for p in iswap_coverage_set if 3 == len(p.operations))
        cphase3_polytope = next(p for p in cphase_coverage_set if 3 == len(p.operations))
        self.assertTrue(cx3_polytope.contains(iswap3_polytope))
        self.assertTrue(iswap3_polytope.contains(cphase3_polytope))
        self.assertTrue(cphase3_polytope.contains(cx3_polytope))
