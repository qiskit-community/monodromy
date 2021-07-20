"""
monodromy/xx_decompose/paths.py

Routines for producing right-angled paths through the Weyl alcove.  Consider a
set of native interactions with an associated minimal covering set of minimum-
cost circuit polytopes, as well as a target coordinate.  The coverage set
associates to the target coordinate a circuit type C = (O1 ... On) consisting of
a sequence of native interactions Oj.  A _path_ is a sequence (I P1 ... Pn) of
intermediate Weyl points, where Pj is accessible from P(j-1) by Oj.  A path is
said to be _right-angled_ when at each step one coordinate is fixed (up to
possible Weyl reflection) when expressed in canonical coordinates.

The key inputs to our method are:

+ A family of "b coordinates" which describe the target canonical coordinate.
+ A family of "a coordinates" which describe the source canonical coordinate.
+ A sequence of interaction strengths for which the b-coordinate can be modeled,
  with one selected to be stripped from the sequence ("beta").  The others are
  bundled as the sum of the sequence (s+), its maximum value (s1), and its
  second maximum value (s2).

Given the b-coordinate and a set of intersection strengths, the procedure for
backsolving for the a-coordinates is then extracted from the monodromy polytope.
See monodromy/static/qlr_table.py for a method for regenerating these extracted
polytopes.
"""

import numpy as np

from ..io.base import ConvexPolytopeData, PolytopeData

from .scipy import manual_get_random_vertex, polytope_has_element


xx_region_polytope = PolytopeData(convex_subpolytopes=[
    ConvexPolytopeData(inequalities=[[999, 0, 0, 0, -1, 0, 0, 0], [0, 0, 0, 0, 0, 1, -1, 0], [0, 0, 0, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, -2, 0, 0], [1, 0, 0, 0, 0, 0, 0, -2], [0, 0, 1, -1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, -1, -1, 0], [1, -1, -1, 0, 0, 0, 0, 0], [0, -1, -1, -1, 1, 0, 0, 1], [0, 1, -1, 0, 0, 0, 0, 0], [0, 1, -1, -1, 1, -2, 0, 1], [0, 1, -1, -1, 1, 0, 0, -1], [0, 0, 0, -1, 1, -1, 0, 0], [0, 0, -1, 0, 1, -1, -1, 1], [0, 0, 0, 0, 0, 0, 0, 1]], equalities=[], name='I ∩ A alcove ∩ A unreflected ∩ ah slant ∩ al frustrum ∩ B alcove ∩ B unreflected ∩ AF=B3'),
    ConvexPolytopeData(inequalities=[[999, 0, 0, 0, -1, 0, 0, 0], [0, 0, 0, 0, 0, 1, -1, 0], [0, 0, 0, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, -2, 0, 0], [1, 0, 0, 0, 0, 0, 0, -2], [0, 0, 1, -1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, -1, -1, 0], [1, -1, -1, 0, 0, 0, 0, 0], [1, -1, -1, -1, 1, -2, 0, 1], [0, 1, -1, 0, 0, 0, 0, 0], [-1, 1, -1, -1, 1, 0, 0, 1], [1, -1, -1, -1, 1, 0, 0, -1], [0, 0, 0, -1, 1, -1, 0, 0], [0, 0, -1, 0, 1, -1, -1, 1], [0, 0, 0, 0, 0, 0, 0, 1]], equalities=[], name='I ∩ A alcove ∩ A reflected ∩ ah strength ∩ al frustrum ∩ B alcove ∩ B reflected ∩ AF=B3'),
    ConvexPolytopeData(inequalities=[[999, 0, 0, 0, -1, 0, 0, 0], [0, 0, 0, 0, 0, 1, -1, 0], [0, 0, 0, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, -2, 0, 0], [1, 0, 0, 0, 0, 0, 0, -2], [0, 1, -1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], [1, -1, -1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, -1, -1, 0], [0, 1, -1, -1, 1, -2, 0, 1], [0, -1, -1, -1, 1, 0, 0, 1], [0, 0, 1, -1, 0, 0, 0, 0], [1, -1, 1, -1, 0, 0, 0, -1], [0, 1, 1, -1, 1, -2, 0, -1], [0, -1, 1, -1, 1, 0, 0, -1], [0, 0, 0, -1, 1, -1, -1, 1], [0, 0, 0, 0, 0, 0, 0, 1]], equalities=[], name='I ∩ A alcove ∩ A unreflected ∩ af slant ∩ al frustrum ∩ B alcove ∩ B unreflected ∩ AF=B1'),
    ConvexPolytopeData(inequalities=[[999, 0, 0, 0, -1, 0, 0, 0], [0, 0, 0, 0, 0, 1, -1, 0], [0, 0, 0, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, -2, 0, 0], [1, 0, 0, 0, 0, 0, 0, -2], [0, 1, -1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], [1, -1, -1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, -1, -1, 0], [-1, 1, -1, -1, 1, 0, 0, 1], [1, -1, -1, -1, 1, -2, 0, 1], [0, 0, 1, -1, 0, 0, 0, 0], [1, -1, 1, -1, 0, 0, 0, -1], [-1, 1, 1, -1, 1, 0, 0, -1], [1, -1, 1, -1, 1, -2, 0, -1], [0, 0, 0, -1, 1, -1, -1, 1], [0, 0, 0, 0, 0, 0, 0, 1]], equalities=[], name='I ∩ A alcove ∩ A reflected ∩ af strength ∩ al frustrum ∩ B alcove ∩ B reflected ∩ AF=B1')
])
"""
This polytope has the same point set as `b_polytope`, but it is broken into a
certain sum of convex regions. Each region is tagged with a name, and it is the
projection of the region of `xx_lift_polytope` with the corresponding name.

The coordinates are [k, b1, b2, b3, s+, s1, s2, beta].

NOTE: Includes an inequality which asserts 999 ≥ s+.  This bound is a dummy
      value, necessarily only because (our interface to) lrs does not deal well
      with unbounded regions.
"""


xx_lift_polytope = PolytopeData(convex_subpolytopes=[
    ConvexPolytopeData(inequalities=[[999, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0], [1, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0], [0, -1, -1, -1, 0, 0, 0, 1, 0, 0, 0], [0, 1, -1, -1, 0, 0, 0, 1, -2, 0, 0], [0, 0, -1, 0, 0, 0, 0, 1, -1, -1, 0], [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [1, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0], [0, 0, 0, 0, -1, -1, -1, 1, 0, 0, 1], [0, 0, 0, 0, 1, -1, -1, 1, -2, 0, 1], [0, 0, 0, 0, 1, -1, -1, 1, 0, 0, -1], [0, 0, 0, 0, 0, 0, -1, 1, -1, -1, 1], [0, 0, 0, 0, 0, 0, -1, 1, -1, 0, 0], [0, -1, -1, 0, 1, 1, 0, 0, 0, 0, 1], [2, -1, -1, 0, -1, -1, 0, 0, 0, 0, -1], [0, 1, 1, 0, -1, -1, 0, 0, 0, 0, 1], [0, -1, 1, 0, 1, -1, 0, 0, 0, 0, 1], [0, 1, -1, 0, -1, 1, 0, 0, 0, 0, 1], [0, 1, -1, 0, 1, -1, 0, 0, 0, 0, -1]], equalities=[[0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0]], name='I ∩ A alcove ∩ A unreflected ∩ ah slant ∩ al frustrum ∩ B alcove ∩ B unreflected ∩ AF=B3'),
    ConvexPolytopeData(inequalities=[[999, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0], [1, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0], [0, -1, -1, -1, 0, 0, 0, 1, 0, 0, 0], [0, -1, -1, 1, 0, 0, 0, 1, -2, 0, 0], [0, 0, -1, 0, 0, 0, 0, 1, -1, -1, 0], [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [1, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0], [0, 0, 0, 0, -1, -1, -1, 1, 0, 0, 1], [0, 0, 0, 0, 1, -1, -1, 1, -2, 0, 1], [0, 0, 0, 0, 1, -1, -1, 1, 0, 0, -1], [0, 0, 0, 0, 0, 0, -1, 1, -1, -1, 1], [0, 0, 0, 0, 0, 0, -1, 1, -1, 0, 0], [0, -1, -1, 0, 0, 1, 1, 0, 0, 0, 1], [2, -1, -1, 0, 0, -1, -1, 0, 0, 0, -1], [0, 1, 1, 0, 0, -1, -1, 0, 0, 0, 1], [0, -1, 1, 0, 0, 1, -1, 0, 0, 0, 1], [0, 1, -1, 0, 0, -1, 1, 0, 0, 0, 1], [0, 1, -1, 0, 0, 1, -1, 0, 0, 0, -1]], equalities=[[0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0]], name='I ∩ A alcove ∩ A unreflected ∩ af slant ∩ al frustrum ∩ B alcove ∩ B unreflected ∩ AF=B1'),
    ConvexPolytopeData(inequalities=[[999, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0], [1, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0], [1, -1, -1, -1, 0, 0, 0, 1, -2, 0, 0], [-1, -1, -1, 1, 0, 0, 0, 1, 0, 0, 0], [0, 0, -1, 0, 0, 0, 0, 1, -1, -1, 0], [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [1, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0], [-1, 0, 0, 0, 1, -1, -1, 1, 0, 0, 1], [1, 0, 0, 0, -1, -1, -1, 1, 0, 0, -1], [1, 0, 0, 0, -1, -1, -1, 1, -2, 0, 1], [0, 0, 0, 0, 0, 0, -1, 1, -1, -1, 1], [0, 0, 0, 0, 0, 0, -1, 1, -1, 0, 0], [0, -1, -1, 0, 0, 1, 1, 0, 0, 0, 1], [2, -1, -1, 0, 0, -1, -1, 0, 0, 0, -1], [0, 1, 1, 0, 0, -1, -1, 0, 0, 0, 1], [0, -1, 1, 0, 0, 1, -1, 0, 0, 0, 1], [0, 1, -1, 0, 0, -1, 1, 0, 0, 0, 1], [0, 1, -1, 0, 0, 1, -1, 0, 0, 0, -1]], equalities=[[0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0]], name='I ∩ A alcove ∩ A reflected ∩ af strength ∩ al frustrum ∩ B alcove ∩ B reflected ∩ AF=B1'),
    ConvexPolytopeData(inequalities=[[999, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [1, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0], [1, -1, 0, -1, 0, 0, 0, 0, 0, 0, 0], [1, -1, -1, -1, 0, 0, 0, 1, -2, 0, 0], [-1, 1, -1, -1, 0, 0, 0, 1, 0, 0, 0], [0, 0, -1, 0, 0, 0, 0, 1, -1, -1, 0], [0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [1, 0, 0, 0, -1, -1, 0, 0, 0, 0, 0], [-1, 0, 0, 0, 1, -1, -1, 1, 0, 0, 1], [1, 0, 0, 0, -1, -1, -1, 1, 0, 0, -1], [1, 0, 0, 0, -1, -1, -1, 1, -2, 0, 1], [0, 0, 0, 0, 0, 0, -1, 1, -1, -1, 1], [0, 0, 0, 0, 0, 0, -1, 1, -1, 0, 0], [0, -1, -1, 0, 1, 1, 0, 0, 0, 0, 1], [2, -1, -1, 0, -1, -1, 0, 0, 0, 0, -1], [0, 1, 1, 0, -1, -1, 0, 0, 0, 0, 1], [0, -1, 1, 0, 1, -1, 0, 0, 0, 0, 1], [0, 1, -1, 0, -1, 1, 0, 0, 0, 0, 1], [0, 1, -1, 0, 1, -1, 0, 0, 0, 0, -1]], equalities=[[0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 0]], name='I ∩ A alcove ∩ A reflected ∩ ah strength ∩ al frustrum ∩ B alcove ∩ B reflected ∩ AF=B3')
])
"""
This polytope is a union of convex regions which project to cover `b_polytope`
(namely, they project to the regions in `xx_region_polytope`), with the added
property that each specialization of (af, b1, b2, b3, s+, s1, s2, beta) yields
a _nonempty_ polytope on (ah, al).  Points in this region can then be used to
calculate circuits using `decompose_xxyy_into_xxyy_xx`.

The coordinates are [k, ah, al, af, b1, b2, b3, s+, s1, s2, beta].

NOTE: Includes an inequality which asserts 999 ≥ s+.  This bound is a dummy
      value, necessarily only because (our interface to) lrs does not deal well
      with unbounded regions.
"""


def polytope_from_strengths(strengths, scale_factor=1):
    """
    Given a list of interaction `strengths` each lying in [0, 1/2], produces a
    `Polytope` so that a normalized positive canonical coordinate (see
    monodromy/coordinates.py) belongs to the polytope if and only if there
    exists a circuit modeling it of the form

        local * CAN(strength[0]) * local * CAN(strength[1]) * local * ...
            ... * local * CAN(strength[-1]) * local .

    NOTE: A strength of 0 correspond to the identity gate, and a strength of
          1/2 lies in the local equivalence class of a CX (or a CZ).
    NOTE: `scale_factor` rescales the resulting polytope. For scale_factor=pi/2,
          this places CX at pi/4, in agreement with canonical coordinates.
    """
    strengths = strengths + [0, 0]
    total_strength = 0
    ultimate_strength, penultimate_strength = 0, 0
    for strength in strengths:
        total_strength += strength
        if strength > ultimate_strength:
            ultimate_strength, penultimate_strength = strength, ultimate_strength
        elif strength > penultimate_strength:
            penultimate_strength = strength

    polytope = PolytopeData(convex_subpolytopes=[
        ConvexPolytopeData([
                [scale_factor, -1,  0,  0],  # 1  >= c1
                [           0,  1, -1,  0],  # c1 >= c2
                [           0,  0,  1, -1],  # c2 >= c3
                [scale_factor, -1, -1,  0],  # 1 - c1 >= c2
                [           0,  0,  0,  1],  # c3 >=  0, the C2 inequality
                [total_strength, -1, -1, -1],  # strength bound
                [total_strength - 2 * ultimate_strength, 1, -1, -1],  # slant bound
                [total_strength - ultimate_strength - penultimate_strength, 0, 0, -1],  # frustrum bound
            ],
            name='·'.join(f"RZX({x / np.pi:.5f}π)" for x in strengths[:-2]) + 'unreflected'),
        ConvexPolytopeData([
                [scale_factor, -1,  0,  0],  # 1  >= c1
                [           0,  1, -1,  0],  # c1 >= c2
                [           0,  0,  1, -1],  # c2 >= c3
                [scale_factor, -1, -1,  0],  # 1 - c1 >= c2
                [           0,  0,  0,  1],  # c3 >=  0, the C2 inequality
                [total_strength - scale_factor, 1, -1, -1],  # strength bound
                [scale_factor + total_strength - 2 * ultimate_strength, -1, -1, -1], # slant bound
                [total_strength - ultimate_strength - penultimate_strength, 0, 0, -1],  # frustrum bound
            ],
            name='·'.join(f"RZX({x / np.pi:.5f}π)" for x in strengths[:-2]) + 'reflected')
    ])

    return polytope


def get_augmented_coordinate(target_coordinate, strengths):
    *strengths, beta = strengths
    strengths = sorted(strengths + [0, 0])
    interaction_coordinate = [sum(strengths), strengths[-1], strengths[-2], beta]
    return [*target_coordinate, *interaction_coordinate]


def decomposition_hop(target_coordinate, strengths):
    """
    Given a `target_coordinate` and a list of interaction `strengths`, produces
    a new canonical coordinate which is one step back along `strengths`.

    `target_coordinate` is taken to be in positive canonical coordinates, and
    the entries of strengths are taken to be
    [0, 1/2], so that (sj, 0, 0) is a normalized positive canonical coordinate.
    """

    target_coordinate = [x / (np.pi / 2) for x in target_coordinate]
    strengths = [x / np.pi for x in strengths]

    augmented_coordinate = get_augmented_coordinate(target_coordinate, strengths)
    specialized_polytope = None
    for cp in xx_region_polytope.convex_subpolytopes:
        if not polytope_has_element(cp, augmented_coordinate):
            continue
        if "AF=B1" in cp.name:
            af, bh, bl = target_coordinate
        elif "AF=B2" in cp.name:
            bh, af, bl = target_coordinate
        elif "AF=B3" in cp.name:
            bh, bl, af = target_coordinate
        else:
            raise ValueError("Couldn't find a coordinate to fix.")

        raw_convex_polytope = next(
            (cpp for cpp in xx_lift_polytope.convex_subpolytopes
             if cpp.name == cp.name), None)

        coefficient_dict = {}
        for inequality in raw_convex_polytope.inequalities:
            if inequality[1] == 0 and inequality[2] == 0:
                continue
            offset = (
                    inequality[0]  # old constant term
                    + inequality[3] * af
                    + inequality[4] * augmented_coordinate[0]  # b1
                    + inequality[5] * augmented_coordinate[1]  # b2
                    + inequality[6] * augmented_coordinate[2]  # b3
                    + inequality[7] * augmented_coordinate[3]  # s+
                    + inequality[8] * augmented_coordinate[4]  # s1
                    + inequality[9] * augmented_coordinate[5]  # s2
                    + inequality[10] * augmented_coordinate[6]  # beta
            )

            if offset <= coefficient_dict.get((inequality[1], inequality[2]),
                                              offset):
                coefficient_dict[(inequality[1], inequality[2])] = offset

        specialized_polytope = PolytopeData(convex_subpolytopes=[
            ConvexPolytopeData(inequalities=[
                [v, h, l] for ((h, l), v) in coefficient_dict.items()
            ])
        ])

        break

    if specialized_polytope is None:
        raise ValueError("Failed to match a constrained_polytope summand.")

    ah, al = manual_get_random_vertex(specialized_polytope)
    return [x * (np.pi / 2) for x in sorted([ah, al, af], reverse=True)]
