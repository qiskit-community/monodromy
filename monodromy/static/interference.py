"""
monodromy/static/interference.py

Static data related to XX-interaction polytopes and interference inequalities.
We make two main claims:

(1) For a sequence [s1, s2, ..., sn] of interaction strengths, the space of
    circuits which can be written as

        local CAN(s1) local ... CAN(s(n-1)) local CAN(sn) local

    are exactly carved out by a triple of inequalities

         a1 + a2 + a3 ≤                            s1 + ... + sn
        -a1 + a2 + a3 ≤ min_j       { -2 sj     + (s1 + ... + sn) }
                   a3 ≤ min_{j ≠ k} { - sj - sk + (s1 + ... + sn) }

    and a similarly described rho-reflected component.  We refer to this as the
    XX-interaction polytope for this (unordered) sequence of strengths.

(2) For any point (b1, b2, b3) belonging to the XX-interaction polytope for
    [s1, ..., sn, beta], we can always find a point (a1, a2, a3) belonging to
    the XX-interaction polytope for [s1, ..., sn] which shares a coordinate in
    common with b and whose remaining components are subject to the
    "interference inequalities":

        |ah - al - beta| ≤ bh - bl ≤              ah - al + beta ,
         ah + al - beta  ≤ bh + bl ≤ 2m - | 2m - (ah + al + beta) |,

    where m is the value of the canonical coordinate of CX.

This file contains polytope definitions and factories related to these claims,
as well as routines used to give computer-aided verifications of the claims.
"""

import numpy as np

from ..coordinates import monodromy_alcove, monodromy_alcove_c2, \
    monodromy_to_positive_canonical_polytope
from ..coverage import rho_reflect
from ..elimination import cylinderize, project
from ..io.base import ConvexPolytopeData, PolytopeData
from ..polytopes import ConvexPolytope, make_convex_polytope, Polytope

from .examples import empty_polytope
from .qlr_table import qlr_polytope


strength_polytope = make_convex_polytope([
    # k s+ s1 s2 beta
    [999, -1,  0,  0,  0],  # 999 - s+ ≥ 0
    [  0,  1, -1, -1,  0],  # s+ - s1 - s2 ≥ 0
    [  0,  0,  1, -1,  0],  # s1 - s2 ≥ 0
    [  0,  0,  0,  1,  0],  # s2 ≥ 0
    [  1,  0, -2,  0,  0],  # 1/2 - s1 ≥ 0
    [  1,  0,  0,  0, -2],  # 1/2 - beta ≥ 0
    [  0,  0,  0,  0,  1],  # beta ≥ 0
], name="I")
"""
This polytope puts constraints on the four strength coordinates used in the
proof of the main local theorem: given strengths (alpha_1, ..., alpha_n, beta),
s+ models the sum of the alphas, s1 their maximum, s2 the second maximum, and
beta is the strength we intend to strip off.
"""


b_polytope = make_convex_polytope([
    # k b1 b2 b3 s+ s1 s2 beta
    [0,  1, -1,  0, 0, 0, 0, 0],  # b1 - b2 ≥ 0
    [0,  0,  1, -1, 0, 0, 0, 0],  # b2 - b3 ≥ 0
    [0,  0,  0,  1, 0, 0, 0, 0],  # b3 ≥ 0
    [1, -1, -1,  0, 0, 0, 0, 0],  # 1 - b1 - b2 ≥ 0
], name="B alcove").intersect(
    # MGC inequalities
    empty_polytope.union(make_convex_polytope([
        # strength
        [0, -1, -1, -1, 1,  0,  0,  1],  # - b1 - b2 - b3 + s+ + beta ≥ 0
        # slant
        [0,  1, -1, -1, 1, -2,  0,  1],  # b1 - b2 - b3 + s+ - 2 s1 + beta ≥ 0
        [0,  1, -1, -1, 1,  0,  0, -1],  # b1 - b2 - b3 + s+ - beta ≥ 0
        # frustrum
        [0,  0,  0, -1, 1, -1, -1,  1],  # - b3 + s+ - s1 - s2 + beta ≥ 0
        [0,  0,  0, -1, 1, -1,  0,  0],  # - b3 + s+ - s1 ≥ 0
    ], name="B unreflected")).union(make_convex_polytope([
        # strength
        [-1 , 1, -1, -1, 1,  0,  0,  1],  # -1 + b1 - b2 - b3 + s+ + beta ≥ 0
        # slant
        [ 1, -1, -1, -1, 1,  0,  0, -1],  # 1 - b1 - b2 - b3 + s+ - beta ≥ 0
        [ 1, -1, -1, -1, 1, -2,  0,  1],  # 1 - b1 - b2 - b3 + s+ + beta - 2s1 ≥ 0
        # frustrum
        [ 0,  0,  0, -1, 1, -1, -1,  1],  # - b3 + s+ - s1 - s2 + beta ≥ 0
        [ 0,  0,  0, -1, 1, -1,  0,  0],  # - b3 + s+ - s1 ≥ 0
    ], name="B reflected"))
)
"""
This polytope records the constraints on the "b coordinates", i.e., the "target"
of the edge in the canonical path.  They're taken to be sorted, to belong to the
normalized positive canonical alcove, and to obey the constraints of the main
global theorem for the sequence of strengths [s1, s2, ..., sn, beta] (see
`polytope_from_strengths`).
"""


a_polytope = make_convex_polytope([
    # k ah al af s+ s1 s2
    [0,  1, -1,  0, 0, 0, 0],  # ah - al ≥ 0
    [0,  0,  1,  0, 0, 0, 0],  # al ≥ 0
    [0,  0,  0,  1, 0, 0, 0],  # af ≥ 0
    [1, -1, -1,  0, 0, 0, 0],  # 1 - ah - al ≥ 0
    [1, -1,  0, -1, 0, 0, 0],  # 1 - ah - af ≥ 0
], name="A alcove").intersect(
    # MGC inequalities
    empty_polytope.union(
        # unreflected
        make_convex_polytope([
            # strength bound
            [0, -1, -1, -1, 1, 0, 0],  # - ah - al - af + s+ ≥ 0
        ], name="A unreflected").intersect(
            # slant bound
            empty_polytope.union(make_convex_polytope([
                [0,  1, -1, -1, 1, -2, 0]  # ah - al - af + s+ - 2 s1 ≥ 0
            ], name="ah slant")).union(make_convex_polytope([
                [0, -1, -1,  1, 1, -2, 0]  # - ah - al + af + s+ - 2 s1 ≥ 0
            ], name="af slant"))
        )
    ).union(
        # reflected
        make_convex_polytope([
            # slant bound
            [1, -1, -1, -1, 1, -2, 0],  # 1 - ah - al - af + s+ - 2 s1 ≥ 0
        ], name="A reflected").intersect(
            # strength bound
            empty_polytope.union(make_convex_polytope([
                [-1, -1, -1,  1, 1, 0, 0],  # -1 - ah - al + af + s+ ≥ 0
            ], name="af strength")).union(make_convex_polytope([
                [-1,  1, -1, -1, 1, 0, 0],  # -1 + ah - al - af + s+ ≥ 0
            ], name="ah strength"))
        )
    ).intersect(
        # frustrum bound
        empty_polytope.union(make_convex_polytope([
            [0, 0, -1,  0, 1, -1, -1]  # - al + s+ - s1 - s2 ≥ 0
        ], name="al frustrum")).union(make_convex_polytope([
            [0, 0,  0, -1, 1, -1, -1]  # - af + s+ - s1 - s2 ≥ 0
        ], name="af frustrum"))
    )
)
"""
This polytope records the constraints on the "a coordinates", i.e., the "source"
of the edge in the canonical path.  They're taken to be _partially unsorted_:
the first coordinate is larger than the second, but the third has no specified
order relation to the first two.  The two sorted coordinates are referred to as
"ah" for "a high" and "al" for "a low", and the remaining coordinate is referred
to as "af" for "a fixed" — see the documentation for `interference_polytope` for
an explanation of the "fixed" terminology.  Otherwise, they obey the necessary
linear relations so that if they _were_ sorted, then they _would_ give a
normalized positive canonical coordinate which belongs the polytope produced by
the main global theorem (again, see `polytope_from_strengths`) for the strength
sequence [s1, s2, ..., sn].
"""


interference_polytope = Polytope(convex_subpolytopes=[
    # af = b1
    ConvexPolytope([
        # k ah al af b1 b2 b3 beta
        [0, -1, -1, 0,  0,  1,  1,  1],  # - ah - al + b2 + b3 + beta ≥ 0
        [2, -1, -1, 0,  0, -1, -1, -1],  # 2 - ah - al - b2 - b3 - beta ≥ 0
        [0,  1,  1, 0,  0, -1, -1,  1],  # ah + al - b2 - b3 + beta ≥ 0
        [0, -1,  1, 0,  0,  1, -1,  1],  # - ah + al + b2 - b3 + beta ≥ 0
        [0,  1, -1, 0,  0, -1,  1,  1],  # ah - al - b2 + b3 + beta ≥ 0
        [0,  1, -1, 0,  0,  1, -1, -1],  # ah - al + b2 - b3 - beta ≥ 0
    ], equalities=[
        [0,  0,  0, 1, -1,  0,  0,  0],  # af = b1
    ], name="AF=B1"),
    # OR af = b2
    ConvexPolytope([
        # k ah al af b1 b2 b3 beta
        [0, -1, -1, 0,  1,  0,  1,  1],  # - ah - al + b1 + b3 + beta ≥ 0
        [2, -1, -1, 0, -1,  0, -1, -1],  # 2 - ah - al - b1 - b3 - beta ≥ 0
        [0,  1,  1, 0, -1,  0, -1,  1],  # ah + al - b1 - b3 + beta ≥ 0
        [0, -1,  1, 0,  1,  0, -1,  1],  # - ah + al + b1 - b3 + beta ≥ 0
        [0,  1, -1, 0, -1,  0,  1,  1],  # ah - al - b1 + b3 + beta ≥ 0
        [0,  1, -1, 0,  1,  0, -1, -1],  # ah - al + b1 - b3 - beta ≥ 0
    ], equalities=[
        [0,  0,  0, 1,  0, -1,  0,  0],  # af = b2
    ], name="AF=B2"),
    # OR af = b3
    ConvexPolytope([
        # k ah al af b1 b2 b3 beta
        [0, -1, -1, 0,  1,  1,  0,  1],  # - ah - al + b1 + b2 + beta ≥ 0
        [2, -1, -1, 0, -1, -1,  0, -1],  # 2 - ah - al - b1 - b2 - beta ≥ 0
        [0,  1,  1, 0, -1, -1,  0,  1],  # ah + al - b1 - b2 + beta ≥ 0
        [0, -1,  1, 0,  1, -1,  0,  1],  # - ah + al + b1 - b2 + beta ≥ 0
        [0,  1, -1, 0, -1,  1,  0,  1],  # ah - al - b1 + b2 + beta ≥ 0
        [0,  1, -1, 0,  1, -1,  0, -1],  # ah - al + b1 - b2 - beta ≥ 0
    ], equalities=[
        [0,  0,  0, 1,  0,  0, -1,  0],  # af = b3
    ], name="AF=B3"),
])
"""
This polytope interrelates the a-coordinates and the b-coordinates using the
"interference inequalities": it fixes one of the three b-coordinates to agree
with the a-coordinate named "af"; and with the remaining coordinates, it applies
the linear constraints needed to guarantee a solution for Qiskit's
`xx_decompose.circuits.decompose_xxyy_into_xxyy_xx`.
"""


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
            name='·'.join(f"RZX({x / np.pi:.5f}π)" for x in strengths[:-2]) + ' unreflected'),
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
            name='·'.join(f"RZX({x / np.pi:.5f}π)" for x in strengths[:-2]) + ' reflected')
    ])

    return polytope


def get_augmented_coordinate(target_coordinate, strengths):
    """
    Produce a coordinate suitable for use with `xx_region_polytope`.
    """
    *strengths, beta = strengths
    strengths = sorted(strengths + [0, 0])
    interaction_coordinate = [sum(strengths), strengths[-1], strengths[-2], beta]
    return [*target_coordinate, *interaction_coordinate]


def regenerate_xx_solution_polytopes():
    """
    Recalculates the partition of b-coordinates into regions with solvable lifts
    with the interference relations.

    Recreates the pair (xx_region_polytope, xx_lift_polytope).

    NOTE: This routine amounts to a computer-calculated _proof_ of the main
          local theorem, which it checks as an assertion.
    """

    raw_polytope = (
        # k s+ s1 s2 beta
        cylinderize(strength_polytope, [0, 7, 8, 9, 10], 11)
        # k ah al af s+ s1 s2
        .intersect(cylinderize(a_polytope, [0, 1, 2, 3, 7, 8, 9], 11))
        # k b1 b2 b3 s+ s1 s2 beta
        .intersect(cylinderize(b_polytope, [0, 4, 5, 6, 7, 8, 9, 10], 11))
        # k ah al af b1 b2 b3 beta
        .intersect(cylinderize(interference_polytope, [0, 1, 2, 3, 4, 5, 6, 10], 11))
    )

    constrained_polytope = raw_polytope.reduce()
    # project away the a polytope: af, then al, then ah.
    constrained_polytope = project(constrained_polytope, 3).reduce()
    constrained_polytope = project(constrained_polytope, 2).reduce()
    constrained_polytope = project(constrained_polytope, 1).reduce()

    # compare with the original b polytope
    big_polytope = b_polytope.intersect(
        cylinderize(strength_polytope, [0, 4, 5, 6, 7], 8)
    ).reduce()
    assert constrained_polytope.contains(big_polytope)

    constrained_polytope_names = \
        [cp.name for cp in constrained_polytope.convex_subpolytopes]
    raw_polytope = Polytope(convex_subpolytopes=[
        cp for cp in big_polytope.convex_subpolytopes
        if cp.name in constrained_polytope_names
    ])

    return constrained_polytope, raw_polytope


def check_main_xx_theorem():
    """
    Checks the inductive step in the presentation of XX circuit polytopes.
    """
    global a_polytope, b_polytope, strength_polytope

    biswas_relations = (qlr_polytope
        # enlarge to the pu_4 version of the QLR relations
        .union(rho_reflect(qlr_polytope, [0, 7, 8, 9]))
        # constrain in- and out-coordinates to the appropriate alcove
        .intersect(cylinderize(monodromy_alcove, [0, 1, 2, 3], 10))
        .intersect(cylinderize(monodromy_alcove_c2, [0, 7, 8, 9], 10))
    )

    # constrain interaction coordinates to be of XX-type
    biswas_relations = biswas_relations.intersect(Polytope(convex_subpolytopes=[
        ConvexPolytope(
            inequalities=[[1, 0, 0, 0, -4, 0, 0, 0, 0, 0]],
            equalities=[
                [0, 0, 0, 0, 1, -1, 0, 0, 0, 0],  # x1 == x2
                [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],  # x2 == -x3
            ]
        )
    ]))

    # switch to canonical coordinates
    biswas_relations = monodromy_to_positive_canonical_polytope(
        biswas_relations, coordinates=[0, 1, 2, 3])
    biswas_relations = monodromy_to_positive_canonical_polytope(
        biswas_relations, coordinates=[0, 4, 5, 6])
    biswas_relations = monodromy_to_positive_canonical_polytope(
        biswas_relations, coordinates=[0, 7, 8, 9])

    # reduce the biswas relations to have following coordinates:
    # k a1 a2 a3 beta b1 b2 b3
    biswas_relations = biswas_relations.reduce()
    biswas_relations = project(biswas_relations, 6).reduce()
    biswas_relations = project(biswas_relations, 5).reduce()

    # build the monodromy polytope relating the in-coordinates, constrained by
    # the inductive hypothesis, to the output coordinates through the biswas
    # relations + the XX interaction constraint. we'll get these coordinates:
    # k a1 a2 a3 b1 b2 b3 s+ s1 s2 beta
    induction_polytope = (empty_polytope
        # k a1 a2 a3 beta b1 b2 b3
        .union(cylinderize(biswas_relations, [0, 1, 2, 3, 10, 4, 5, 6], 11))
        # k a1 a2 a3 s+ s1 s2
        .intersect(cylinderize(a_polytope, [0, 1, 2, 3, 7, 8, 9], 11))
        # k s+ s1 s2 beta
        .intersect(cylinderize(strength_polytope, [0, 7, 8, 9, 10], 11)))

    # project away the a coordinates to get the next inductive step
    induction_polytope = project(induction_polytope, 3).reduce()
    induction_polytope = project(induction_polytope, 2).reduce()
    induction_polytope = project(induction_polytope, 1).reduce()

    # now check that this calculation actually closes the induction
    b_and_strength = b_polytope.intersect(
        cylinderize(strength_polytope, [0, 4, 5, 6, 7], 8)
    )

    assert b_and_strength.contains(induction_polytope) and \
        induction_polytope.contains(b_and_strength)
