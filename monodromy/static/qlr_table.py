"""
monodromy/static/qlr_table.py

Precomputed descriptions of the monodromy polytope for SU(4) (and PU(4)).
"""

from ..elimination import cylinderize, project
from ..polytopes import ConvexPolytope, make_convex_polytope, Polytope
from ..static.examples import empty_polytope


#              r  k   a    b    c   d
qlr_table = [[1, 3, [0], [0], [0], 0],
             [1, 3, [0], [1], [1], 0],
             [1, 3, [0], [2], [2], 0],
             [1, 3, [0], [3], [3], 0],
             [1, 3, [1], [1], [2], 0],
             [1, 3, [1], [2], [3], 0],
             [1, 3, [1], [3], [0], 1],
             [1, 3, [2], [2], [0], 1],
             [1, 3, [2], [3], [1], 1],
             [1, 3, [3], [3], [2], 1],
             # r k    a       b       c     d
             [2, 2, [0, 0], [0, 0], [0, 0], 0],
             [2, 2, [0, 0], [1, 0], [1, 0], 0],
             [2, 2, [0, 0], [1, 1], [1, 1], 0],
             [2, 2, [0, 0], [2, 0], [2, 0], 0],
             [2, 2, [0, 0], [2, 1], [2, 1], 0],
             [2, 2, [0, 0], [2, 2], [2, 2], 0],
             [2, 2, [1, 0], [1, 0], [1, 1], 0],
             [2, 2, [1, 0], [1, 0], [2, 0], 0],
             [2, 2, [1, 0], [1, 1], [2, 1], 0],
             [2, 2, [1, 0], [2, 0], [2, 1], 0],
             [2, 2, [1, 0], [2, 1], [2, 2], 0],
             [2, 2, [1, 0], [2, 1], [0, 0], 1],
             [2, 2, [1, 0], [2, 2], [1, 0], 1],
             [2, 2, [1, 1], [1, 1], [2, 2], 0],
             [2, 2, [1, 1], [2, 0], [0, 0], 1],
             [2, 2, [1, 1], [2, 1], [1, 0], 1],
             [2, 2, [1, 1], [2, 2], [2, 0], 1],
             [2, 2, [2, 0], [2, 0], [2, 2], 0],
             [2, 2, [2, 0], [2, 1], [1, 0], 1],
             [2, 2, [2, 0], [2, 2], [1, 1], 1],
             [2, 2, [2, 1], [2, 1], [2, 0], 1],
             [2, 2, [2, 1], [2, 1], [1, 1], 1],
             [2, 2, [2, 1], [2, 2], [2, 1], 1],
             [2, 2, [2, 2], [2, 2], [0, 0], 2],
             # r k      a          b          c      d
             [3, 1, [0, 0, 0], [0, 0, 0], [0, 0, 0], 0],
             [3, 1, [0, 0, 0], [1, 0, 0], [1, 0, 0], 0],
             [3, 1, [0, 0, 0], [1, 1, 0], [1, 1, 0], 0],
             [3, 1, [0, 0, 0], [1, 1, 1], [1, 1, 1], 0],
             [3, 1, [1, 0, 0], [1, 0, 0], [1, 1, 0], 0],
             [3, 1, [1, 0, 0], [1, 1, 0], [1, 1, 1], 0],
             [3, 1, [1, 0, 0], [1, 1, 1], [0, 0, 0], 1],
             [3, 1, [1, 1, 0], [1, 1, 0], [0, 0, 0], 1],
             [3, 1, [1, 1, 0], [1, 1, 1], [1, 0, 0], 1],
             [3, 1, [1, 1, 1], [1, 1, 1], [1, 1, 0], 1]]
"""
Precomputed table of quantum Littlewood-Richardson coefficients for the small
quantum cohomology ring of k-planes in C^4, 0 < k < 4.  Each entry is of the
form [r, k, [*a], [*b], [*c], d], corresponding to the relation

    N_{ab}^{c, d} = 1   or   <sigma_a, sigma_b, sigma_{*c}> = q^d.

NOTE: We include only entries with a <= b in the traversal ordering used by
      `monodromy.io.lrcalc.displacements`.

NOTE: This table can be regenerated using
      `monodromy.io.lrcalc.regenerate_qlr_table` .
"""


def ineq_from_qlr(r, k, a, b, c, d):
    """
    Generates a monodromy polytope inequality from the position of a nonzero
    quantum Littlewood-Richardson coefficient for su_4.

    See (*) in Theorem 23 of /1904.10541 .

    NOTE: `r` is ignored, since `4 = r + k` makes it redundant with `k`.
    """

    # $$d - \sum_{i=1}^r \alpha_{k+i-a_i}
    #     - \sum_{i=1}^r \beta_{k+i-b_i}
    #     + \sum_{i=1}^r \delta_{k+i-c_i} \ge 0$$

    new_row = [d,
               0, 0, 0, 0,  # alpha's
               0, 0, 0, 0,  # beta's
               0, 0, 0, 0, ]  # gamma's
    for i, ai in enumerate(a):
        index = k + (i + 1) - ai  # subscript in the Biswas inequality
        offset = 0  # last entry before alpha
        new_row[index + offset] -= 1  # poke the value in
    for i, bi in enumerate(b):
        index = k + (i + 1) - bi  # subscript in the Biswas inequality
        offset = 4  # last entry before beta
        new_row[index + offset] -= 1  # poke the value in
    for i, ci in enumerate(c):
        index = k + (i + 1) - ci  # subscript in the Biswas inequality
        offset = 8  # last entry before gamma
        new_row[index + offset] += 1  # poke the value in

    # now remember that a4 = -a1-a2-a3 and so on
    new_row = [new_row[0],
               *[x - new_row[4] for x in new_row[1:4]],
               *[x - new_row[8] for x in new_row[5:8]],
               *[x - new_row[12] for x in new_row[9:12]]
               ]

    return new_row


def generate_qlr_inequalities():
    """
    Regenerates the set of monodromy polytope inequalities from the stored table
    `qlr_table` of quantum Littlewood-Richardson coefficients.
    """
    qlr_inequalities = []
    for r, k, a, b, c, d in qlr_table:
        qlr_inequalities.append(ineq_from_qlr(r, k, a, b, c, d))
        if a != b:
            qlr_inequalities.append(ineq_from_qlr(r, k, b, a, c, d))

    return qlr_inequalities


qlr_polytope = make_convex_polytope(
    generate_qlr_inequalities(),
    name="QLR relations"
)
"""
This houses the monodromy polytope, the main static input of the whole calc'n.
This polytope does _not_ also contain the alcove constraints.
"""


#
# In the second half of this file, we set up the ingredients (themselves
# extracted from the above table of QLR relations) used to recompute the
# monodromy relations for XX circuits.
#
# See monodromy/xx_decompose/paths.py for more information.
#


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
the linear constraints needed to guarantee a solution for
`monodromy.xx_decompose.circuits.decompose_xxyy_into_xxyy_xx`.
"""


def regenerate_xx_solution_polytopes():
    """
    Recalculates the partition of b-coordinates into regions with solvable lifts
    with the interference relations.

    Returns the pair (xx_region_polytope, xx_lift_polytope).

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
