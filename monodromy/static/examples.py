"""
monodromy/static/examples.py

A variety of "standard" polytopes and gates.
"""

from ..polytopes import ConvexPolytope, make_convex_polytope, Polytope


def exactly(*coordinates, name=None) -> Polytope:
    """
    Produce a family of inequalities that forces equality with `coordinates`.
    """
    table = []
    for index, coordinate in enumerate(coordinates):
        row = [0] * (1 + len(coordinates))
        row[0] = coordinate.numerator
        row[1 + index] = -coordinate.denominator
        table.append(row)
    return make_convex_polytope([], equalities=table, name=name)


everything_polytope = Polytope(convex_subpolytopes=[
    ConvexPolytope(inequalities=[], name="True")
])
"""
The basic boolean "True" polytope: all points belong.

NOTE: This polytope is dimensionless.
"""


empty_polytope = Polytope(convex_subpolytopes=[])
"""
The basic boolean "False" polytope: no points belong.

NOTE: This polytope is dimensionless.
"""


identity_polytope = Polytope(convex_subpolytopes=[
    ConvexPolytope(inequalities=[], name="origin", equalities=[
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])
])
"""
A polytope containing only the canonical coordinate of the identity gate, i.e.,
the origin in 3-space.
"""


# # some parametric gates of interest
# CPHASE_polytope = make_convex_polytope([
#     [0,  1, -1,  0,],  # x1 == x2
#     [0, -1,  1,  0,],
#     [0,  0,  1,  1,],   # x2 == -x3
#     [0,  0, -1, -1,],
#     *monodromy_alcove_c2.convex_subpolytopes[0].inequalities,
# ])
# XY_polytope = make_convex_polytope([
#     [0, 0,  1,  0],  # x2 == 0
#     [0, 0, -1,  0],
#     [0, 0,  0,  1],  # x3 == 0
#     [0, 0,  0, -1],
#     *monodromy_alcove_c2.convex_subpolytopes[0].inequalities,
# ])
#
#
# # some other gates of interest
# sqrtCX_polytope   = exactly(Fraction(1,  8), Fraction(1,  8), Fraction(-1,  8))
# thirdCX_polytope  = exactly(Fraction(1, 12), Fraction(1, 12), Fraction(-1, 12))
