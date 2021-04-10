"""
monodromy/examples.py

A variety of "standard" polytopes and gates.
"""

from fractions import Fraction

import numpy as np

from .polytopes import ConvexPolytope, make_convex_polytope, Polytope
from .qlr_table import fractionify, alcove_c2


def exactly(*coordinates) -> Polytope:
    """
    Produce a family of inequalities that forces equality with `coordinates`.
    """
    table = []
    for index, coordinate in enumerate(coordinates):
        row = [Fraction(0)] * (1 + len(coordinates))
        row[0] = Fraction(coordinate)
        row[1 + index] = Fraction(-1)
        table.append(row)
        table.append([-x for x in row])
    return make_convex_polytope(table)


# corners of the A_{C_2} polytope, in the order that they appear in /1904.10541
identity_polytope = exactly(        0,     0,      0)
CX_polytope       = exactly(    1 / 4, 1 / 4, -1 / 4)
ISWAP_polytope    = exactly(    1 / 2,     0,      0)
SWAP_polytope     = exactly(    1 / 4, 1 / 4,  1 / 4)
sqrtSWAP_polytope = exactly(Fraction(3, 8), Fraction( 3, 8), Fraction(-1, 8))
e6_polytope       = exactly(Fraction(3, 8), Fraction(-1, 8), Fraction(-1, 8))
# this last one is rho(Pi(sqrt SWAP)).


# some parametric gates of interest
CPHASE_polytope = make_convex_polytope(fractionify([
    [0,  1, -1,  0,],  # x1 == x2
    [0, -1,  1,  0,],
    [0,  0,  1,  1,],   # x2 == -x3
    [0,  0, -1, -1,],
    *alcove_c2.convex_subpolytopes[0].inequalities,
]))
XY_polytope = make_convex_polytope(fractionify([
    [0, 0,  1,  0],  # x2 == 0
    [0, 0, -1,  0],
    [0, 0,  0,  1],  # x3 == 0
    [0, 0,  0, -1],
    *alcove_c2.convex_subpolytopes[0].inequalities,
]))


# some other gates of interest
sqrtCX_polytope   = exactly(1 / 8, 1 / 8, -1 / 8)
thirdCX_polytope  = exactly(Fraction(1, 12), Fraction(1, 12), Fraction(-1, 12))


# gate-unrelated standard polytopes
everything_polytope = Polytope(convex_subpolytopes=[
    ConvexPolytope(inequalities=[])
])
empty_polytope = Polytope(convex_subpolytopes=[])


# some gates of interest

def canonical_matrix(a=0, b=0, c=0):
    """The canonical operator exp(-i(a XX + b YY + c ZZ))."""
    cplus, cminus = np.cos(a + b), np.cos(a - b)
    splus, sminus = np.sin(a + b), np.sin(a - b)
    eic = np.exp(1j * c)

    return np.array([
        [     cminus * eic,                  0,                 0, -1j * sminus * eic, ],
        [                 0,       cplus / eic, -1j * splus / eic,                  0, ],
        [                 0, -1j * splus / eic,       cplus / eic,                  0, ],
        [-1j * sminus * eic,                 0,                 0,       cminus * eic, ],
    ])


def rx_matrix(theta=0):
    """The rotation operator exp(-i theta X)."""
    return np.array([
        [      np.cos(theta), -1j * np.sin(theta), ],
        [-1j * np.sin(theta),       np.cos(theta), ],
    ])


def ry_matrix(theta=0):
    """The rotation operator exp(-i theta Y)."""
    return np.array([
        [np.cos(theta), -np.sin(theta), ],
        [np.sin(theta),  np.cos(theta), ],
    ])


def rz_matrix(theta=0):
    """The rotation operator exp(-i theta Z)."""
    return np.array([
        [np.exp(1j * -theta),                  0, ],
        [                  0, np.exp(1j * theta), ],
    ])
