"""
monodromy/coordinates.py

This file contains routines for converting from one system to another. There are
several common coordinate systems / choices of Weyl alcove in pu_4:

+ (Balanced) canonical parameters: These are the (a, b, c) in the exponential

      exp(-i(a XX + b YY + c ZZ))

  with a, b, c further chosen to satisfy pi/4 ≥ a ≥ b ≥ |c|.

+ (Positive) canonical parameters: These are the (a, b, c) in the exponential

      exp(-i(a XX + b YY + c ZZ))

  with a, b, c further chosen to satisfy a ≥ b ≥ c ≥ 0 and pi - a ≥ b.

+ Alcove coordinates: These are the a, b, c in the formula

      exp(diag(2 pi i a, 2 pi i b, 2 pi i c, -2 pi i(a + b + c)),

  with a, b, c further chosen to satisfy a ≥ b ≥ c ≥ -(a + b + c) ≥ a - 1 and
  b + c ≥ 0.

  NOTE: In the Monodromy Polytope paper, we used the condition c + 1/2 ≥ a
        instead.  This is mathematically legal, but the resulting alcove (a
        triangular prism) is related to the canonical alcove (a tetrahedron) by
        a "scissors congruence" rather than by a linear transformation.  For
        this reason, we prefer this alternative normalization condition.

NOTE: Working in pu_4 rather than su_4 adds one extra inequality, corresponding
      to a preference for one of the half-spaces picked out by reflection
      through the `rho` operator.
"""

from copy import copy
from functools import reduce
from typing import List, Optional

import numpy as np

from .polytopes import ConvexPolytope, make_convex_polytope, Polytope
from .utilities import epsilon


"""
Inequalities defining the standard choice of fundamental Weyl alcove in
normalized positive canonical coordinates for SU(4).

cf. Eqn 6 of Entropy.
"""
positive_canonical_convex_polytope_su4 = make_convex_polytope([
    [1, -1,  0,  0],  # 1  >= c1
    [0,  1, -1,  0],  # c1 >= c2
    [0,  0,  1, -1],  # c2 >= c3
    [1, -1, -1,  0],  # 1 - c1 >= c2
])


"""
Inequalities defining the standard choice of fundamental Weyl alcove in
normalized positive canonical coordinates for PU(4).

cf. Eqn 6 of Entropy.
"""
positive_canonical_convex_polytope = make_convex_polytope([
    *positive_canonical_convex_polytope_su4.convex_subpolytopes[0].inequalities,
    [0, 0, 0, 1],  # c3 >=  0, the C2 inequality
])


"""
Inequalities defining the fundamental Weyl alcove used in monodromy polytope
calculations for SU(4).
"""
alcove = make_convex_polytope([
    [0,  1, -1,  0, ],  # a1 - a2 >= 0
    [0,  0,  1, -1, ],  # a2 - a3 >= 0
    [0,  1,  1,  2, ],  # a3 - a4 >= 0
    [1, -2, -1, -1, ],  # a4 - (a1 - 1) >= 0
])


"""
Inequalities defining the fundamental Weyl alcove used in monodromy polytope
calculations for PU(4).
"""
alcove_c2 = make_convex_polytope([
    [1, -2, -1, -1],
    [0,  0,  1,  1],
    [0,  1, -1,  0],
    [0,  0,  1, -1]
])
# alcove_c2 = make_convex_polytope([
#     *alcove.convex_subpolytopes[0].inequalities,
#     [0, 0, 1, 1, ],  # a2 + a3 >= 0 , the C2 inequality
# ]).reduce()


def normalize_logspec_A(coordinate):
    """
    Utility function for rotating a LogSpec tuple (targeting A)
    into its normal form.

    Expects a sorted input in the range [0, 1].
    """
    total = sum(coordinate)
    if total > epsilon:
        return normalize_logspec_A([*coordinate[1:], coordinate[0] - 1])
    elif total < -epsilon:
        raise ValueError(
            f"Over-rotated: {total}. I don't think this should happen."
        )
    else:
        return coordinate


def normalize_logspec_AC2(coordinate):
    """
    Utility function for rotating a LogSpec tuple (targeting A_{C_2})
    into its normal form.

    Expects a sorted input in the range [0, 1].
    """
    partially_normalized_coordinate = normalize_logspec_A(coordinate)
    if partially_normalized_coordinate[1] >= \
            -partially_normalized_coordinate[2]:
        return partially_normalized_coordinate
    else:
        return [partially_normalized_coordinate[2] + 1 / 2,
                partially_normalized_coordinate[3] + 1 / 2,
                partially_normalized_coordinate[0] - 1 / 2,
                partially_normalized_coordinate[1] - 1 / 2]


def unitary_to_alcove_coordinate(unitary):
    """
    Given a unitary matrix, produces its alcove coordinate.
    """
    unitary = unitary * np.linalg.det(unitary) ** (-1 / 4)
    sysy = np.array([
        [0,  0,  0, 1],
        [0,  0, -1, 0],
        [0, -1,  0, 0],
        [1,  0,  0, 0],
    ], dtype=complex)
    gammaQ = reduce(np.dot, [unitary, sysy, unitary.T, sysy])
    logspec_coordinate = np.real(
        np.log(np.linalg.eigvals(gammaQ)) / (2 * np.pi * 1j))
    return normalize_logspec_AC2(
        sorted(np.mod(logspec_coordinate, 1.0), reverse=True)
    )


def alcove_to_positive_canonical_coordinate(x, y, z):
    """
    Given a monodromy alcove coordinate, produces its image as a _unnormalized_
    positive canonical coordinate.
    """
    normalizing_factor = np.pi
    return (
        (x + y) / 2 * normalizing_factor,
        (z + x) / 2 * normalizing_factor,
        (y + z) / 2 * normalizing_factor
    )


def positive_canonical_to_monodromy_coordinate(x, y, z):
    """
    Converts a normalized positive alcove coordinate to a monodromy alcove
    coordinate.
    """
    normalizing_factor = np.pi

    return ((x + y - z) / normalizing_factor,
            (x - y + z) / normalizing_factor,
            (-x + y + z) / normalizing_factor)


def monodromy_to_positive_canonical_polytope(
        monodromy_polytope: Polytope,
        coordinates: Optional[List[int]] = None
) -> Polytope:
    """
    Converts a Polytope in monodromy coordinates to the corresponding Polytope
    in normalized positive canonical coordinates.
    """
    coordinates = [0, 1, 2, 3] if coordinates is None else coordinates

    canonical_convex_subpolytopes = []
    for convex_subpolytope in monodromy_polytope.convex_subpolytopes:
        inequalities, equalities = [], []
        for inequality in convex_subpolytope.inequalities:
            new_inequality = copy(inequality)
            k, x, y, z = (inequality[c] for c in coordinates)
            for c, v in zip(coordinates, [k, x + y - z, x - y + z, -x + y + z]):
                new_inequality[c] = v
            inequalities.append(new_inequality)

        for equality in convex_subpolytope.equalities:
            new_equality = copy(equality)
            k, x, y, z = (equality[c] for c in coordinates)
            for c, v in zip(coordinates, [k, x + y - z, x - y + z, -x + y + z]):
                new_equality[c] = v
            equalities.append(new_equality)

        canonical_convex_subpolytopes.append(ConvexPolytope(
            inequalities=inequalities,
            equalities=equalities,
        ))

    canonical_polytope = copy(monodromy_polytope)
    canonical_polytope.convex_subpolytopes = canonical_convex_subpolytopes
    return canonical_polytope
