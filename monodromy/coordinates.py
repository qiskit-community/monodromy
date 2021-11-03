"""
monodromy/coordinates.py

This file contains routines for converting from one system to another. There are
several common coordinate systems / choices of Weyl alcove in pu_4:

+ (Balanced) canonical parameters: These are the (a, b, c) in the exponential

      exp(-i(a XX + b YY + c ZZ))

  with a, b, c further chosen to satisfy pi/4 ≥ a ≥ b ≥ |c|.  One may further
  _normalize_ these coordinates by dividing by pi, so that the polytope becomes
  integrally specified.

+ (Positive) canonical parameters: These are the (a, b, c) in the exponential

      exp(-i(a XX + b YY + c ZZ))

  with a, b, c further chosen to satisfy a ≥ b ≥ c ≥ 0 and pi - a ≥ b.  One may
  further _normalize_ these coordinates by dividing by pi, so that the polytope
  becomes integrally specified.

+ Monodromy coordinates: These are the a, b, c in the formula

      exp(diag(2 pi i a, 2 pi i b, 2 pi i c, -2 pi i(a + b + c)),

  with a, b, c further chosen to satisfy a ≥ b ≥ c ≥ -(a + b + c) ≥ a - 1 and
  b + c ≥ 0.  These coordinates are always taken to be "normalized".

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
import math
from typing import List, Optional

import numpy as np

from .polytopes import ConvexPolytope, make_convex_polytope, Polytope
from .utilities import clear_memoization, epsilon


positive_canonical_alcove = make_convex_polytope([
    [1, -1,  0,  0],  # 1  >= c1
    [0,  1, -1,  0],  # c1 >= c2
    [0,  0,  1, -1],  # c2 >= c3
    [1, -1, -1,  0],  # 1 - c1 >= c2
    [0,  0,  1,  1],  # c3 >= -c2
], name="A_SU(4)^can")
"""
Inequalities defining the standard choice of fundamental Weyl alcove in
normalized positive canonical coordinates for SU(4).

cf. Eqn 6 of Entropy.
"""


positive_canonical_alcove_c2 = make_convex_polytope([
    *positive_canonical_alcove.convex_subpolytopes[0].inequalities,
    [0, 0, 0, 1],  # c3 >=  0, the C2 inequality
], name="A_PU(4)^can")
"""
Inequalities defining the standard choice of fundamental Weyl alcove in
normalized positive canonical coordinates for PU(4).

cf. Eqn 6 of Entropy.
"""


monodromy_alcove = make_convex_polytope([
    [0,  1, -1,  0, ],  # a1 - a2 >= 0
    [0,  0,  1, -1, ],  # a2 - a3 >= 0
    [0,  1,  1,  2, ],  # a3 - a4 >= 0
    [1, -2, -1, -1, ],  # a4 - (a1 - 1) >= 0
], name="A_SU(4)^monodromy")
"""
Inequalities defining the fundamental Weyl alcove used in monodromy polytope
calculations for SU(4).
"""


monodromy_alcove_c2 = make_convex_polytope([
    [1, -2, -1, -1],
    [0,  0,  1,  1],
    [0,  1, -1,  0],
    [0,  0,  1, -1]
], name="A_PU(4)^monodromy")
# monodromy_alcove_c2 = make_convex_polytope([
#     *monodromy_alcove.convex_subpolytopes[0].inequalities,
#     [0, 0, 1, 1, ],  # a2 + a3 >= 0 , the C2 inequality
# ]).reduce()
"""
Inequalities defining the fundamental Weyl alcove used in monodromy polytope
calculations for PU(4).
"""


monodromy_alcove_c2_pcs = make_convex_polytope([
    *monodromy_alcove.convex_subpolytopes[0].inequalities,
    [1, -2, 0, 2, ],  # a3 + 1/2 >= a1 , the C2 inequality
], name="A_PU(4)^PCS")
"""
Inequalities defining the fundamental Weyl alcove used in monodromy polytope
calculations for PU(4) according to Peterson-Crooks-Smith.

IMPORTANT NOTE: This is only scissors-congruent to alcove_c2, which is itself
    linearly equivalent to positive_canonical_convex_polytope.  We advise _not_
    using these coordinates.
"""


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


def unitary_to_monodromy_coordinate(unitary):
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


def monodromy_to_positive_canonical_coordinate(x, y, z):
    """
    Given a monodromy alcove coordinate, produces its image as an unnormalized
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
    Converts an unnormalized positive canonical coordinate to a monodromy
    coordinate.
    """
    normalizing_factor = np.pi

    return ((x + y - z) / normalizing_factor,
            (x - y + z) / normalizing_factor,
            (-x + y + z) / normalizing_factor)


def positive_to_balanced_canonical_coordinate(x, y, z):
    """
    Converts a balanced canonical coordinate to a positive one.
    """
    if x <= np.pi / 4:
        return x, y, z
    else:
        return tuple(sorted([np.pi / 2 - x, y, -z], reverse=True))


def monodromy_to_positive_canonical_polytope(
        monodromy_polytope: Polytope,
        coordinates: Optional[List[int]] = None
) -> Polytope:
    """
    Converts a Polytope in monodromy coordinates to a Polytope in positive
    canonical coordinates, normalized so that CX lies at (1/2, 0, 0).
    """
    coordinates = [0, 1, 2, 3] if coordinates is None else coordinates

    canonical_convex_subpolytopes = []
    for convex_subpolytope in monodromy_polytope.convex_subpolytopes:
        inequalities, equalities = [], []
        for inequality in convex_subpolytope.inequalities:
            new_inequality = copy(inequality)
            k, x, y, z = (inequality[c] for c in coordinates)
            for c, v in zip(coordinates, [2 * k, x + y - z, x - y + z, -x + y + z]):
                new_inequality[c] = v
            for c in range(len(new_inequality)):
                if c not in coordinates:
                    new_inequality[c] *= 2
            inequalities.append(new_inequality)

        for equality in convex_subpolytope.equalities:
            new_equality = copy(equality)
            k, x, y, z = (equality[c] for c in coordinates)
            for c, v in zip(coordinates, [2 * k, x + y - z, x - y + z, -x + y + z]):
                new_equality[c] = v
            for c in range(len(new_equality)):
                if c not in coordinates:
                    new_equality[c] *= 2
            equalities.append(new_equality)

        canonical_convex_subpolytopes.append(ConvexPolytope(
            inequalities=inequalities,
            equalities=equalities,
        ))

    canonical_polytope = copy(monodromy_polytope)
    clear_memoization(canonical_polytope)

    canonical_polytope.convex_subpolytopes = canonical_convex_subpolytopes
    return canonical_polytope


def average_infidelity(p, q):
    """
    Computes the infidelity distance between two points p, q expressed in
    positive canonical coordinates.
    """

    a, b, c = p
    d, e, f = q

    return 1 - 1 / 20 * (4 + 16 * (
        math.cos(a - d) ** 2 * math.cos(b - e) ** 2 * math.cos(c - f) ** 2 +
        math.sin(a - d) ** 2 * math.sin(b - e) ** 2 * math.sin(c - f) ** 2
    ))


def rho_reflect(polytope, coordinates=None):
    """
    Applies rho-reflection to the indicated `coordinates` of `polytope`.
    If `coordinates` is not supplied, uses the final three coordinates.
    """

    if coordinates is None:
        coordinates = [0, -3, -2, -1]

    # an inequality
    #     d + x a1 + y a2 + z a3 >= 0
    # induces on rho-application
    #     d + x (a3 + 1/2) + y (a4 + 1/2) + z (a1 - 1/2) >= 0, or
    #    (d + 1/2 x + 1/2 y - 1/2 z) + (z - y) a1 + (-y) a2 + (x - y) a3 >= 0.
    rho_subpolytopes = []
    for convex_subpolytope in polytope.convex_subpolytopes:
        rotated_equalities, rotated_inequalities = [], []
        for inequality in convex_subpolytope.inequalities:
            d = inequality[coordinates[0]]
            x = inequality[coordinates[1]]
            y = inequality[coordinates[2]]
            z = inequality[coordinates[3]]

            new_inequality = [2 * x for x in inequality]
            new_inequality[coordinates[0]] = 2 * d + x + y - z
            new_inequality[coordinates[1]] = 2 * z - 2 * y
            new_inequality[coordinates[2]] = 2 * 0 - 2 * y
            new_inequality[coordinates[3]] = 2 * x - 2 * y

            rotated_inequalities.append(new_inequality)

        for equality in convex_subpolytope.equalities:
            d = equality[coordinates[0]]
            x = equality[coordinates[1]]
            y = equality[coordinates[2]]
            z = equality[coordinates[3]]

            new_equality = [2 * x for x in equality]
            new_equality[coordinates[0]] = 2 * d + x + y - z
            new_equality[coordinates[1]] = 2 * z - 2 * y
            new_equality[coordinates[2]] = 2 * 0 - 2 * y
            new_equality[coordinates[3]] = 2 * x - 2 * y

            rotated_equalities.append(new_equality)

        rho_subpolytopes.append(ConvexPolytope(
            inequalities=rotated_inequalities,
            equalities=rotated_equalities,
        ))

    return Polytope(convex_subpolytopes=rho_subpolytopes)


def mirror_positive_canonical_coordinate(coordinate):
    """
    Produces the SWAP-mirror of an unnormalized positive canonical coordinate.
    """
    alpha, beta, gamma = coordinate
    if alpha >= np.pi / 4 + epsilon:
        return [np.pi / 4 - gamma, np.pi / 4 - beta, np.pi / 4 + alpha]
    else:
        return [np.pi / 4 + gamma, np.pi / 4 - beta, np.pi / 4 - alpha]


def mirror_monodromy_polytope(polytope):
    """
    Calculates the mirror of a polytope expressed in monodromy coordinates.
    """
    low_polytope = copy(polytope)
    low_polytope.convex_subpolytopes = []
    for cp in polytope.convex_subpolytopes:
        # for a < 1/2, get (1/2 + c, 1/2 - b, 1/2 - a)
        mirror_cp = ConvexPolytope(
            inequalities=[[1, -2, 0, 0]],
            equalities=[],
            name=cp.name
        )
        # k + c1 a + c2 b + c3 c ≥ 0 becomes
        # (2 k + c1 + c2 + c3) + -2 c3 a + -2 c2 b + 2 c1 c ≥ 0
        for ineq in cp.inequalities:
            k, c1, c2, c3, = ineq
            mirror_cp.inequalities.append([
                2*k + c1 + c2 + c3, -2 * c3, -2 * c2, 2 * c1
            ])
        for eq in cp.equalities:
            k, c1, c2, c3, = eq
            mirror_cp.equalities.append([
                2 * k + c1 + c2 + c3, -2 * c3, -2 * c2, 2 * c1
            ])
        low_polytope.convex_subpolytopes.append(mirror_cp)

    high_polytope = copy(polytope)
    high_polytope.convex_subpolytopes = []
    for cp in polytope.convex_subpolytopes:
        # for a > 1/2, get (1/2 - c, 1/2 - b, a - 1/2).
        mirror_cp = ConvexPolytope(
            inequalities=[[1, -2, 0, 0]],
            equalities=[],
            name=cp.name
        )
        # k + c1 a + c2 b + c3 c ≥ 0 becomes
        # (2 k + c1 + c2 - c3) + 2 c3 a - 2 c2 b - 2 c1 c ≥ 0
        for ineq in cp.inequalities:
            k, c1, c2, c3, = ineq
            mirror_cp.inequalities.append([
                2 * k + c1 + c2 - c3, 2 * c3, -2 * c2, -2 * c1
            ])
        for eq in cp.equalities:
            k, c1, c2, c3, = eq
            mirror_cp.equalities.append([
                2 * k + c1 + c2 - c3, 2 * c3, -2 * c2, -2 * c1
            ])
        high_polytope.convex_subpolytopes.append(mirror_cp)

    return (
        low_polytope
            .union(high_polytope)
            .intersect(positive_canonical_alcove_c2)
    )


def monodromy_to_monodromy_pcs_polytope(polytope):
    """
    Converts a `polytope`, expressed in monodromy coordinates with the standard
    rho-normalization condition, to a polytope in monodromy coordinates with the
    Peterson-Crooks-Smith normalization condition instead.
    """

    return polytope.intersect(monodromy_alcove_c2_pcs).union(
        rho_reflect(polytope).intersect(monodromy_alcove_c2_pcs)
    )
