"""
monodromy/coordinates.py

There are several common coordinate systems / choices of Weyl alcove in su_4:

+ (Balanced) canonical parameters: These are the (a, b, c) in the exponential

      exp(-i(a XX + b YY + c ZZ))

  with a, b, c further chosen to satisfy pi/4 ≥ a ≥ b ≥ |c|.

+ (Positive) canonical parameters: These are the (a, b, c) in the exponential

      exp(-i(a XX + b YY + c ZZ))

  with a, b, c further chosen to satisfy a ≥ b ≥ c ≥ 0 and pi - a ≥ b.

+ Alcove coordinates: These are the alpha, beta, delta in the formula

      exp(diag(2 pi i a, 2 pi i b, 2 pi i c, -2 pi i(a + b + c)),

  with a, b, c further chosen to satisfy a ≥ b ≥ c ≥ -(a + b + c) ≥ a - 1 and
  c + 1/2 ≥ a.

(Working in pu_4 adds one extra inequality, enforced elsewhere in this repo
by reflection through the `rho` operator.)

This file contains routines for converting from one system to another.
"""

from copy import copy
from functools import reduce

import numpy as np

from .polytopes import ConvexPolytope, Polytope
from .utilities import fractionify


positive_canonical_convex_polytope = ConvexPolytope(
    inequalities=fractionify([
        [1, -1, 0, 0],  # cf. Eqn 6 of Entropy
        [0, 1, -1, 0],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, -1, -1, 0]
    ])
)


def normalize_logspec_A(coordinate):
    """
    Utility function for rotating a LogSpec tuple (targeting A)
    into its normal form.

    Expects a sorted input in the range [0, 1].
    """
    total = sum(coordinate)
    if total > 0.01:
        return normalize_logspec_A([*coordinate[1:], coordinate[0] - 1])
    elif total < -0.01:
        raise ValueError(
            f"Over-rotated: {total}. I don't think this should happen.")
    else:
        return coordinate


def normalize_logspec_AC2(coordinate):
    """
    Utility function for rotating a LogSpec tuple (targeting A_{C_2})
    into its normal form.

    Expects a sorted input in the range [0, 1].
    """
    partially_normalized_coordinate = normalize_logspec_A(coordinate)
    if partially_normalized_coordinate[0] < \
            partially_normalized_coordinate[2] + 1 / 2:
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
    Given an alcove coordinate, produces its image as a canonical coordinate.
    """
    normalizing_factor = np.pi
    retval = ((x + y) / 2 * normalizing_factor,
              (z + x) / 2 * normalizing_factor,
              (y + z) / 2 * normalizing_factor)

    # Just shearing the monodromy coordinate system to the canonical system
    # takes the monodromy alcove, a triangular prism, to another triangular
    # prism.  This last step enacts the scissors-congruence that turns this into
    # the expected tetrahedron.
    if retval[-1] < -0.01:
        retval = (
            np.pi / 2 - retval[0],
            retval[1],
            -1 * retval[2]
        )

    return retval


def monodromy_to_positive_canonical_polytope(monodromy_polytope):
    """
    Converts a Polytope in monodromy coordinates to the corresponding Polytope
    in (rescaled) (positive) canonical coordinates.
    """
    canonical_convex_subpolytopes = []
    for convex_subpolytope in monodromy_polytope.convex_subpolytopes:
        inequalities, equalities = [], []
        reflected_inequalities, reflected_equalities = [], []
        for inequality in convex_subpolytope.inequalities:
            k, x, y, z = inequality
            new_inequality = [k, x + y - z, x - y + z, -x + y + z]
            inequalities.append(new_inequality)
            reflected_inequalities.append(
                [new_inequality[0] + new_inequality[1] / 2,
                 -new_inequality[1], new_inequality[2], -new_inequality[3]]
            )
        for equality in convex_subpolytope.equalities:
            k, x, y, z = equality
            new_equality = [k, x + y - z, x - y + z, -x + y + z]
            equalities.append(new_equality)
            reflected_equalities.append(
                [new_equality[0] + new_equality[1] / 2,
                 -new_equality[1], new_equality[2], -new_equality[3]]
            )

        canonical_convex_subpolytopes.append(ConvexPolytope(
            inequalities=inequalities,
            equalities=equalities,
        ))
        canonical_convex_subpolytopes.append(ConvexPolytope(
            inequalities=reflected_inequalities,
            equalities=reflected_equalities,
        ))

    canonical_polytope = copy(monodromy_polytope)
    canonical_polytope.convex_subpolytopes = canonical_convex_subpolytopes
    canonical_polytope = canonical_polytope.intersect(
        Polytope(convex_subpolytopes=[positive_canonical_convex_polytope]))
    canonical_polytope = canonical_polytope.reduce()

    return canonical_polytope


# TODO: This needs a corresponding scissors congruence branch.
def canonical_to_alcove_coordinate(x, y, z):
    normalizing_factor = np.pi

    return ((x + y - z) / normalizing_factor,
            (x - y + z) / normalizing_factor,
            (-x + y + z) / normalizing_factor)
