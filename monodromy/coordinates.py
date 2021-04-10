"""
monodromy/coordinates.py

There are two common coordinate systems / choices of Weyl alcove in su_4:

+ Canonical parameters: These are the (a, b, c) in the exponential formula:

      exp(-i(a XX + b YY + c ZZ))

  with a, b, c further chosen to satisfy pi/4 >= a >= b >= |c|.
+ Alcove coordinates: These are the alpha, beta, delta in the formula

      exp(diag(i a, i b, i c, -i(a + b + c)),

  with a, b, c further chosen to satisfy a >= b >= c, a - 1 >= -(a + b + c).

(Working in pu_4 adds one extra inequality, enforced elsewhere in this repo
by reflection through the `rho` operator.)

This file contains routines for converting from one system to another.
"""

from functools import reduce

import numpy as np


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
        [0, 0, 0, 1],
        [0, 0, -1, 0],
        [0, -1, 0, 0],
        [1, 0, 0, 0],
    ], dtype=complex)
    gammaQ = reduce(np.dot, [unitary, sysy, unitary.T, sysy])
    logspec_coordinate = np.real(
        np.log(np.linalg.eigvals(gammaQ)) / (2 * np.pi * 1j))
    return normalize_logspec_AC2(
        sorted(np.mod(logspec_coordinate, 1.0), reverse=True)
    )


def alcove_to_canonical_coordinate(x, y, z):
    """
    Given an alcove coordinate, produces its image as a canonical coordinate.
    """
    normalizing_factor = np.pi
    return ((x + y) / 2 * normalizing_factor,
            (z + x) / 2 * normalizing_factor,
            (y + z) / 2 * normalizing_factor)
