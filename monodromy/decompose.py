"""
monodromy/decompose.py

Routines for decomposing canonical gates into certain interactions.
"""

from functools import reduce
import warnings

import numpy as np

from .examples import canonical_matrix, rz_matrix


# TODO: THIS IS A STOPGAP!!!
def safe_arccos(numerator, denominator):
    """
    Computes arccos(n/d) with different (better?) numerical stability.
    """
    threshold = 0.005

    if abs(numerator) > abs(denominator) and \
            abs(numerator - denominator) < threshold:
        return 0.0
    elif abs(numerator) > abs(denominator) and \
            abs(numerator + denominator) < threshold:
        return np.pi
    else:
        return np.arccos(numerator / denominator)


def decompose_xxyy_into_xxyy_xx(a_target, b_target, a1, b1, a2):
    """
    Consumes a target canonical interaction CAN(a_target, b_target) and
    source interactions CAN(a1, b1), CAN(a2), then manufactures a
    circuit identity of the form

    CAN(a_target, b_target) = (Zr, Zs) CAN(a1, b1) (Zu, Zv) CAN(a2) (Zx, Zy).

    Returns the 6-tuple (r, s, u, v, x, y).
    """

    cplus, cminus = np.cos(a1 + b1), np.cos(a1 - b1)
    splus, sminus = np.sin(a1 + b1), np.sin(a1 - b1)
    ca, sa = np.cos(a2), np.sin(a2)

    uplusv = 1 / 2 * safe_arccos(
        cminus ** 2 * ca ** 2 + sminus ** 2 * sa ** 2 - np.cos(a_target - b_target) ** 2,
        2 * cminus * ca * sminus * sa
    )
    uminusv = 1 / 2 * safe_arccos(
        cplus ** 2 * ca ** 2 + splus ** 2 * sa ** 2 - np.cos(a_target + b_target) ** 2,
        2 * cplus * ca * splus * sa
    )

    u, v = (uplusv + uminusv) / 2, (uplusv - uminusv) / 2

    # NOTE: the target matrix is phase-free
    middle_matrix = reduce(np.dot, [
        canonical_matrix(a1, b1),
        np.kron(rz_matrix(u), rz_matrix(v)),
        canonical_matrix(a2),
    ])

    phase_solver = np.array([
        [1 / 4, 1 / 4, 1 / 4, 1 / 4, ],
        [1 / 4, -1 / 4, -1 / 4, 1 / 4, ],
        [1 / 4, 1 / 4, -1 / 4, -1 / 4, ],
        [1 / 4, -1 / 4, 1 / 4, -1 / 4, ],
    ])
    inner_phases = [
        np.angle(middle_matrix[0, 0]),
        np.angle(middle_matrix[1, 1]),
        np.angle(middle_matrix[1, 2]) + np.pi / 2,
        np.angle(middle_matrix[0, 3]) + np.pi / 2,
    ]
    r, s, x, y = np.dot(phase_solver, inner_phases)

    # If there's a phase discrepancy, need to conjugate by an extra Z/2 (x) Z/2.
    generated_matrix = reduce(np.dot, [
        np.kron(rz_matrix(r), rz_matrix(s)),
        middle_matrix,
        np.kron(rz_matrix(x), rz_matrix(y)),
    ])
    if ((abs(np.angle(generated_matrix[3, 0]) - np.pi / 2) < 0.01 and a_target > b_target) or
            (abs(np.angle(generated_matrix[3, 0]) + np.pi / 2) < 0.01 and a_target < b_target)):
        x += np.pi / 4
        y += np.pi / 4
        r -= np.pi / 4
        s -= np.pi / 4

    return r, s, u, v, x, y
