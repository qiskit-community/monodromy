"""
monodromy/static/matrices.py

Matrix representations of various standard gates.
"""

import numpy as np


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
