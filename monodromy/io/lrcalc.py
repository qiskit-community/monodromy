"""
monodromy/io/lrcalc.py

Extracts quantum Littlewood-Richardson coefficients from the package `lrcalc`.

This package is cumbersome to install, so we provide a prebaked copy of this
table in `qlr_table.py`.
"""

from copy import copy

import lrcalc


def qlr(r, k, a, b):
    """
    Computes the quantum Littlewood-Richardson coefficients N_{ab}^{c, d} in the
    small quantum cohomology ring of the Grassmannian Gr(r, k).  For supplied
    a and b, this computes the set of c and d for which N = 1.

    Returns a dictionary of the form {c: d} over values where N_ab^{c, d} = 1.
    """
    return {
        tuple(list(c) + [0]*(r - len(c))):
            (sum(a) + sum(b) - sum(c)) // (r + k)
        for c, value in lrcalc.mult_quantum(a, b, r, k).items() if value == 1
    }


def displacements(r, k, skip_to=None):
    """
    Iterates over the ordered sequence of partitions of total size `r + k` into
    `r` parts, presented as displacements from the terminal partiiton.

    If `skip_to` is supplied, start enumeration from this element.
    """
    def normalize(p, r, k):
        """
        Roll the odometer `p` over until it becomes a legal (`r`, `k`)
        displacement.
        """
        if p[0] > k:
            return None
        for index, (item, next_item) in enumerate(zip(p, p[1:])):
            if next_item > item:
                p[index+1] = 0
                p[index] += 1
                return normalize(p, r, k)
        return p
    
    ok = skip_to is None
    p = [0 for j in range(0, r)]
    while p is not None:
        if p == skip_to:
            ok = True
        if ok:
            yield copy(p)
        p[-1] += 1
        p = normalize(p, r, k)


def regenerate_qlr_table():
    """
    Uses `lrcalc` to rebuild the table stored in `qlr_table.py`.
    """
    qlr_table = []  # [[r, k, [*a], [*b], [*c], d], ...]
    for r in range(1, 4):
        k = 4 - r
        # r bounds the length; k bounds the contents
        for a in displacements(r, k):
            for b in displacements(r, k, skip_to=a):
                for c, d in qlr(r, k, a, b).items():
                    qlr_table.append([r, k, a, b, list(c), d])
    return qlr_table
