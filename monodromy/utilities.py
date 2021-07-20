"""
monodromy/utilities.py

Depository for generic python utility snippets.
"""


from fractions import Fraction
from functools import wraps
from typing import List
import warnings

import numpy as np


epsilon = 1e-6  # Fraction(1, 1_000_000)


memoized_attr_bucket = '_memoized_attrs'


def memoized_property(fget):
    attr_name = f'_{fget.__name__}'

    @wraps(fget)
    def fget_memoized(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fget(self))
            if hasattr(self, memoized_attr_bucket):
                getattr(self, memoized_attr_bucket).append(attr_name)
            else:
                setattr(self, memoized_attr_bucket, [attr_name])
        return getattr(self, attr_name)

    return property(fget_memoized)


def clear_memoization(obj):
    if hasattr(obj, memoized_attr_bucket):
        for field in getattr(obj, memoized_attr_bucket):
            delattr(obj, field)
        delattr(obj, memoized_attr_bucket)
    return obj


# https://graphics.stanford.edu/~seander/bithacks.html#NextBitPermutation
def bit_iteration(length, weight):
    """Iterate over bitpatterns of size `length` with `weight` bits flagged."""
    if weight == 0:
        yield 0
        return

    pattern = 2 ** weight - 1
    while pattern < 2 ** length:
        yield pattern
        t = (pattern | (pattern - 1)) + 1
        pattern = t | ((((t & -t) // (pattern & -pattern)) >> 1) - 1)


def bitcount(bits):
    return bin(bits).count('1')


def bitscatter(bits, mask):
    """
    Scatters the contents of bitvector `bits` onto the raised bits in `mask`.
    """
    value = 0
    mask_walker = enumerate(reversed(bin(mask)[2:]))
    for bit_index, mask_index in enumerate([x for x, y in mask_walker if y == '1']):
        value |= (bits & (1 << bit_index)) << (mask_index - bit_index)
    return value


def fractionify(table) -> List[List[Fraction]]:
    """
    Convenience routine for not writing Fraction() a whole bunch.

    NOTE: This can be poorly behaved if your rationals don't have exact floating
          point representations!
    """
    return [[Fraction(i) for i in j] for j in table]


def lcm(*numbers):
    import math

    assert 1 <= len(numbers)
    ret = numbers[0]
    for number in numbers[1:]:
        ret = ret * number // math.gcd(ret, number)
    return ret


def nearp(x, y, modulus=np.pi/2, epsilon=epsilon):
    """
    Checks whether two points are near each other, accounting for float jitter
    and wraparound.
    """
    return abs(np.mod(abs(x - y), modulus)) < epsilon or \
           abs(np.mod(abs(x - y), modulus) - modulus) < epsilon


def l1_distance(x, y):
    """
    Computes the l_1 / Manhattan distance between two coordinates.
    """
    return sum([abs(xx - yy) for xx, yy in zip(x, y)])


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
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            return np.arccos(numerator / denominator)
