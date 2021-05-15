"""
monodromy/utilities.py

Depository for generic python utility snippets.
"""


from fractions import Fraction
from functools import wraps
from typing import List


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


def fractionify(table) -> List[List[Fraction]]:
    """
    Convenience routine for not writing Fraction() a whole bunch.

    NOTE: This can be poorly behaved if your rationals don't have exact floating
          point representations!
    """
    return [[Fraction(i) for i in j] for j in table]