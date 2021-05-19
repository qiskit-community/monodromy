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
