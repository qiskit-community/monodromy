"""
optimize.py

Example script showing how to optimize a gateset for performance against a user-
defined cost metric.

NOTE: The optimization loop requires `pybobyqa`, a derivative-free optimizer.
"""

from itertools import count
import math
from time import perf_counter

import numpy as np
import pybobyqa

from monodromy.coverage import *
from monodromy.static.examples import *
from monodromy.haar import expected_cost


cost_table = {}
gateset_dimension = 2  # how many gates to include beyond a full CX


# useful for reproducibility
np.random.seed(0)


# We take cost to be average gate infidelity and assume that it is approximately
# additive in our range of interest.  The following magic values reflect some
# internal experimental data.
def operation_cost(
        strength: Fraction,
        # note: Isaac reports this value in percent per degree
        scale_factor: float = (64 * 90) / (10000 * 100),
        # first component: 2Q invocation cost; second component: local cost
        offset: float = 909 / (10000 * 100) + 1 / 1000,
):
    return strength * scale_factor + offset


def get_operations(*strengths):
    """
    Builds a family of XX-type operations, where each strength in `strengths` is
    specified as a fraction of the "full strength" XX-type operation, CX.
    """
    return [
        CircuitPolytope(
            convex_subpolytopes=exactly(
                Fraction( 1, 4) * strength,
                Fraction( 1, 4) * strength,
                Fraction(-1, 4) * strength
            ).convex_subpolytopes,
            cost=operation_cost(strength),
            operations=[f"{str(strength)} XX"],
        ) for strength in strengths
    ]


def objective(ratios):
    """
    Function to be optimized: consumes a family of interaction strengths, then
    computes the expected cost of compiling a Haar-randomly chosen 2Q operator.
    """
    timer_coverage = perf_counter()
    operations = get_operations(*ratios)
    strengths_string = ', '.join([str(s) + " XX" for s in ratios])
    print(f"Working on " + strengths_string)
    coverage_set = build_coverage_set(operations, chatty=True)
    timer_coverage = perf_counter() - timer_coverage
    timer_haar = perf_counter()
    cost = expected_cost(coverage_set, chatty=True)
    timer_haar = perf_counter() - timer_haar
    cost_table[tuple(ratios)] = cost
    print(
        f"{strengths_string} took {timer_coverage:.3f}s + {timer_haar:.3f}s = "
        f"{timer_coverage + timer_haar:.3f}s")
    return cost


def rescaled_objective(ratios, max_denominator=100):
    """
    `objective` with its domain rescaled for easier use by `pybobyqa`: the
    sequence

        [a1, a2, ..., an]

    is forwarded to `objective` as

        [b1, b2, ..., bn] = [a1, a1 * a2, ..., a1 * a2 * ... an],

    so that 0 <= a1, a2, ..., an <= 1 implies b1 >= b2 >= ... >= bn.
    """
    triangular_strengths = []
    for ratio in [1, *ratios]:  # automatically include a full CX
        if 0 < len(triangular_strengths):
            previous_strength = triangular_strengths[-1]
        else:
            previous_strength = Fraction(1)
        triangular_strengths.append(
            previous_strength * Fraction(ratio)
            .limit_denominator(max_denominator)
        )

    return objective(triangular_strengths)


def print_cost_table():
    """
    Utility function for printing the expected costs calculated so far.
    """
    print("Dumping cost table:")
    for k, v in cost_table.items():
        print("{" + ', '.join([str(kk) for kk in k] + [str(v)]) + "}, ")
    if 0 < len(cost_table):
        minimum_key = min(cost_table, key=cost_table.get)
        print("Best point: {" +
              ', '.join(list([str(kk) for kk in minimum_key]) +
                        [str(cost_table[minimum_key])]) +
              "}")
    print("=====")


################################################################################

# make the best use of time by first using `pybobyqa` to calculate an optimal
# gateset.
x0 = np.array([Fraction(1, 2)] * gateset_dimension)
solution = pybobyqa.solve(
    rescaled_objective, x0,
    bounds=([0] * gateset_dimension, [1] * gateset_dimension),
    objfun_has_noise=False,
    print_progress=True,
    rhoend=1e-4
)

print("Optimizer solution:")
print(solution)


################################################################################

# with that done, we can continue to sample points to flesh out the space to
# turn it into a nice plot overall.

def iterate_over_total(
    total,
    bucket_count,
    fn,
    partial_fill=None
):
    partial_fill = partial_fill if partial_fill is not None else []
    if bucket_count == len(partial_fill):
        return fn(partial_fill)

    if bucket_count == 1 + len(partial_fill):
        if total - sum(partial_fill) >= 2:
            return iterate_over_total(
                total,
                bucket_count,
                fn,
                [*partial_fill, total - sum(partial_fill)]
            )
        else:
            return

    for denominator in range(1, total - sum(partial_fill)):
        iterate_over_total(
            total,
            bucket_count,
            fn,
            [*partial_fill, denominator]
        )


def iterate_over_numerators(
    denominators,
    fn,
    partial_fill=None
):
    partial_fill = partial_fill if partial_fill is not None else []
    if 0 == len(denominators):
        return fn(partial_fill)
    for j in range(1, denominators[0]):
        if 0 < len(partial_fill) and j / denominators[0] >= partial_fill[-1]:
            continue
        if math.gcd(j, denominators[0]) != 1:
            continue
        iterate_over_numerators(
            denominators[1:],
            fn,
            partial_fill=[*partial_fill, Fraction(j, denominators[0])]
        )


# this loop enumerates rational tuples whose denominators grow maximally slowly,
# which do not repeat, and which are sorted descending.
#
# it also includes the full CX in the call to `objective`, to match the behavior
# of the optimization step above.
#
# this runs forever. use KeyboardInterrupt to stop when you're satisfied.
for total in count(1):
    iterate_over_total(
        total,
        gateset_dimension,
        lambda denominators: [
            iterate_over_numerators(
                denominators,
                lambda ratios: objective([1, ] + ratios)
            ),
            print_cost_table()
        ]
    )
