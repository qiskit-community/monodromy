"""
scripts/gateset.py

Example script showing how to optimize a(n XX-based) gateset for performance
against a user-defined cost metric.

NOTE: The optimization loop requires `pybobyqa`, a derivative-free optimizer.

NOTE: We don't make use of tools which are further specialized for XX-based
      gate-sets.  By modifying the `exactly` in `get_operations`, a user could
      optimize over any shape of gateset they please.

NOTE: `rescaled_objective` always includes a full XX.
"""

from itertools import count
import math
from time import perf_counter

import numpy as np
import pybobyqa

from monodromy.coverage import *
from monodromy.static.examples import *
from monodromy.haar import cost_statistics  # , expected_cost

gateset_dimension = 1  # how many gates to include beyond a full CX
filename = "gateset_landscape_1d.dat"  # .dat file with expected cost info

#
# ERROR MODEL
#
# We assume that the infidelity cost of a native 2Q interaction is affinely
# related to the interaction strength.  The following two values track the
# offset and the slope of this affine-linear function.
#
# first summand: 2Q invocation cost; second summand: cost of local gates
offset = 909 / (10000 * 100) + 1 / 1000
# note: Isaac reports this value in percent per degree
scale_factor = (64 * 90) / (10000 * 100)


def operation_cost(
        strength: Fraction,
        scale_factor: float = scale_factor,
        offset: float = offset,
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
        ) for strength in set(strengths)
    ]


# useful for reproducibility
np.random.seed(0)

# tuples of descending strengths in [0, 1]
#     -> {"average_cost", "average_overshot", "sigma_cost", "sigma_overshot"}
cost_table = {}


def objective(ratios):
    """
    Function to be optimized: consumes a family of interaction strengths, then
    computes the expected cost of compiling a Haar-randomly chosen 2Q operator.
    """
    global offset, scale_factor

    timer_coverage = perf_counter()
    operations = get_operations(*ratios)
    strengths_string = ', '.join([str(s) + " XX" for s in ratios])
    print(f"Working on " + strengths_string)
    coverage_set = build_coverage_set(operations, chatty=True)
    timer_coverage = perf_counter() - timer_coverage
    timer_haar = perf_counter()
    cost_table[tuple(ratios)] = cost_statistics(
        coverage_set, offset=offset, scale_factor=scale_factor, chatty=True
    )
    cost = cost_table[tuple(ratios)]["average_cost"]
    timer_haar = perf_counter() - timer_haar
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
    global filename, gateset_dimension

    keys = ["average_cost", "average_overshot", "sigma_cost", "sigma_overshot"]

    print("Dumping cost table...")
    with open(filename, "w") as fh:
        fh.write(' '.join([f'strength{n}' for n in range(1 + gateset_dimension)])
                 + " " + ' '.join(keys) + '\n')
        for k, v in cost_table.items():
            fh.write(' '.join(str(float(entry)) for entry in k) + ' ' +
                     ' '.join(str(v[key]) for key in keys) + '\n')
    print("Dumped.")


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

print(cost_table)


################################################################################

print("Now we enter an infinite loop to flesh out the gateset landscape and "
      "turn it into a nice plot overall.  Use KeyboardInterrupt to quit "
      "whenever you're satisfied.")


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
