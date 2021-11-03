"""
monodromy/io/deflate.py

Routines for deflating a coverage set for export.
"""

from dataclasses import asdict
from typing import Dict, List

from ..coordinates import monodromy_alcove_c2
from ..coverage import CircuitPolytope, deduce_qlr_consequences
from ..xx_decompose.precalculate import calculate_unordered_scipy_coverage_set
from ..static import everything_polytope, identity_polytope


def generate_deflated_coverage_data(
        operations: List[CircuitPolytope],
        chatty=True
) -> Dict:
    """
    Generates the deflated data tables used to prime the `MonodromyZXDecomposer`
    compilation pass.  Returns a dictionary of relevant tables.
    """
    # Generate data for all possible combinations (with multiplicity) of
    # operations, stopping only when adding a gate does not improve coverage.
    buckets = [1] + [0] * (len(operations) - 1)
    coverage_set = {
        (0,) * len(operations): CircuitPolytope(
            cost=0.,
            operations=[],
            convex_subpolytopes=identity_polytope.convex_subpolytopes
        )
    }

    while True:
        if chatty:
            print("Working on " +
                  ', '.join(str(b) + ' ' + o.operations[0]
                            for b, o in zip(buckets, operations)) +
                  ".")
        # find an antecedent CircuitPolytope (and edge) from coverage_set
        first_nonzero_index = next((i for i, j in enumerate(buckets) if j != 0),
                                   None)
        decremented_tuple = tuple(j if i != first_nonzero_index else j - 1
                                  for i, j in enumerate(buckets))
        input_polytope = coverage_set[decremented_tuple]
        operation_polytope = operations[first_nonzero_index]

        # calculate CircuitPolytope for this bucket value
        output_polytope = deduce_qlr_consequences(
            target="c",
            a_polytope=input_polytope,
            b_polytope=operation_polytope,
            c_polytope=everything_polytope,
        )
        output_polytope = CircuitPolytope(
            operations=sum([count * operation.operations
                            for count, operation in zip(buckets, operations)],
                           []),
            cost=0.,
            convex_subpolytopes=output_polytope.convex_subpolytopes
        )

        # stash this into coverage_set
        coverage_set[tuple(buckets)] = output_polytope

        # increment the bucket counters
        if not output_polytope.contains(monodromy_alcove_c2):
            buckets[0] += 1
        else:
            # if it has perfect coverage, roll over:
            # zero out the first nonzero bucket, increment the one after it
            buckets[first_nonzero_index] = 0
            if first_nonzero_index + 1 == len(buckets):
                # if rolling over overflows, break
                break
            else:
                buckets[first_nonzero_index + 1] += 1

    # also perform the scipy precalculation
    scipy_coverage_set = calculate_unordered_scipy_coverage_set(
        list(coverage_set.values()), operations, chatty=chatty
    )

    return {
        "serialized_coverage_set": {
            k: asdict(v)
            for k, v in coverage_set.items()
        },
        "serialized_scipy_coverage_set": [
            asdict(x) for x in scipy_coverage_set
        ],
    }
