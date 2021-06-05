"""
monodromy/serialize.py

A staging area for de/serialization routines.
"""

from collections import Counter
from dataclasses import asdict
from typing import List

from monodromy.circuits import OperationPolytope
from monodromy.coverage import CircuitPolytope, CircuitPolytopeData, \
    intersect_and_project
from monodromy.examples import alcove_c2, everything_polytope, identity_polytope
from monodromy.scipy import calculate_unordered_scipy_coverage_set


def generate_deflated_coverage_data(operations, chatty=True):
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
        output_polytope = intersect_and_project(
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
        if not output_polytope.contains(alcove_c2):
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


def inflate_scipy_data(
        operations: List[OperationPolytope],
        *,
        serialized_coverage_set=None,
        serialized_scipy_coverage_set=None,
        chatty=True,
):
    """
    Reinflates the compilation tables to be supplied to `MonodromyZXDecomposer`.
    """
    # reinflate the polytopes, simultaneously calculating their current cost
    inflated_polytopes = [
        CircuitPolytopeData.inflate(
            {**v, "cost": sum([x * y.cost for x, y in zip(k, operations)])}
        ) for k, v in serialized_coverage_set.items()
    ]

    # retain only the low-cost polytopes, discarding everything after a
    # universal template has been found.
    cost_trimmed_polytopes = []
    for polytope in sorted(inflated_polytopes, key=lambda x: x.cost):
        if chatty:
            print(f"Keeping {'.'.join(polytope.operations)}: {polytope.cost}")
        cost_trimmed_polytopes.append(polytope)
        if (set([tuple(x) for x in
                 polytope.convex_subpolytopes[0].inequalities]) ==
            set([tuple(x) for x in
                 alcove_c2.reduce().convex_subpolytopes[0].inequalities])):
            break

    if chatty:
        print(f"Kept {len(cost_trimmed_polytopes)} regions.")

    # then, discard those polytopes which perfectly overlap previously seen
    # polytopes.  this is computationally expensive to do exactly, so we use an
    # approximate check instead.
    seen_polytopes = []
    coverage_trimmed_polytopes = []
    for polytope in cost_trimmed_polytopes:
        if chatty:
            print(f"Reconsidering {'.'.join(polytope.operations)}... ", end="")
        these_polytopes = [
            (set([tuple(x) for x in cp.inequalities]),
             set([tuple(y) for y in cp.equalities]))
            for cp in polytope.convex_subpolytopes
        ]
        if all(p in seen_polytopes for p in these_polytopes):
            if chatty:
                print("skipping.")
            continue
        if chatty:
            print("keeping.")
        seen_polytopes += these_polytopes
        coverage_trimmed_polytopes.append(polytope)

    if chatty:
        print(f"Kept {len(coverage_trimmed_polytopes)} regions.")

    # finally, re-inflate the relevant subset of the scipy precomputation.
    reinflated_scipy = []
    coverage_trimmed_signatures = [Counter(x.operations)
                                   for x in coverage_trimmed_polytopes]
    for x in serialized_scipy_coverage_set:
        x = CircuitPolytopeData.inflate(x)
        if Counter(x.operations) in coverage_trimmed_signatures:
            reinflated_scipy.append(x)

    return {
        "operations": operations,
        "coverage_set": coverage_trimmed_polytopes,
        "precomputed_backsolutions": reinflated_scipy,
    }
