"""
monodromy/io/inflate.py

Routines for re-inflating a previously exported coverage set.
"""

from collections import Counter
from typing import List

from ..xx_decompose.circuits import OperationPolytope
from .base import CircuitPolytopeData
from ..coordinates import alcove_c2


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
