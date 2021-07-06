"""
monodromy/io/inflate.py

Routines for re-inflating a previously exported coverage set.
"""

from collections import Counter
from typing import Dict, List, Tuple

from ..xx_decompose.circuits import OperationPolytope
from .base import CircuitPolytopeData
from ..coordinates import monodromy_alcove_c2


def inflate_scipy_data(deflated_data):
    """
    Re-inflates serialized coverage set data.
    """

    coverage_set = {
        k: CircuitPolytopeData.inflate(v)
        for k, v in deflated_data["coverage_set"].items()
    }
    precomputed_backsolutions = [
        CircuitPolytopeData.inflate(d)
        for d in deflated_data["precomputed_backsolutions"]
    ]

    return {
        "coverage_set": coverage_set,
        "precomputed_backsolutions": precomputed_backsolutions,
    }


def filter_scipy_data(
        operations: List[OperationPolytope],
        *,
        coverage_set: Dict[Tuple, CircuitPolytopeData] = None,
        precomputed_backsolutions: List[CircuitPolytopeData] = None,
        chatty=True,
):
    """
    Attaches costs to the tables to be supplied to `MonodromyZXDecomposer`.
    """
    # reinflate the polytopes, simultaneously calculating their current cost
    inflated_polytopes = [
        CircuitPolytopeData(
            cost=sum([x * y.cost for x, y in zip(k, operations)]),
            convex_subpolytopes=v.convex_subpolytopes,
            operations=v.operations,
        ) for k, v in coverage_set.items()
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
                 monodromy_alcove_c2.convex_subpolytopes[0].inequalities])):
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
    reinflated_scipy = [
        x for x in precomputed_backsolutions
        if Counter(x.operations) in coverage_trimmed_signatures
    ]

    return {
        "operations": operations,
        "coverage_set": coverage_trimmed_polytopes,
        "precomputed_backsolutions": reinflated_scipy,
    }
