"""
monodromy/xx_decompose/precalculate.py

Precalculates the "backsolution" polytopes used for constructing right-angled
paths.

NOTE: This code is _not_ for export into QISKit.
"""

from collections import Counter
from typing import List

from .circuits import CircuitPolytope
from ..coverage import prereduce_operation_polytopes
from ..elimination import cylinderize
from ..polytopes import ConvexPolytope, Polytope


def calculate_unordered_scipy_coverage_set(
        coverage_set: List[CircuitPolytope],
        operations: List[CircuitPolytope],
        chatty=False
) -> List[CircuitPolytope]:
    """
    The terms in `coverage_set` are related by equations of the form

        P_(a1 ... a(n-1)) O_(an) = P_(a1 ... an),

    where O_an is a choice of some element in `operations`.  As part of
    `build_coverage_set`, we calculate a "frontier" of P_(a1 ... an) in order to
    exhaust the Weyl alcove, but we discard the relationship described above.
    In order to produce circuits, it's useful to have this relationship
    available, so we re-compute it here so that it's available for input to
    `scipy_unordered_decomposition_hops`.

    We make a major further (conjectural) assumption: that the coverage polytope
    corresponding to a fixed sequence of X-interactions is invariant under
    permutation of the interaction strengths.  This has two effects:
     + We can trim the total amount of computation performed.
     + For another, the "right-angled" assumption is only valid if we allow the
       path builder to choose which gate it will cleave from a circuit, rather
       than requiring it to pick the "last" gate in some specified permutation.
       By scanning over permutations during decomposition, we avoid the pitfall.
    """
    coordinates = {
        "a": [0, 1, 2, 3],
        "b": [0, 4, 5, 6],
        "c": [0, 7, 8, 9],
    }

    inflated_operation_polytope = prereduce_operation_polytopes(
        operations=operations,
        target_coordinate="a",
        background_polytope=Polytope(convex_subpolytopes=[
            # equate first source and first target coordinates
            ConvexPolytope(inequalities=[
                [0,  1,  1, 0, 0, 0, 0, -1, -1, 0],
                [0, -1, -1, 0, 0, 0, 0,  1,  1, 0],
            ]),
            # equate first source and second target coordinates
            ConvexPolytope(inequalities=[
                [0,  1,  1, 0, 0, 0, 0, -1, 0, -1],
                [0, -1, -1, 0, 0, 0, 0,  1, 0,  1],
            ]),
            # equate first source and third target coordinates
            ConvexPolytope(inequalities=[
                [0,  1,  1, 0, 0, 0, 0, 0, -1, -1],
                [0, -1, -1, 0, 0, 0, 0, 0,  1,  1],
            ]),
            # equate second source and second target coordinates
            ConvexPolytope(inequalities=[
                [0,  1, 0,  1, 0, 0, 0, -1, 0, -1],
                [0, -1, 0, -1, 0, 0, 0,  1, 0,  1],
            ]),
            # equate second source and third target coordinates
            ConvexPolytope(inequalities=[
                [0,  1, 0,  1, 0, 0, 0, 0, -1, -1],
                [0, -1, 0, -1, 0, 0, 0, 0,  1,  1],
            ]),
            # equate third source and third target coordinates
            ConvexPolytope(inequalities=[
                [0, 0,  1,  1, 0, 0, 0, 0, -1, -1],
                [0, 0, -1, -1, 0, 0, 0, 0,  1,  1],
            ])]),
        chatty=chatty,
    )

    scipy_coverage_set = []

    if chatty:
        print("Working on scipy precalculation.")

    # we're looking to walk "up" from descendants to ancestors.
    for descendant_polytope in coverage_set:
        if 0 == len(descendant_polytope.operations):
            continue

        if chatty:
            print(f"Precalculating for {'.'.join(descendant_polytope.operations)}...")

        for operation_polytope in operations:
            operation = operation_polytope.operations[0]
            # if we don't use this operation, we can't backtrack along it.
            if operation not in descendant_polytope.operations:
                continue

            # otherwise, locate an up-to-reordering ancestor.
            ancestor_polytope = next(
                (polytope for polytope in coverage_set
                 if Counter(descendant_polytope.operations) ==
                    Counter([operation] + polytope.operations)),
                None
            )
            if ancestor_polytope is None:
                if chatty:
                    print(f"{'.'.join(descendant_polytope.operations)} has no "
                          f"ancestor along {operation}.")
                    print("Available coverage set entries:")
                    for x in coverage_set:
                        print(f"{'.'.join(x.operations)}")
                raise ValueError(f"{'.'.join(descendant_polytope.operations)} "
                                 f"has no ancestor along {operation}.")

            if chatty:
                print(f"    ... backtracking along {operation} to "
                      f"{'.'.join(ancestor_polytope.operations)}...")

            # also impose whatever constraints we were given besides
            backsolution_polytope = inflated_operation_polytope[operation] \
                .intersect(cylinderize(
                    ancestor_polytope,
                    coordinates["a"],
                    parent_dimension=7
                )) \
                .reduce()

            scipy_coverage_set.append(CircuitPolytope(
                convex_subpolytopes=backsolution_polytope.convex_subpolytopes,
                cost=descendant_polytope.cost,
                operations=ancestor_polytope.operations + [operation],
            ))

    return scipy_coverage_set
