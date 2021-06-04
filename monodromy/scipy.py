"""
monodromy/scipy.py

Decomposition routines which are based on `scipy` rather than on `lrs`.

NOTE: The routines in this file can fail for numerical reasons, and so they are
      lightly randomized and meant to be repeatedly called.
"""

from collections import Counter
from random import shuffle, uniform
from typing import List
import warnings

import numpy as np

import scipy
from scipy.optimize import linprog

from .circuits import NoBacksolution
from .coverage import CircuitPolytope, prereduce_operation_polytopes
from .elimination import cylinderize
from .examples import exactly
from .polytopes import ConvexPolytope, Polytope


def nearly(ax, ay, az, wiggle=1e-10):
    """
    Like `exactly`, but with floating point wiggle room.
    """
    return Polytope(convex_subpolytopes=[ConvexPolytope(inequalities=[
        [ ax + wiggle, -1,  0,  0],
        [-ax + wiggle,  1,  0,  0],
        [ ay + wiggle,  0, -1,  0],
        [-ay + wiggle,  0,  1,  0],
        [ az + wiggle,  0,  0, -1],
        [-az + wiggle,  0,  0,  1],
    ])])


def calculate_scipy_coverage_set(coverage_set, operations, chatty=False):
    """
    Precalculates a set of backsolution polytopes associated to `covering_set`
    and `operations`.

    Used as efficient input to `scipy_decomposition_hops` below.
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
    for operation_polytope in coverage_set:
        if 0 == len(operation_polytope.operations):
            continue

        if chatty:
            print(f"Working on {'.'.join(operation_polytope.operations)}...")

        ancestor_polytope = next(
            (polytope for polytope in coverage_set
             if polytope.operations == operation_polytope.operations[:-1]),
            exactly(0, 0, 0))

        backsolution_polytope = inflated_operation_polytope[
            operation_polytope.operations[-1]
        ]

        # also impose whatever constraints we were given besides
        backsolution_polytope = backsolution_polytope.intersect(
            cylinderize(
                ancestor_polytope,
                coordinates["a"],
                parent_dimension=7
            )
        )
        backsolution_polytope = backsolution_polytope.reduce()

        scipy_coverage_set.append(CircuitPolytope(
            convex_subpolytopes=backsolution_polytope.convex_subpolytopes,
            cost=operation_polytope.cost,
            operations=operation_polytope.operations,
        ))

    return scipy_coverage_set


def scipy_get_random_vertex(
        convex_polytope: ConvexPolytope
) -> scipy.optimize.OptimizeResult:
    """
    Extracts a random extreme point from `convex_polytope` using scipy.

    Returns an OptimizeResult packet, which may have its `success` slot set to
    False.  In this case (and depending on `status`), you might retry the call.
    """
    dimension = None

    if 0 < len(convex_polytope.inequalities):
        dimension = -1 + len(convex_polytope.inequalities[0])
        A_ub = np.array([[-float(x) for x in ineq[1:]]
                         for ineq in convex_polytope.inequalities])
        b_ub = np.array([float(ineq[0])
                         for ineq in convex_polytope.inequalities])
    else:
        A_ub, b_ub = None, None

    if 0 < len(convex_polytope.equalities):
        dimension = -1 + len(convex_polytope.equalities[0])
        A_eq = np.array([[-float(x) for x in eq[1:]]
                         for eq in convex_polytope.equalities])
        b_eq = np.array([float(eq[0]) for eq in convex_polytope.equalities])
    else:
        A_eq, b_eq = None, None

    c = np.array([uniform(-1, 1) for _ in range(dimension)])
    bounds = [(-1, 1)] * dimension

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=scipy.linalg.LinAlgWarning)
        warnings.filterwarnings("ignore", category=scipy.optimize.OptimizeWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        return linprog(
            c=c,
            A_ub=A_ub, b_ub=b_ub,
            A_eq=A_eq, b_eq=b_eq,
            bounds=bounds,
            options=dict(presolve=True, sym_pos=False,
                         # cholesky=False, lstsq=True,
                         tol=1e-10, ),
        )


def scipy_decomposition_hops(
        coverage_set: List[CircuitPolytope],
        scipy_coverage_set: List[CircuitPolytope],
        target_polytope: Polytope
):
    """
    Fixing a `coverage_set` and a `scipy_coverage_set`, finds a minimal
    decomposition for a canonical interaction in `target_polytope` into a
    sequence of operations linking the polytopes in the coverage sets, together
    with specific intermediate canonical points linked by them.

    Returns a list of tuples of shape (source vertex, operation, target vertex),
    so that each target vertex is accessible from its source vertex by
    application of the operation, each target vertex matches its next source
    vertex, the original source vertex corresponds to the identity, and the
    last target lies in `target_polytope`.

    NOTE: `scipy_coverage_set` is extracted from `coverage_set` using
          `calculate_scipy_coverage_set` above.
    """
    decomposition = []  # retval
    working_polytope = None

    # NOTE: if `target_polytope` were an actual point, could use .has_element
    best_cost = float("inf")
    for polytope in coverage_set:
        if polytope.cost < best_cost:
            for convex_subpolytope in \
                    polytope.intersect(target_polytope).convex_subpolytopes:
                solution = scipy_get_random_vertex(convex_subpolytope)

                if solution.success:
                    working_polytope = polytope
                    best_cost = polytope.cost
                    break

    if working_polytope is None:
        raise ValueError(f"{target_polytope} not contained in coverage set.")

    working_operations = working_polytope.operations

    # if this polytope corresponds to the empty operation, we're done.
    while 0 < len(working_operations):
        backsolution_polytope = None
        solution = None

        for polytope in scipy_coverage_set:
            if polytope.operations == working_operations:
                backsolution_polytope = polytope
                break
        if backsolution_polytope is None:
            raise NoBacksolution()

        # impose the target constraints, which sit on "b"
        # (really on "c", but "b" has already been projected off)
        backsolution_polytope = backsolution_polytope.intersect(
            cylinderize(target_polytope, [0, 4, 5, 6],
                        parent_dimension=7)
        )

        # walk over the backsolution polytopes, try to find one that's solvable
        shuffle(backsolution_polytope.convex_subpolytopes)
        for convex_subpolytope in backsolution_polytope.convex_subpolytopes:
            solution = scipy_get_random_vertex(convex_subpolytope)
            if solution.success:
                break

        if solution is None or not solution.success:
            raise NoBacksolution()

        # a/k/a decomposition.push
        decomposition.insert(
            0,
            (solution.x[:3], working_operations[-1], solution.x[-3:])
        )
        # NOTE: using `exactly` here causes an infinite loop.
        target_polytope = nearly(*solution.x[:3])
        working_operations = working_operations[:-1]

    return decomposition


def calculate_unordered_scipy_coverage_set(
        coverage_set,
        operations,
        chatty=False
):
    """
    See `calculate_scipy_coverage_set`.

    Conjecturally, the coverage polytope corresponding to a fixed sequence of
    XX-interactions is invariant under permutation of the interaction strengths.
    For one, we can use this to trim the total amount of computation performed.
    For another, the assumptions built into `xx_circuit_from_decomposition` in
    turn impose assumptions on `decomposition_hop` — and those assumptions do
    not always hold for all possible permutations on a given input coordinate to
    be decomposed.  By scanning over permutations during decomposition, we avoid
    this pitfall.
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
                        print(f"{'.'.join(coverage_set.operations)}")
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


def scipy_unordered_decomposition_hops(
        coverage_set: List[CircuitPolytope],
        scipy_coverage_set: List[CircuitPolytope],
        target_polytope: Polytope
):
    """
    See `scipy_decomposition_hops`.

    Conjecturally, the coverage polytope corresponding to a fixed sequence of
    XX-interactions is invariant under permutation of the interaction strengths.
    For one, we can use this to trim the total amount of computation performed.
    For another, the assumptions built into `xx_circuit_from_decomposition` in
    turn impose assumptions on `decomposition_hop` — and those assumptions do
    not always hold for all possible permutations on a given input coordinate to
    be decomposed.  By scanning over permutations during decomposition, we avoid
    this pitfall.
    """
    decomposition = []  # retval
    working_polytope = None

    # NOTE: if `target_polytope` were an actual point, could use .has_element
    # NOTE: In practice, this computation has already been done.
    best_cost = float("inf")
    for polytope in coverage_set:
        if polytope.cost < best_cost:
            for convex_subpolytope in \
                    polytope.intersect(target_polytope).convex_subpolytopes:
                solution = scipy_get_random_vertex(convex_subpolytope)

                if solution.success:
                    working_polytope = polytope
                    best_cost = polytope.cost
                    break

    if working_polytope is None:
        raise ValueError(f"{target_polytope} not contained in coverage set.")

    working_operations = working_polytope.operations

    # if this polytope corresponds to the empty operation, we're done.
    while 0 < len(working_operations):
        backsolution_polytope = None
        solution = None

        for ancestor in scipy_coverage_set:
            # check that this is actually an ancestor
            if Counter(ancestor.operations) != Counter(working_operations):
                continue

            # impose the target constraints, which sit on "b"
            # (really on "c", but "b" has already been projected off)
            backsolution_polytope = ancestor.intersect(
                cylinderize(target_polytope, [0, 4, 5, 6],
                            parent_dimension=7)
            )

            # walk over the convex backsolution subpolytopes, try to find one
            # that's solvable
            shuffle(backsolution_polytope.convex_subpolytopes)
            for convex_subpolytope in backsolution_polytope.convex_subpolytopes:
                solution = scipy_get_random_vertex(convex_subpolytope)
                if solution.success:
                    break
                solution = None

            if solution is not None:
                break

        if solution is None:
            raise NoBacksolution()

        # a/k/a decomposition.push
        decomposition.insert(
            0,
            (solution.x[:3], ancestor.operations[-1], solution.x[-3:])
        )
        # NOTE: using `exactly` here causes an infinite loop.
        target_polytope = nearly(*solution.x[:3])
        working_operations = ancestor.operations[:-1]

    return decomposition
