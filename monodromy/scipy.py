"""
monodromy/scipy.py

Decomposition routines which are based on `scipy` rather than on `lrs`.

NOTE: The routines in this file can fail for numerical reasons, and so they are
      lightly randomized and meant to be repeatedly called.
"""

from random import shuffle, uniform
from typing import List
import warnings

import numpy as np

import scipy
from scipy.optimize import linprog

from .coverage import GatePolytope, prereduce_operation_polytopes, rho_reflect
from .elimination import cylinderize, project
from .examples import alcove_c2, exactly, fractionify
from .qlr_table import alcove, qlr_polytope
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


def calculate_scipy_coverage_set(coverage_set, operations):
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
            # fix CAN(a, *, *)
            ConvexPolytope(inequalities=fractionify([
                [0,  1,  1, 0, 0, 0, 0, -1, -1, 0],
                [0, -1, -1, 0, 0, 0, 0,  1,  1, 0],
            ])),
            # fix CAN(*, b, *)
            ConvexPolytope(inequalities=fractionify([
                [0,  1, 0,  1, 0, 0, 0, -1, 0, -1],
                [0, -1, 0, -1, 0, 0, 0,  1, 0,  1],
            ])),
            # fix CAN(*, *, c)
            ConvexPolytope(inequalities=fractionify([
                [0, 0,  1,  1, 0, 0, 0, 0, -1, -1],
                [0, 0, -1, -1, 0, 0, 0, 0,  1,  1],
            ]))]),
    )

    scipy_coverage_set = []

    for operation_polytope in coverage_set:
        if 0 == len(operation_polytope.operations):
            continue

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

        scipy_coverage_set.append(GatePolytope(
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
        coverage_set: List[GatePolytope],
        scipy_coverage_set: List[GatePolytope],
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
            raise ValueError("Unable to find precalculated backsolution polytope.")

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

        if solution is not None and solution.success:
            # a/k/a decomposition.push
            decomposition.insert(
                0,
                (solution.x[:3], working_operations[-1], solution.x[-3:])
            )
            # NOTE: using `exactly` here causes an infinite loop.
            target_polytope = nearly(*solution.x[:3])
            working_operations = working_operations[:-1]
        else:
            raise ValueError("Empty backsolution polytope.")

    return decomposition
