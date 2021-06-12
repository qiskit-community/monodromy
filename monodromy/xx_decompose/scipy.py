"""
monodromy/xx_decompose/scipy.py

Utilities for interacting with polytope data through scipy.
"""

from copy import copy
from itertools import combinations
from random import sample, shuffle
import warnings

import numpy as np
import scipy.optimize

from ..backend.backend_abc import NoFeasibleSolutions
from ..io.base import ConvexPolytopeData, PolytopeData
from ..utilities import epsilon


def nearly(ax, ay, az, wiggle=epsilon):
    """
    Like `exactly`, but with floating point wiggle room.
    """
    return PolytopeData(convex_subpolytopes=[ConvexPolytopeData(inequalities=[
        [ ax + wiggle, -1,  0,  0],
        [-ax + wiggle,  1,  0,  0],
        [ ay + wiggle,  0, -1,  0],
        [-ay + wiggle,  0,  1,  0],
        [ az + wiggle,  0,  0, -1],
        [-az + wiggle,  0,  0,  1],
    ])])


def optimize_over_polytope(
        fn,
        convex_polytope: ConvexPolytopeData
) -> scipy.optimize.OptimizeResult:
    """
    Optimizes the function `fn`: array --> reals over `convex_polytope`.
    """
    dimension = None

    constraints = []

    if 0 < len(convex_polytope.inequalities):
        dimension = -1 + len(convex_polytope.inequalities[0])
        A_ub = np.array([[float(x) for x in ineq[1:]]
                         for ineq in convex_polytope.inequalities])
        b_ub = np.array([float(ineq[0])
                         for ineq in convex_polytope.inequalities])
        constraints.append(dict(
            type='ineq',
            fun=lambda x: A_ub @ x + b_ub
        ))

    if 0 < len(convex_polytope.equalities):
        dimension = -1 + len(convex_polytope.equalities[0])
        A_eq = np.array([[float(x) for x in eq[1:]]
                         for eq in convex_polytope.equalities])
        b_eq = np.array([float(eq[0]) for eq in convex_polytope.equalities])
        constraints.append(dict(
            type='ineq',
            fun=lambda x: A_eq @ x + b_eq
        ))
        constraints.append(dict(
            type='ineq',
            fun=lambda x: -A_eq @ x - b_eq
        ))

    return scipy.optimize.minimize(
        fun=fn,
        x0=np.array([1 / 4] * dimension),
        constraints=constraints
    )


def has_element(polytope, point):
    """
    A standalone variant of Polytope.has_element.
    """
    return any([(all([-epsilon <= inequality[0] +
                      sum(x * y for x, y in
                          zip(point, inequality[1:]))
                      for inequality in cp.inequalities]) and
                 all([abs(equality[0] + sum(x * y for x, y in
                                            zip(point, equality[1:])))
                      <= epsilon
                      for equality in cp.equalities]))
                for cp in polytope.convex_subpolytopes])


def scipy_get_random_vertex(
        convex_polytope: ConvexPolytopeData
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

    c = np.array([np.random.uniform(-1, 1) for _ in range(dimension)])
    bounds = [(-1, 1)] * dimension

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=scipy.linalg.LinAlgWarning)
        warnings.filterwarnings("ignore", category=scipy.optimize.OptimizeWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        return scipy.optimize.linprog(
            c=c,
            A_ub=A_ub, b_ub=b_ub,
            A_eq=A_eq, b_eq=b_eq,
            bounds=bounds,
            options=dict(presolve=True, sym_pos=False,
                         # cholesky=False, lstsq=True,
                         tol=1e-10, ),
        )


def manual_get_random_vertex(polytope: PolytopeData):
    """
    Returns a single random vertex from `polytope`.

    Same as `scipy_get_random_vertex`, but computed without scipy.
    """
    vertices = []

    paragraphs = copy(polytope.convex_subpolytopes)
    shuffle(paragraphs)
    for convex_subpolytope in paragraphs:
        sentences = convex_subpolytope.inequalities + \
                    convex_subpolytope.equalities
        shuffle(sentences)
        for inequalities in combinations(sentences, 3):
            A = np.array([x[1:] for x in inequalities])
            b = np.array([x[0] for x in inequalities])
            try:
                vertex = np.linalg.inv(-A) @ b
                if has_element(polytope, vertex):
                    # vertices.append(vertex)
                    return vertex
            except np.linalg.LinAlgError:
                pass

    if 0 == len(vertices):
        raise NoFeasibleSolutions()

    return sample(vertices, 1)[0]
