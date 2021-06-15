"""
monodromy/xx_decompose/scipy.py

Utilities for interacting with polytope data through scipy.
"""

from copy import copy
from itertools import combinations
from random import sample, shuffle
from typing import Optional, Callable
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
        convex_polytope: ConvexPolytopeData,
        jacobian: Optional[Callable] = None,
        hessian: Optional[Callable] = None,
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
        constraints.append(scipy.optimize.LinearConstraint(
            A_ub, -b_ub, [np.inf] * b_ub.shape[0]  # trivial upper bound
        ))

    if 0 < len(convex_polytope.equalities):
        dimension = -1 + len(convex_polytope.equalities[0])
        A_eq = np.array([[float(x) for x in eq[1:]]
                         for eq in convex_polytope.equalities])
        b_eq = np.array([float(eq[0]) for eq in convex_polytope.equalities])
        constraints.append(scipy.optimize.LinearConstraint(
            A_eq, -b_eq, b_eq
        ))

    kwargs = {}
    if jacobian is not None:
        kwargs["jac"] = jacobian
    if hessian is not None:
        kwargs["hess"] = hessian

    return scipy.optimize.minimize(
        # method='trust-constr',
        fun=fn,
        x0=np.array([1 / 4] * dimension),
        constraints=constraints,
        # options={'verbose': 3},
        **kwargs
    )


def polyhedron_has_element(polytope, point):
    """
    A standalone variant of Polytope.has_element, specialized to the 3D case.
    """
    for cp in polytope.convex_subpolytopes:
        violated = False
        for inequality in cp.inequalities:
            value = inequality[0] + point[0] * inequality[1] + \
                point[1] * inequality[2] + point[2] * inequality[3]
            # this inequality has to be (near) positive
            if -epsilon > value:
                violated = True
                break
        if violated:
            continue
        for equality in cp.equalities:
            value = equality[0] + point[0] * equality[1] + \
                point[1] * equality[2] + point[2] * equality[3]
            # this equality has to be (near) zero)
            if epsilon < abs(value):
                violated = True
        if not violated:
            return True
    return False


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
                if polyhedron_has_element(polytope, vertex):
                    # vertices.append(vertex)
                    return vertex
            except np.linalg.LinAlgError:
                pass

    if 0 == len(vertices):
        raise NoFeasibleSolutions()

    return sample(vertices, 1)[0]
