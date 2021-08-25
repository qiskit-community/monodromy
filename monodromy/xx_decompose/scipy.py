"""
monodromy/xx_decompose/scipy.py

Utilities for interacting with polytope data through scipy / by hand.
"""

from copy import copy
from itertools import combinations
import math
from random import shuffle
from typing import Optional, Callable
import warnings

import numpy as np
import scipy.optimize

from ..exceptions import NoFeasibleSolutions
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


def polytope_has_element(polytope, point):
    """
    A standalone variant of Polytope.has_element.
    """
    return (all([-epsilon <= inequality[0] +
                 sum(x * y for x, y in
                     zip(point, inequality[1:]))
                 for inequality in polytope.inequalities]) and
            all([abs(equality[0] + sum(x * y for x, y in
                                       zip(point, equality[1:])))
                 <= epsilon
                 for equality in polytope.equalities]))


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


def manual_get_random_vertex(polytope):
    """
    Returns a single random vertex from `polytope`.

    Same as `scipy_get_random_vertex`, but computed without scipy.
    """
    if isinstance(polytope, PolytopeData):
        paragraphs = copy(polytope.convex_subpolytopes)
    elif isinstance(polytope, ConvexPolytopeData):
        paragraphs = [polytope]
    else:
        raise TypeError(f"{type(polytope)} is not polytope-like.")

    shuffle(paragraphs)
    for convex_subpolytope in paragraphs:
        sentences = convex_subpolytope.inequalities + \
                    convex_subpolytope.equalities
        if len(sentences) == 0:
            continue
        dimension = len(sentences[0]) - 1
        shuffle(sentences)
        for inequalities in combinations(sentences, dimension):
            A = np.array([x[1:] for x in inequalities])
            b = np.array([x[0] for x in inequalities])
            try:
                vertex = np.linalg.inv(-A) @ b
                if polytope_has_element(convex_subpolytope, vertex):
                    return vertex
            except np.linalg.LinAlgError:
                pass

    raise NoFeasibleSolutions()


def nearest_point_plane(point, plane):
    """
    Computes the point nearest `point` on an affine `plane` in R^3, specified as
    the quadruple of coefficients in

        a0 + a1 * x1 + a2 * x2 + a3 * x3 == 0 .

    Raises NoFeasibleSolutions if the plane is degenerate.
    """
    b, n1, n2, n3 = plane
    p1, p2, p3 = point
    nn = (n1 * n1 + n2 * n2 + n3 * n3)
    if nn < epsilon:
        raise NoFeasibleSolutions()
    k = (p1 * n1 + p2 * n2 + p3 * n3 + b) / nn
    return p1 - k * n1, p2 - k * n2, p3 - k * n3


def nearest_point_line(point, plane1, plane2):
    """
    Computes the point nearest `point` on an affine line in R^3, specified as
    the intersection of two planes, each themselves specified as the quadruple
    of coefficients in

        a0 + a1 * x1 + a2 * x2 + a3 * x3 == 0 .

    Raises NoFeasibleSolutions if the planes or their arrangement is degenerate.
    """
    # pivot plane2 about the intersection of plane1 and plane2, so that they
    # become orthogonal.
    n1n1 = plane1[1] ** 2 + plane1[2] ** 2 + plane1[3] ** 2
    if n1n1 < epsilon:
        raise NoFeasibleSolutions()
    n1n2 = plane1[1] * plane2[1] + plane1[2] * plane2[2] + plane1[3] * plane2[3]
    plane2 = (
        plane2[0] - plane1[0] * n1n2 / n1n1,
        plane2[1] - plane1[1] * n1n2 / n1n1,
        plane2[2] - plane1[2] * n1n2 / n1n1,
        plane2[3] - plane1[3] * n1n2 / n1n1,
    )
    point1 = nearest_point_plane(point, plane1)
    point2 = nearest_point_plane(point1, plane2)
    return point2


def point_from_implicit(plane1, plane2, plane3):
    """
    Computes the intersection of a nondegenerate triple of nondegenerate affine
    planes in R^3.

    Raises NoFeasibleSolutions if the planes or their arrangement is degenerate.
    """
    try:
        A = np.array([plane1[1:], plane2[1:], plane3[1:]])
        b = np.array([-plane1[0], -plane2[0], -plane3[0]])
        return np.linalg.inv(A) @ b
    except np.linalg.LinAlgError:
        raise NoFeasibleSolutions()


def point_point_distance(point1, point2):
    """
    Computes the Euclidean distance between two points in R^3.
    """
    return math.sqrt(
        (point1[0] - point2[0]) ** 2 +
        (point1[1] - point2[1]) ** 2 +
        (point1[2] - point2[2]) ** 2
    )


def nearest_point_polyhedron(point, polytope):
    """
    Computes the nearest point, specified as a triple, to a polyhedron,
    specified as a `PolytopeData` instance.

    Raises NoFeasibleSolutions if the polytope is empty.
    """
    if polyhedron_has_element(polytope, point):
        return point

    candidates = []

    # iterate over faces
    for cp in polytope.convex_subpolytopes:
        for plane in cp.inequalities + cp.equalities:
            try:
                candidate = nearest_point_plane(point, plane)
                if polyhedron_has_element(polytope, candidate):
                    candidates.append(
                        (candidate, point_point_distance(candidate, point))
                    )
            except NoFeasibleSolutions:
                pass

    # iterate over lines
    for cp in polytope.convex_subpolytopes:
        for (plane1, plane2) in combinations(cp.inequalities + cp.equalities, 2):
            try:
                candidate = nearest_point_line(point, plane1, plane2)
                if polyhedron_has_element(polytope, candidate):
                    candidates.append(
                        (candidate, point_point_distance(candidate, point))
                    )
            except NoFeasibleSolutions:
                pass

    # iterate over vertices
    for cp in polytope.convex_subpolytopes:
        # TODO: this could be computed once and for all.
        for (plane1, plane2, plane3) in combinations(
                cp.inequalities + cp.equalities, 3
        ):
            try:
                candidate = point_from_implicit(plane1, plane2, plane3)
                if polyhedron_has_element(polytope, candidate):
                    candidates.append(
                        (candidate, point_point_distance(candidate, point))
                    )
            except NoFeasibleSolutions:
                pass

    if 0 == len(candidates):
        raise NoFeasibleSolutions()

    return sorted(candidates, key=lambda x: x[1])[0][0]
