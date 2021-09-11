"""
monodromy/xx_decompose/scipy.py

Utilities for interacting with polytope data through scipy / by hand.
"""

from copy import copy
from itertools import combinations

import numpy as np

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


def manual_get_vertex(polytope):
    """
    Returns a single random vertex from `polytope`.

    Same as `manual_get_random_vertex`, but not random.
    """
    if isinstance(polytope, PolytopeData):
        paragraphs = copy(polytope.convex_subpolytopes)
    elif isinstance(polytope, ConvexPolytopeData):
        paragraphs = [polytope]
    else:
        raise TypeError(f"{type(polytope)} is not polytope-like.")

    for convex_subpolytope in paragraphs:
        sentences = convex_subpolytope.inequalities + \
                    convex_subpolytope.equalities
        if len(sentences) == 0:
            continue
        dimension = len(sentences[0]) - 1
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
