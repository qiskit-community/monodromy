"""
monodromy/haar.py

Routines for calculating the Haar volume of polytopes expressed in positive
canonical coordinates.
"""

import numpy as np

from .coordinates import monodromy_to_positive_canonical_polytope
from .examples import empty_polytope
from .polynomials import Polynomial, TrigPolynomial
from .polytopes import alternating_sum


# duck typing means poor dispatching...
def _haar_volume_tetrahedron(tetrahedron):
    """
    Integrates the PU(4) Haar form, expressed in positive canonical coordinates,
    over a 3D tetrahedron.
    """
    tetrahedron = list([list(x) for x in tetrahedron])

    if len(tetrahedron) != 4 or any([len(x) != 3 for x in tetrahedron]):
        return 0

    ((x0, y0, z0), (x1, y1, z1), (x2, y2, z2), (x3, y3, z3)) = tetrahedron

    determinant = np.linalg.det(np.array([
        [x1 - x0, y1 - y0, z1 - z0],
        [x2 - x0, y2 - y0, z2 - z0],
        [x3 - x0, y3 - y0, z3 - z0],
    ]))

    c1 = Polynomial.from_linear_list([x0, x1 - x0, x2 - x0, x3 - x0])
    c2 = Polynomial.from_linear_list([y0, y1 - y0, y2 - y0, y3 - y0])
    c3 = Polynomial.from_linear_list([z0, z1 - z0, z2 - z0, z3 - z0])

    haar_form = []
    for left, right in [(c1, c2), (c2, c3), (c3, c1)]:
        haar_form += [
            TrigPolynomial(
                trig_fn="cos",
                coefficients=Polynomial.from_linear_list([1]),
                arguments=(left * 2 - right * 4)
            ),
            TrigPolynomial(
                trig_fn="cos",
                coefficients=Polynomial.from_linear_list([1]),
                arguments=(left * 2 + right * 4)
            ),
            TrigPolynomial(
                trig_fn="cos",
                coefficients=Polynomial.from_linear_list([-1]),
                arguments=(left * 4 - right * 2)
            ),
            TrigPolynomial(
                trig_fn="cos",
                coefficients=Polynomial.from_linear_list([-1]),
                arguments=(left * 4 + right * 2)
            ),
        ]

    haar_form = sum([term.integrate(2, Polynomial.from_linear_list([0]),
                                    Polynomial.from_linear_list([1, -1, -1]))
                     for term in haar_form], [])
    haar_form = sum([term.integrate(1, Polynomial.from_linear_list([0]),
                                    Polynomial.from_linear_list([1, -1]))
                     for term in haar_form], [])
    haar_form = sum([term.integrate(0, Polynomial.from_linear_list([0]),
                                    Polynomial.from_linear_list([1]))
                     for term in haar_form], [])

    return abs(determinant) / (2 / 3 * np.pi) * \
           sum(term.to_number() for term in haar_form)


def _haar_volume_convex_polytope(convex_polytope):
    """
    Integrates the PU(4) Haar form, expressed in positive canonical coordinates,
    over a 3D convex polytope.
    """
    vertices = [[2 * np.pi * x for x in vertex]
                for vertex in convex_polytope.vertices]
    mapped_tetrahedra = [(vertices[index] for index in tetrahedron)
                         for tetrahedron in convex_polytope.triangulation]
    return sum([_haar_volume_tetrahedron(tetrahedron)
                for tetrahedron in mapped_tetrahedra])


def haar_volume(polytope):
    """
    Integrates the PU(4) Haar form, expressed in positive canonical coordinates,
    over a 3D polytope.
    """
    return alternating_sum(polytope, _haar_volume_convex_polytope)


def expected_cost(coverage_set, chatty=False):
    """
    Calculates the expected cost, using the Haar measure, of a `coverage_set`
    expressed in monodromy coordinates.
    """
    expected_cost = 0
    polytope_so_far = empty_polytope

    if chatty:
        print("   Total vol\t |   Fresh vol\t | Sequence name")

    for polytope in coverage_set:
        polytope = monodromy_to_positive_canonical_polytope(polytope)
        positive_volume = haar_volume(polytope)
        negative_volume = haar_volume(
            polytope_so_far.intersect(polytope).reduce()
        )
        if chatty:
            print(
                f"{float(100 * positive_volume):8.2f}%"
                f"\t | {float(100 * (positive_volume - negative_volume)):8.2f}%"
                f"\t | {'.'.join(polytope.operations)}")
        expected_cost += polytope.cost * (positive_volume - negative_volume)
        polytope_so_far = polytope_so_far.union(polytope).reduce()

    return expected_cost


# def _haar_volume_tetrahedron(tetrahedron):
#     """
#     Integrates the PU(4) Haar form over a 3D tetrahedron.
#     """
#     import sympy
#
#     ((x0, y0, z0), (x1, y1, z1), (x2, y2, z2), (x3, y3, z3)) = tetrahedron
#     x, y, z = sympy.symbols("x y z")
#
#     c1 = x0 + x * x1 + y * x2 + z * x3
#     c2 = y0 + x * y1 + y * y2 + z * y3
#     c3 = z0 + x * z1 + y * z2 + z * z3
#
#     expr = sum(
#         [sympy.cos(2 * left - 4 * right) + sympy.cos(2 * left + 4 * right)
#          for left, right in [(c1, c2), (c2, c3), (c3, c1)]])
#     expr -= sum(
#         [sympy.cos(4 * left - 2 * right) + sympy.cos(4 * left + 2 * right)
#          for left, right in [(c1, c2), (c2, c3), (c3, c1)]])
#     expr = sympy.integrate(expr, (z, 0, 1 - x - y))
#     expr = sympy.integrate(expr, (y, 0, 1 - x))
#     expr = sympy.integrate(expr, (x, 0, 1))
#     return expr.evalf()
