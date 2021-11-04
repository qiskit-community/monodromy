"""
monodromy/haar.py

Routines for calculating the Haar volume of polytopes expressed in positive
canonical coordinates.
"""

from math import sqrt
from typing import List

import numpy as np

from .coordinates import monodromy_to_positive_canonical_polytope
from .io.base import CircuitPolytopeData
from .polynomials import Polynomial, TrigPolynomial
from .polytopes import alternating_sum, make_convex_polytope
from .static.examples import empty_polytope
from .utilities import epsilon


# duck typing means poor dispatching...
def _haar_volume_tetrahedron(tetrahedron, integrand=None):
    """
    Integrates the PU(4) Haar form over a 3D tetrahedron expressed in positive
    canonical coordinates, normalized so that CX lies at (pi/2, 0, 0).  (NOTE:
    This differs from almost everywhere else in this codebase by a factor of 2!)

    See Watts, O'Connor, and Vala's _Metric Structure of the Space of Two-Qubit
    Gates, Perfect Entanglers and Quantum Control_, Equation (27), which we've
    rewritten one step further to remove all products.

    Takes an optional `integrand`, a polynomial expressed in (c1, c2, c3), to
    integrate against the Haar measure.
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

    # transform integrand into tetrahedral coordinates
    if integrand is None:
        transformed_integrand = Polynomial.from_linear_list([1])
    else:
        transformed_integrand = Polynomial()
        for power_tuple, coefficient in integrand.coefficient_table.items():
            summand = Polynomial.from_linear_list([1])
            summand = summand * coefficient
            for _ in [] if len(power_tuple) < 1 else range(power_tuple[0]):
                summand = summand * c1
            for _ in [] if len(power_tuple) < 2 else range(power_tuple[1]):
                summand = summand * c2
            for _ in [] if len(power_tuple) < 3 else range(power_tuple[2]):
                summand = summand * c3
            transformed_integrand = transformed_integrand + summand

    haar_form = []
    for left, right in [(c1, c2), (c2, c3), (c3, c1)]:
        haar_form += [
            TrigPolynomial(
                trig_fn="cos",
                coefficients=transformed_integrand,
                arguments=(left * 2 - right * 4)
            ),
            TrigPolynomial(
                trig_fn="cos",
                coefficients=transformed_integrand,
                arguments=(left * 2 + right * 4)
            ),
            TrigPolynomial(
                trig_fn="cos",
                coefficients=transformed_integrand * -1,
                arguments=(left * 4 - right * 2)
            ),
            TrigPolynomial(
                trig_fn="cos",
                coefficients=transformed_integrand * -1,
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


def _haar_volume_convex_polytope(convex_polytope, integrand=None):
    """
    Integrates the PU(4) Haar form, expressed in positive canonical coordinates,
    over a 3D convex polytope.

    Takes an optional `integrand`, a polynomial expressed in (c1, c2, c3), to
    integrate against the Haar measure.
    """

    vertices = [[np.pi * x for x in vertex]
                for vertex in convex_polytope.vertices]
    mapped_tetrahedra = list([list(vertices[index] for index in tetrahedron)
                              for tetrahedron in convex_polytope.triangulation])
    return sum([_haar_volume_tetrahedron(tetrahedron, integrand=integrand)
                for tetrahedron in mapped_tetrahedra])


def haar_volume(polytope, integrand=None):
    """
    Integrates the PU(4) Haar form, expressed in positive canonical coordinates,
    over a 3D polytope.

    Takes an optional `integrand`, a polynomial expressed in (c1, c2, c3), to
    integrate against the Haar measure.
    """
    def volume_fn(convex_polytope):
        return _haar_volume_convex_polytope(convex_polytope, integrand=integrand)

    return alternating_sum(polytope, volume_fn)


def distance_polynomial_integrals(
        coverage_set: List[CircuitPolytopeData],
        max_degree=0,
        chatty=False
):
    """
    Computes the integrals of dist^n dHaar over the "fresh" part of each member
    of `coverage_set` for exponent n in the range [0, max_degree].  Returns a
    dictionary mapping operations tuples from the `coverage_set` to a list of
    calculated integration values.
    """
    # the nth moment is given by integrating the nth power of
    #  min { l_1(-, (0, 0, 0)), l_1(-, (pi, 0, 0)) },
    # which we split into the two integrals ("positive" and "negative")
    # depending on which min argument actually represents the minimum.
    positive_halfspace = make_convex_polytope(
        [[1, -2, 0, 0]], name="positive halfspace"
    )
    negative_halfspace = make_convex_polytope(
        [[-1, 2, 0, 0]], name="negative halfspace"
    )

    positive_polytopes_so_far = empty_polytope
    negative_polytopes_so_far = empty_polytope

    polynomial_averages = dict()

    if chatty:
        for degree in range(1 + max_degree):
            print(f" deg {degree}\t | ", end="")
        print("Sequence name")

    for polytope in coverage_set:
        polytope = monodromy_to_positive_canonical_polytope(polytope)
        positive_polytope = polytope.intersect(positive_halfspace).reduce()
        negative_polytope = polytope.intersect(negative_halfspace).reduce()
        positive_complementary_polytope = positive_polytopes_so_far \
            .intersect(positive_polytope).reduce()
        negative_complementary_polytope = negative_polytopes_so_far \
            .intersect(negative_polytope).reduce()

        # could reuse these, but probably as cheap to recreate them
        positive_polynomial_form = Polynomial.from_linear_list([1])
        negative_polynomial_form = Polynomial.from_linear_list([1])
        polynomial_averages[tuple(polytope.operations)] = []
        for degree in range(1 + max_degree):
            integral = (
                haar_volume(positive_polytope, positive_polynomial_form)
                + haar_volume(negative_polytope, negative_polynomial_form)
                - haar_volume(positive_complementary_polytope, positive_polynomial_form)
                - haar_volume(negative_complementary_polytope, negative_polynomial_form)
            )
            polynomial_averages[tuple(polytope.operations)].append(integral)

            if chatty:
                print(f"{integral:5.5f}\t | ", end="")

            # update the polynomial forms
            positive_polynomial_form = positive_polynomial_form * \
                Polynomial.from_linear_list([0, 1, 1, 1])
            negative_polynomial_form = negative_polynomial_form * \
                Polynomial.from_linear_list([np.pi, -1, 1, 1])

        if chatty:
            print(f"{'.'.join(polytope.operations)}")

        positive_polytopes_so_far = positive_polytopes_so_far.union(positive_polytope).reduce()
        negative_polytopes_so_far = negative_polytopes_so_far.union(negative_polytope).reduce()

    return polynomial_averages


def expected_cost(coverage_set, chatty=False):
    """
    Calculates the expected cost, using the Haar measure, of a `coverage_set`
    expressed in monodromy coordinates.
    """

    integrals = distance_polynomial_integrals(coverage_set, chatty=chatty)
    expected_cost = 0

    for polytope in coverage_set:
        expected_cost += polytope.cost * integrals[tuple(polytope.operations)][0]

    return expected_cost


def cost_statistics(coverage_set, offset, scale_factor, chatty=False):
    """
    Calculates a variety of summary statistics involving the expected cost, as
    in the Haar measure, of a `coverage_set` expressed in monodromy coordinates.

    Assumes an affine-linear cost model for operation infidelity in interaction
    strength (cf. `optimize.py`):

        cost = (# interactions) * offset + (interaction total) * scale_factor .
    """

    polynomial_integrals = distance_polynomial_integrals(
        coverage_set, max_degree=2, chatty=chatty,
    )

    average_cost = 0
    square_sigma_cost = 0
    square_sigma_overshot = 0

    for polytope in coverage_set:
        # these are the integrals of dist^0, dist^1, and dist^2 over the
        # subregion of p which is cost-minimal.
        d0, d1, d2 = polynomial_integrals[tuple(polytope.operations)]

        # if this region is negligible, neglect it.
        if abs(d0) < epsilon:
            continue

        average_cost += d0 * polytope.cost

        square_sigma_cost += d0 * polytope.cost ** 2

        # sigma_overshot**2 = int (overshot - average overshot)**2 dHaar. expand
        # overshot as a difference, then expand that in powers of the distance
        # functional.
        square_sigma_overshot += d0 * (polytope.cost - 3 * offset) ** 2
        square_sigma_overshot += d1 * (-2 * (polytope.cost - 3 * offset) * scale_factor * 2 / np.pi)
        square_sigma_overshot += d2 * (scale_factor * 2 / np.pi) ** 2

    # postprocessing
    average_overshot = average_cost - (3 * offset + 3 / 2 * scale_factor)

    square_sigma_cost = square_sigma_cost - average_cost ** 2
    square_sigma_overshot = square_sigma_overshot - average_overshot ** 2
    if 0 > square_sigma_cost > -1e-10:
        square_sigma_cost = 0
    if 0 > square_sigma_overshot > -1e-10:
        square_sigma_overshot = 0

    return {
        "average_cost": average_cost,
        "average_overshot": average_overshot,
        "sigma_cost": sqrt(square_sigma_cost),
        "sigma_overshot": sqrt(square_sigma_overshot),
    }


# Here is a `sympy` version that I wish I could use instead of all of
# `polynomials.py`.
#
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
