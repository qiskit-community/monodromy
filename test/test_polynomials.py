"""
test/test_polynomials.py

Tests for monodromy/polynomials.py .
"""

import ddt
import unittest

import numpy as np

from monodromy.polynomials import *

epsilon = 0.001


@ddt.ddt
class TestMonodromyPolynomials(unittest.TestCase):
    """Check various polynomial routines."""

    @ddt.data((Polynomial.from_coefficient_list([1, 2]),
               Polynomial.from_coefficient_list([3, 4, 5]),
               Polynomial.from_coefficient_list([4, 6, 5])),
              (Polynomial.from_linear_list([1, 1, 0, 2]),
               Polynomial.from_linear_list([3, 0, 2, 1]),
               Polynomial.from_linear_list([4, 1, 2, 3])))
    @ddt.unpack
    def test_sums(self, left_addend, right_addend, result):
        """Check that polynomials add."""
        self.assertEqual(left_addend + right_addend, result)

    @ddt.data((Polynomial.from_coefficient_list([1, 2]),
               Polynomial.from_coefficient_list([3, 4, 5]),
               Polynomial.from_coefficient_list([-2, -2, -5])),
              (Polynomial.from_linear_list([1, 1, 0, 2]),
               Polynomial.from_linear_list([3, 0, 2, 1]),
               Polynomial.from_linear_list([-2, 1, -2, 1])))
    @ddt.unpack
    def test_differences(self, minuend, subtrahend, result):
        """Check that polynomials subtract."""
        self.assertEqual(minuend - subtrahend, result)

    @ddt.data((Polynomial.from_coefficient_list([1, 1]),
               Polynomial.from_coefficient_list([1, 1]),
               Polynomial.from_coefficient_list([1, 2, 1])),
              (Polynomial.from_coefficient_list([1, 2, 1]),
               Polynomial.from_coefficient_list([1, 2, 1]),
               Polynomial.from_coefficient_list([1, 4, 6, 4, 1])),)
    @ddt.unpack
    def test_products(self, left_factor, right_factor, result):
        """Check that polynomials multiply."""
        self.assertEqual(left_factor * right_factor, result)

    @ddt.data((Polynomial.from_coefficient_list([1, 2, 1]),
               0, 1,
               Polynomial.from_coefficient_list([4])),
              (Polynomial.from_coefficient_list([1, 2, 1]),
               0, 0,
               Polynomial.from_coefficient_list([1])),
              (Polynomial.from_coefficient_list([1, 2, 1]),
               0, -1,
               Polynomial.from_coefficient_list([0])),
              (Polynomial.from_coefficient_list([1, 2, 1]),
               1, 1,
               Polynomial.from_coefficient_list([1, 2, 1])),
              (Polynomial.from_linear_list([1, 2, 5, 13]),
               0, 1,
               Polynomial.from_linear_list([3, 0, 5, 13])),
              (Polynomial.from_linear_list([1, 2, 5, 13]),
               2, 2,
               Polynomial.from_linear_list([27, 2, 5])),
              (Polynomial.from_linear_list([1, 2, 5, 13]),
               4, 1,
               Polynomial.from_linear_list([1, 2, 5, 13])),
              )
    @ddt.unpack
    def test_evaluations(self, polynomial, variable, value, result):
        """Check that polynomials evaluate."""
        self.assertEqual(polynomial.evaluate(variable, value), result)

    @ddt.data((Polynomial.from_coefficient_list([1, 1, -1]), 0,
               Polynomial.from_coefficient_list([0, 1, 1/2, -1/3])),)
    @ddt.unpack
    def test_indefinite_integrals(self, integrand, variable, result):
        """Check that polynomials have primitives."""
        self.assertEqual(integrand.indefinite_integral(variable), result)

    @ddt.data((Polynomial.from_coefficient_list([1, 2, -3]),
               0, -1, 1,
               Polynomial.from_coefficient_list([0])), )
    @ddt.unpack
    def test_definite_integrals(self, integrand, variable, lower, upper, result):
        """Check that polynomials have areas."""
        self.assertEqual(integrand.definite_integral(variable, lower, upper), result)

    @ddt.data((Polynomial.from_coefficient_list([1, 2, -3]), 0,
               Polynomial.from_coefficient_list([2, -6])),
              (Polynomial.from_linear_list([1, 2, 3, 4]), 0,
               Polynomial.from_linear_list([2])))
    @ddt.unpack
    def test_derivatives(self, integrand, variable, result):
        """Check that polynomials have derivatives."""
        self.assertEqual(integrand.derivative(variable), result)

    @ddt.data((Polynomial.from_coefficient_list([1, 2, 3]), 0),
              (Polynomial.from_linear_list([1, 2, 3]), 1),
              (Polynomial.from_coefficient_list([1, 2, 3]), 2),
              (Polynomial.from_linear_list([1, 2, 3]), 5))
    @ddt.unpack
    def test_ftc(self, polynomial, variable):
        self.assertEqual(
            polynomial,
            polynomial.indefinite_integral(variable)
                      .derivative(variable)
        )


@ddt.ddt
class TestMonodromyTrigPolynomials(unittest.TestCase):
    """Check various trig polynomial routines."""

    @ddt.data((Polynomial.from_coefficient_list([1, 2, 3]),
               "sin", Polynomial.from_coefficient_list([-np.pi, np.pi]),
               0, 0, 1,
               (12 - 7 * np.pi ** 2) / (np.pi ** 3)),)
    @ddt.unpack
    def test_definite_integrals(
            self, coefficients, trig_fn, arguments,
            variable, lower, upper, result
    ):
        """Test that trig polynomials can be integrated."""
        polynomial = TrigPolynomial(
            coefficients=coefficients,
            arguments=arguments,
            trig_fn=trig_fn,
        )
        primitives = polynomial.integrate(variable, lower, upper)
        self.assertTrue(abs(
            result -
            sum([primitive.to_number() for primitive in primitives])
        ) < epsilon)
