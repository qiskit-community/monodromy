"""
monodromy/polynomials.py

Calculus with symbolic (trigonometric) polynomials.

NOTE: My strong preference would have been to use something like sympy, but I
      couldn't make it performant.  :(  See the bottom of haar.py for the kind
      of thing I need to be able to do.
"""

from collections import defaultdict
from copy import copy
from dataclasses import dataclass, field
from itertools import zip_longest
import math
from numbers import Real
from typing import Dict, Tuple

from .utilities import epsilon


def canonicalize_power_tuple(power_tuple):
    """
    Entries in `Polynomial` are stored as (exponent tuple): coefficient, where
    the exponent tuple is required not to have any trailing zeroes.  This takes
    a tuple and rewrites it into that form.
    """
    while len(power_tuple) > 0 and power_tuple[-1] == 0:
        power_tuple = power_tuple[:-1]
    return power_tuple


def get_from_power_tuple(power_tuple, index):
    """
    Extracts an exponent from an exponent tuple that may have been canonicalized
    to be shorter than the expected length, by removing extra zeroes.
    """
    if index >= len(power_tuple):
        return 0
    else:
        return power_tuple[index]


@dataclass
class Polynomial:
    """
    Models a (multivariate) polynomial with fractional coefficients.
    """

    coefficient_table: Dict[Tuple, Real] = field(
        default_factory=lambda: defaultdict(lambda: 0)
    )

    @classmethod
    def from_coefficient_list(cls, coefficient_list):
        """
        Converts a list [c0, c1, c2, ..., cn] to the polynomial

             c0 + c1 x0 + c2 x0^2 + ... + cn x0^n.
        """
        polynomial = Polynomial()
        for power, value in enumerate(coefficient_list):
            power_tuple = canonicalize_power_tuple((power,))
            polynomial.coefficient_table[power_tuple] = value
        return polynomial

    @classmethod
    def from_linear_list(cls, linear_list):
        """
        Converts a list [k, d0, d1, ..., dn] to the polynomial

            k + d0 x0 + d1 x1 + ... + dn xn.
        """
        polynomial = Polynomial()
        head, *linear_list = linear_list
        polynomial.coefficient_table[()] = head
        for place, value in enumerate(linear_list):
            key = ((0,) * place) + (1,)
            polynomial.coefficient_table[key] = value
        return polynomial

    def __add__(self, right):
        if not isinstance(right, Polynomial):
            right = Polynomial.from_linear_list([right])

        polynomial = Polynomial(coefficient_table=copy(self.coefficient_table))
        for k, v in right.coefficient_table.items():
            polynomial.coefficient_table[k] += v

        return polynomial

    def __sub__(self, right):
        return self + (right * -1)

    def __mul__(self, right):
        if not isinstance(right, Polynomial):
            right = Polynomial.from_linear_list([right])

        polynomial = Polynomial()
        for k, v in self.coefficient_table.items():
            for kp, vp in right.coefficient_table.items():
                kpp = tuple(x + y for x, y in zip_longest(k, kp, fillvalue=0))
                polynomial.coefficient_table[kpp] += v * vp

        return polynomial

    def __str__(self):
        output = ""
        for k, v in self.coefficient_table.items():
            if v == 0:
                continue
            output += str(v)
            if any([power != 0 for power in k]):
                output += f" * x^{k}"
            output += " + "

        return output[:-3]

    def __eq__(self, other):
        for k, v in self.coefficient_table.items():
            if v != other.coefficient_table[k]:
                return False
        for k, v in other.coefficient_table.items():
            if v != self.coefficient_table[k]:
                return False
        return True

    def evaluate(self, variable, value):
        """
        Replaces the variable indexed at `variable` by the Polynomial `value`.
        """
        value_powers = [1, value]

        evaluated_polynomial = Polynomial()

        for k, v in self.coefficient_table.items():
            power = get_from_power_tuple(k, variable)

            while power >= len(value_powers):  # extend value_powers as needed
                value_powers.append(value_powers[-1] * value)

            monomial = Polynomial()
            suppressed_key = canonicalize_power_tuple(
                k[:variable] + (0,) + k[variable + 1:]
            )
            monomial.coefficient_table[suppressed_key] = v
            evaluated_polynomial += monomial * value_powers[power]

        return evaluated_polynomial

    def indefinite_integral(self, variable):
        """
        Produces the indefinite integral against the variable indexed at
        `variable`.  The result has vanishing constant term.
        """
        integrated_polynomial = Polynomial()

        for k, v in self.coefficient_table.items():
            if variable < len(k):
                shifted_key = k[:variable] + (1 + k[variable],) + k[1 + variable:]
            else:
                shifted_key = k + (0,) * (variable - len(k)) + (1,)
            integrated_polynomial.coefficient_table[shifted_key] = \
                v / (shifted_key[variable])

        return integrated_polynomial

    def definite_integral(self, var, lower, upper):
        """
        Produces the definite integral against the variable indexed at
        `variable`, as integrated from `lower` to `upper`, which may themselves
        be `Polynomial` expressions.
        """
        integrated_polynomial = self.indefinite_integral(var)

        return integrated_polynomial.evaluate(var, upper) - \
               integrated_polynomial.evaluate(var, lower)

    def derivative(self, var):
        """
        Produces the derivative against the variable indexed at `variable`.
        """
        differentiated_polynomial = Polynomial()

        for k, v in self.coefficient_table.items():
            if get_from_power_tuple(k, var) == 0:
                continue

            shifted_key = (*k[:var], -1 + k[var], *k[1 + var:])
            shifted_key = canonicalize_power_tuple(shifted_key)
            differentiated_polynomial.coefficient_table[shifted_key] = k[var] * v

        return differentiated_polynomial

    def to_number(self):
        """
        Extracts from a constant polynomial its literal constant value.
        """
        for k, v in self.coefficient_table.items():
            if k != () and abs(v) > epsilon:
                raise ValueError("Cannot convert a nonconstant to a number.")
        return self.coefficient_table[()]


@dataclass
class TrigPolynomial:
    """
    Models a term of the form (multivar. poly.) * trig_fn(linear poly.),
    as arising in the expression for the Haar measure pushed forward to the
    positive canonical alcove.

    NOTE: This could be extended to handle multiplication of terms, since
          product-to-sum formulas preserve linearity in the argument, but not
          higher-degree arguments, since integrating those requires Fresnel
          integrals.
    """

    coefficients: Polynomial
    arguments: Polynomial
    trig_fn: str = "cos"  # in ["sin", "cos"]

    def __mul__(self, other):
        return TrigPolynomial(
            trig_fn=self.trig_fn,
            arguments=self.arguments,
            coefficients=(self.coefficients * other),
        )

    def __str__(self):
        return f"[{self.coefficients}]{self.trig_fn}({self.arguments})"

    def evaluate(self, variable, value):
        """
        Replaces the variable indexed at `variable` by the (linear) Polynomial
        `value`.
        """
        return TrigPolynomial(
            trig_fn=self.trig_fn,
            arguments=self.arguments.evaluate(variable, value),
            coefficients=self.coefficients.evaluate(variable, value),
        )

    def integrate(self, variable, lower, upper):  # -> List[TrigPolynomial]
        """
        Produces the definite integral against the variable indexed at
        `variable`, as integrated from `lower` to `upper`, which may themselves
        be (linear) `Polynomial` expressions.
        """
        linear_coefficient = sum(
            [v if 1 == get_from_power_tuple(k, variable) else 0
             for k, v in self.arguments.coefficient_table.items()])
        args_constant_p = abs(linear_coefficient) < epsilon
        coeffs_constant_p = all(
            [0 == get_from_power_tuple(k, variable) or abs(v) < epsilon
             for k, v in self.coefficients.coefficient_table.items()])

        # base case 1: a bare polynomial
        if args_constant_p:
            integrated_coefficients = self.coefficients.definite_integral(
                variable, lower, upper
            )
            return [TrigPolynomial(
                trig_fn=self.trig_fn,
                arguments=self.arguments,  # no need to integrate this constant
                coefficients=integrated_coefficients
            )]

        if self.trig_fn == "sin":
            linear_coefficient *= -1
        trig_integral = TrigPolynomial(
            trig_fn="sin" if self.trig_fn == "cos" else "cos",
            arguments=self.arguments,
            coefficients=self.coefficients * (1 / linear_coefficient),
        )

        # base case 2: a bare trig function
        if coeffs_constant_p:
            return [
                trig_integral.evaluate(variable, upper),
                trig_integral.evaluate(variable, lower) * -1
            ]

        # recursive case: integrate by parts
        return [
            trig_integral.evaluate(variable, upper),
            trig_integral.evaluate(variable, lower) * -1,
            *(TrigPolynomial(
                coefficients=trig_integral.coefficients.derivative(variable) \
                             * -1,
                trig_fn=trig_integral.trig_fn,
                arguments=trig_integral.arguments,
            ).integrate(variable, lower, upper))
        ]

    def to_number(self):
        """
        Extracts from a constant trig polynomial its literal constant value.
        """
        if self.trig_fn == "sin":
            return self.coefficients.to_number() * math.sin(
                self.arguments.to_number())
        elif self.trig_fn == "cos":
            return self.coefficients.to_number() * math.cos(
                self.arguments.to_number())
        else:
            raise ValueError("Only sin and cos are supported trig functions.")
