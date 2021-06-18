"""
monodromy/xx_decompose/paths.py

Routines for producing right-angled paths through the Weyl alcove.  Consider a
set of native interactions with an associated minimal covering set of minimum-
cost circuit polytopes, as well as a target coordinate.  The coverage set
associates to the target coordinate a circuit type C = (O1 ... On) consisting of
a sequence of native interactions Oj.  A _path_ is a sequence (I P1 ... Pn) of
intermediate Weyl points, where Pj is accessible from P(j-1) by Oj.  A path is
said to be _right-angled_ when at each step one coordinate is fixed (up to
possible Weyl reflection) when expressed in canonical coordinates.

Conjecturally, a right-angled path can be realized by local gates whose tensor
components are of the form

    (quarter rotations) * (a IZ + b ZI) * (quarter rotations),

which we assume in `circuits.py`.

NOTE: The routines in this file can fail for numerical reasons, and so they are
      lightly randomized and meant to be repeatedly called.
"""

from collections import Counter

from ..exceptions import NoBacksolution, NoFeasibleSolutions
from ..io.base import ConvexPolytopeData, PolytopeData
from .scipy import manual_get_random_vertex


def single_unordered_decomposition_hop(
    target, working_operations, scipy_coverage_set
):
    """
    Produces a single inverse step in a right-angled path.  The target of the
    step is `target`, expressed in monodromy coordinates, and which belongs to
    to the circuit type consisting of XX-type operations enumerated in
    `working_operations`.  The step is taken along one such operation in
    `working_operations`, and the source of the step belongs

    Returns a dictionary keyed on "hop", "ancestor", and "operations_remaining",
    which respectively are: a triple (source, operation, target) describing the
    single step; the source coordinate of the step; and the remaining set of
    operations yet to be stripped off.

    NOTE: Operates with the assumption that gates within the circuit
          decomposition may be freely permuted.
    """
    backsolution_polytope = PolytopeData(convex_subpolytopes=[])
    for ancestor in scipy_coverage_set:
        # check that this is actually an ancestor
        if Counter(ancestor.operations) != Counter(working_operations):
            continue

        # impose the target constraints, which sit on "b"
        # (really on "c", but "b" has already been projected off)
        backsolution_polytope.convex_subpolytopes += [
            ConvexPolytopeData(
                inequalities=[
                    [ineq[0] + sum(x * y for x, y in zip(ineq[4:], target)),
                     ineq[1], ineq[2], ineq[3]]
                    for ineq in cp.inequalities
                ],
                equalities=[
                    [eq[0] + sum(x * y for x, y in zip(eq[4:], target)),
                     eq[1], eq[2], eq[3]]
                    for eq in cp.equalities
                ],
            )
            for cp in ancestor.convex_subpolytopes
        ]

        # walk over the convex backsolution subpolytopes, try to find one
        # that's solvable
        try:
            solution = manual_get_random_vertex(backsolution_polytope)

            return {
                "hop": (solution, ancestor.operations[-1], target),
                "ancestor": solution,
                "operations_remaining": ancestor.operations[:-1]
            }
        except NoFeasibleSolutions:
            pass

    raise NoBacksolution()
