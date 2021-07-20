"""

"""

from dataclasses import dataclass
from typing import List


@dataclass
class OperationPolytopeData(CircuitPolytopeData):
    """
    A polytope which describes a single gate, together with a precomputed
    QISKit circuit expressing its canonical form in native operations.

    For example, the native operation sqrtCX on a device would be encoded as an
    OperationPolytope with the same canonical coordinates as 1/2 XX, and with a
    `canonical_circuit` slot containing

        H 1 ; sqrtCX ; H 1

    which expresses 1/2 XX in terms of this native multiqubit interaction.
    """
    canonical_circuit: qiskit.QuantumCircuit


@dataclass
class OperationPolytope(OperationPolytopeData, CircuitPolytope):
    """
    See OperationPolytopeData.
    """
    pass


def decomposition_hop(
        coverage_set: List[CircuitPolytope],
        operations: List[OperationPolytope],
        container: Polytope,
        target_polytope: Polytope
):
    """
    Using a fixed `coverage_set` and `operations`, takes a `target_polytope`
    describing some canonical gates to be modeled within `container`, then finds
    a lower-cost member of the coverage set and a preimage for the target within
    it.

    Returns a tuple: (
        preimage canonical point,
        operation name,
        target canonical point,
        coverage polytope to which the preimage belongs
    )
    """
    ancestor_polytope, operation_polytope = None, None

    # otherwise, find the ancestor and edge for this polytope.
    for polytope in operations:
        if polytope.operations[0] == container.operations[-1]:
            operation_polytope = polytope
            break
    for polytope in coverage_set:
        if polytope.operations == container.operations[:-1]:
            ancestor_polytope = polytope
            break

    if ancestor_polytope is None or operation_polytope is None:
        raise ValueError("Unable to find ancestor / operation polytope.")

    # calculate the intersection of qlr + (ancestor, operation, target),
    # then project to the first tuple.
    # NOTE: the extra condition is to force compatibility with
    #       `decompose_xxyy_into_xxyy_xx`, but it isn't necessary in general.
    #       in fact, it's also not sufficient: we may have to retry this
    #       this decomposition step if that routine fails later on.
    backsolution_polytope = intersect_and_project(
        target="a",
        a_polytope=ancestor_polytope,
        b_polytope=operation_polytope,
        c_polytope=target_polytope,
        extra_polytope=Polytope(convex_subpolytopes=[
            # equate first source and first target coordinates
            ConvexPolytope(inequalities=[
                [0, 1, 1, 0, 0, 0, 0, -1, -1, 0],
                [0, -1, -1, 0, 0, 0, 0, 1, 1, 0],
            ]),
            # equate first source and second target coordinates
            ConvexPolytope(inequalities=[
                [0, 1, 1, 0, 0, 0, 0, -1, 0, -1],
                [0, -1, -1, 0, 0, 0, 0, 1, 0, 1],
            ]),
            # equate first source and third target coordinates
            ConvexPolytope(inequalities=[
                [0, 1, 1, 0, 0, 0, 0, 0, -1, -1],
                [0, -1, -1, 0, 0, 0, 0, 0, 1, 1],
            ]),
            # equate second source and second target coordinates
            ConvexPolytope(inequalities=[
                [0, 1, 0, 1, 0, 0, 0, -1, 0, -1],
                [0, -1, 0, -1, 0, 0, 0, 1, 0, 1],
            ]),
            # equate second source and third target coordinates
            ConvexPolytope(inequalities=[
                [0, 1, 0, 1, 0, 0, 0, 0, -1, -1],
                [0, -1, 0, -1, 0, 0, 0, 0, 1, 1],
            ]),
            # equate third source and third target coordinates
            ConvexPolytope(inequalities=[
                [0, 0, 1, 1, 0, 0, 0, 0, -1, -1],
                [0, 0, -1, -1, 0, 0, 0, 0, 1, 1],
            ]),
        ])
    )

    # pick any nonzero point in the backsolution polytope,
    # then recurse on that point and the ancestor polytope
    all_vertices = []
    for convex_polytope in backsolution_polytope.convex_subpolytopes:
        all_vertices += convex_polytope.vertices
    if 0 != len(all_vertices):
        return (
            # TODO: THIS IS A STOPGAP MEASURE!!!
            sample(all_vertices, 1)[0],
            operation_polytope.operations[0],
            target_polytope.convex_subpolytopes[0].vertices[0],
            ancestor_polytope
        )
    else:
        raise ValueError("Empty backsolution polytope.")


def decomposition_hops(
        coverage_set: List[CircuitPolytope],
        operations: List[OperationPolytope],
        target_polytope: Polytope
):
    """
    Fixing a `coverage_set` and a set of `operations`, finds a minimal
    decomposition for a canonical interaction in `target_polytope` into a
    sequence of operations drawn from `operations`, together with specific
    intermediate canonical points linked by them.

    Returns a list of tuples of shape (source vertex, operation, target vertex),
    so that each target vertex is accessible from its source vertex by
    application of the operation, each target vertex matches its next source
    vertex, the original source vertex corresponds to the identity, and the
    last target lies in `target_polytope`.
    """
    decomposition = []

    working_polytope = cheapest_container(coverage_set, target_polytope)

    if working_polytope is None:
        raise ValueError(f"{target_polytope} not contained in coverage set.")

    # if this polytope corresponds to the empty operation, we're done.
    while 0 != len(working_polytope.operations):
        source_vertex, operation, target_vertex, working_polytope = \
            decomposition_hop(
                coverage_set, operations, working_polytope, target_polytope
            )

        # a/k/a decomposition.push
        decomposition.insert(0, (source_vertex, operation, target_vertex))
        target_polytope = exactly(*source_vertex)

    return decomposition
