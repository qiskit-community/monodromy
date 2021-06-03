"""
monodromy/src/circuits.py

Tools for designing circuits which implement a particular canonical gate.

NOTE: This is _not_ known to be an exactly solvable problem in general.  The
      monodromy polytope techniques tell us _when_ a circuit is available, but
      not how to produce it.  Accordingly, this file is more of a grab-bag of
      techniques for specific native gates.
"""

from fractions import Fraction
import math
import numpy as np
from random import randint, sample  # TODO: THE USE OF `sample` IS A STOPGAP!!!
from typing import List

import qiskit
from qiskit.circuit.library import RXGate, RYGate, RZGate
import qiskit.quantum_info

from .coordinates import alcove_to_positive_canonical_coordinate,\
    unitary_to_alcove_coordinate
from .coverage import intersect_and_project, GatePolytope
from .decompose import decompose_xxyy_into_xxyy_xx
from .examples import exactly, fractionify, canonical_matrix
from .polytopes import ConvexPolytope, Polytope
from .utilities import epsilon


class NoBacksolution(Exception):
    """
    Signaled when the circuit backsolver can't find a suitable preimage point.

    Conjectured to be probabilistically meaningless: should be fine to re-run
    the call after catching this error.
    """
    pass


reflection_options = {
    "no reflection":  ([1, 1, 1],    1, []),        # we checked this phase
    "reflect XX, YY": ([-1, -1, 1],  1, [RZGate]),  # we checked this phase
    "reflect XX, ZZ": ([-1, 1, -1],  1, [RYGate]),  # we checked this phase, but only in a pair with Y shift
    "reflect YY, ZZ": ([1, -1, -1], -1, [RXGate]),  # unchecked
}

shift_options = {
    "no shift":    ([0, 0, 0],   1, []),  # we checked this phase
    "Z shift":     ([0, 0, 1],  1j, [RZGate]),  # we checked this phase
    "Y shift":     ([0, 1, 0], -1j, [RYGate]),  # we checked this phase, but only in a pair with reflect XX, ZZ
    "Y,Z shift":   ([0, 1, 1],  -1, [RYGate, RZGate]),
    "X shift":     ([1, 0, 0], -1j, [RXGate]),  # we checked this phase
    "X,Z shift":   ([1, 0, 1],   1, [RXGate, RZGate]),  # we checked this phase
    "X,Y shift":   ([1, 1, 0],  -1, [RXGate, RYGate]),
    "X,Y,Z shift": ([1, 1, 1], -1j, [RXGate, RYGate, RZGate]),
}


def apply_reflection(reflection_name, coordinate, q):
    """
    Given a reflection type and a canonical coordinate, applies the reflection
    and describes a circuit which enacts the reflection + a global phase shift.
    """
    reflection_scalars, reflection_phase_shift, source_reflection_gates = \
        reflection_options[reflection_name]
    reflected_coord = [x * y for x, y in zip(reflection_scalars, coordinate)]
    source_reflection = qiskit.QuantumCircuit(q)
    for gate in source_reflection_gates:
        source_reflection.append(gate(np.pi), [q[0]])

    return reflected_coord, source_reflection, reflection_phase_shift


# TODO: I wonder if the global phase shift can be attached to the circuit...
def apply_shift(shift_name, coordinate, q):
    """
    Given a shift type and a canonical coordinate, applies the shift and
    describes a circuit which enacts the shift + a global phase shift.
    """
    shift_scalars, shift_phase_shift, source_shift_gates = \
        shift_options[shift_name]
    shifted_coord = [np.pi / 2 * x + y for x, y in zip(shift_scalars, coordinate)]

    source_shift = qiskit.QuantumCircuit(q)
    for gate in source_shift_gates:
        source_shift.append(gate(np.pi), [q[0]])
        source_shift.append(gate(np.pi), [q[1]])

    return shifted_coord, source_shift, shift_phase_shift


def nearp(x, y, modulus=np.pi/2, epsilon=1e-2):
    """
    Checks whether two points are near each other, accounting for float jitter
    and wraparound.
    """
    return abs(np.mod(abs(x - y), modulus)) < epsilon or \
           abs(np.mod(abs(x - y), modulus) - modulus) < epsilon


def l1_distance(x, y):
    """
    Computes the l_1 / Manhattan distance between two coordinates.
    """
    return sum([abs(xx - yy) for xx, yy in zip(x, y)])


# NOTE: if `point_polytope` were an actual point, you could use .has_element .
def cheapest_container(coverage_set, point_polytope):
    """
    Finds the least costly coverage polytope in `coverage_set` which intersects
    `point_polytope` nontrivially.
    """
    best_cost = float("inf")
    working_polytope = None

    for polytope in coverage_set:
        if polytope.cost < best_cost and polytope.contains(point_polytope):
            working_polytope = polytope
            best_cost = polytope.cost

    return working_polytope


def decomposition_hop(
        coverage_set: List[GatePolytope],
        operations: List[GatePolytope],
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
            # fix CAN(a, *, *)
            ConvexPolytope(inequalities=fractionify([
                [0,  1,  1, 0, 0, 0, 0, -1, -1, 0],
                [0, -1, -1, 0, 0, 0, 0,  1,  1, 0],
            ])),
            # fix CAN(*, b, *)
            ConvexPolytope(inequalities=fractionify([
                [0,  1, 0,  1, 0, 0, 0, -1, 0, -1],
                [0, -1, 0, -1, 0, 0, 0,  1, 0,  1],
            ])),
            # fix CAN(*, *, c)
            ConvexPolytope(inequalities=fractionify([
                [0, 0,  1,  1, 0, 0, 0, 0, -1, -1],
                [0, 0, -1, -1, 0, 0, 0, 0,  1,  1],
            ])),
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
        coverage_set: List[GatePolytope],
        operations: List[GatePolytope],
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
        source_vertex, operation, target_vertex, working_polytope = decomposition_hop(
            coverage_set, operations, working_polytope, target_polytope
        )

        # a/k/a decomposition.push
        decomposition.insert(0, (source_vertex, operation, target_vertex))
        target_polytope = exactly(*source_vertex)

    return decomposition


def canonical_rotation_circuit(first_index, second_index, q):
    """
    Given a pair of distinct indices 0 ≤ (first_index, second_index) ≤ 2,
    produces a two-qubit circuit (on qubits `q`) which rotates a canonical gate
    a0 XX + a1 YY + a2 ZZ into a[first] XX + a[second] YY + a[other] ZZ.
    """
    conj = qiskit.QuantumCircuit(q)

    if (0, 1) == (first_index, second_index):
        pass  # no need to do anything
    elif (0, 2) == (first_index, second_index):
        conj.rx(-np.pi / 2, q[0])
        conj.rx(np.pi / 2, q[1])
    elif (1, 0) == (first_index, second_index):
        conj.rz(-np.pi / 2, q[0])
        conj.rz(-np.pi / 2, q[1])
    elif (1, 2) == (first_index, second_index):
        conj.rz(np.pi / 2, q[0])
        conj.rz(np.pi / 2, q[1])
        conj.ry(np.pi / 2, q[0])
        conj.ry(-np.pi / 2, q[1])
    elif (2, 0) == (first_index, second_index):
        conj.rz(np.pi / 2, q[0])
        conj.rz(np.pi / 2, q[1])
        conj.rx(np.pi / 2, q[0])
        conj.rx(-np.pi / 2, q[1])
    elif (2, 1) == (first_index, second_index):
        conj.ry(np.pi / 2, q[0])
        conj.ry(-np.pi / 2, q[1])

    return conj


def xx_circuit_from_decomposition(
        decomposition,
        operations: List[GatePolytope]
) -> qiskit.QuantumCircuit:
    """
    Extracts a circuit, with interactions drawn from `operations`, based on a
    decomposition produced by `decomposition_hops`.

    Returns a QISKit circuit modeling the decomposed interaction.
    """
    canonical_coordinate_table = {
        operation.operations[0]: alcove_to_positive_canonical_coordinate(
            *operation.convex_subpolytopes[0].vertices[0])
        for operation in operations
    }

    # make sure that all the operations are of XX-interaction type
    assert all([abs(c[1]) < 0.01 and abs(c[2]) < 0.01
                for c in canonical_coordinate_table.values()])

    canonical_gate_table = {
        k: qiskit.extensions.UnitaryGate(
            canonical_matrix(v[0]),
            label=f"XX({k})"
        )
        for k, v in canonical_coordinate_table.items()
    }

    # TODO: is this proper form?
    q = qiskit.QuantumRegister(2)
    qc = qiskit.QuantumCircuit(q)

    # empty decompositions are easy!
    if 0 == len(decomposition):
        return qc

    # the first canonical gate is easy!
    qc.append(canonical_gate_table[decomposition[0][1]], q)

    global_phase = 1 + 0j

    # from here, we have to work.
    for decomposition_depth, (source_alcove_coord, operation, target_alcove_coord) \
            in enumerate(decomposition[1:]):
        source_canonical_coord = alcove_to_positive_canonical_coordinate(*source_alcove_coord)
        target_canonical_coord = alcove_to_positive_canonical_coordinate(*target_alcove_coord)

        permute_source_for_overlap, permute_target_for_overlap = None, None

        # apply all possible reflections, shifts to the source
        for source_reflection_name in reflection_options.keys():
            reflected_source_coord, source_reflection, reflection_phase_shift = \
                apply_reflection(source_reflection_name, source_canonical_coord, q)
            for source_shift_name in shift_options.keys():
                shifted_source_coord, source_shift, shift_phase_shift = \
                    apply_shift(source_shift_name, reflected_source_coord, q)

                # check for overlap, back out permutation
                source_shared, target_shared = None, None
                for i, j in [(0, 0), (0, 1), (0, 2),
                             (1, 0), (1, 1), (1, 2),
                             (2, 0), (2, 1), (2, 2)]:
                    if nearp(shifted_source_coord[i], target_canonical_coord[j],
                             modulus=np.pi):
                        source_shared, target_shared = i, j
                        break
                if source_shared is None:
                    continue

                # pick out the other coordinates
                source_first, source_second = [x for x in [0, 1, 2] if
                                               x != source_shared]
                target_first, target_second = [x for x in [0, 1, 2] if
                                               x != target_shared]

                # check for arccos validity
                r, s, u, v, x, y = decompose_xxyy_into_xxyy_xx(
                    float(target_canonical_coord[target_first]),
                    float(target_canonical_coord[target_second]),
                    float(shifted_source_coord[source_first]),
                    float(shifted_source_coord[source_second]),
                    float(canonical_coordinate_table[operation][0]),
                )
                if any([math.isnan(val) for val in (r, s, u, v, x, y)]):
                    continue

                # OK: this combination of things works.
                # save the permutation which rotates the shared coordinate into ZZ.
                permute_source_for_overlap = canonical_rotation_circuit(
                    source_first, source_second, q)
                permute_target_for_overlap = canonical_rotation_circuit(
                    target_first, target_second, q)
                break

            if permute_source_for_overlap is not None:
                break

        if permute_source_for_overlap is None:
            raise NoBacksolution()

        # target^p_t_f_o = rs * (source^s_reflection * s_shift)^p_s_f_o * uv * operation * xy
        # start with source conjugated by source_reflection, shifted by source_shift, conjugated by p_s_f_o
        output_circuit = qiskit.QuantumCircuit(q)
        output_circuit += permute_source_for_overlap
        output_circuit += source_reflection
        output_circuit.compose(qc, inplace=True)
        output_circuit += source_reflection.inverse()
        output_circuit += source_shift
        output_circuit += permute_source_for_overlap.inverse()
        qc = output_circuit
        global_phase *= reflection_phase_shift * shift_phase_shift

        # target^p_t_f_o = rs * qc * uv * operation * xy
        # install the local Z rolls
        output_circuit = qiskit.QuantumCircuit(q)
        output_circuit.rz(2 * x, q[0])
        output_circuit.rz(2 * y, q[1])
        output_circuit.append(canonical_gate_table[operation], q)
        output_circuit.rz(2 * u, q[0])
        output_circuit.rz(2 * v, q[1])
        output_circuit.compose(qc, inplace=True)
        output_circuit.rz(2 * r, q[0])
        output_circuit.rz(2 * s, q[1])
        qc = output_circuit

        # target = qc^p_t_f_o*
        # finally, conjugate by the (inverse) target permutation
        output_circuit = qiskit.QuantumCircuit(q)
        output_circuit += permute_target_for_overlap.inverse()
        output_circuit.compose(qc, inplace=True)
        output_circuit += permute_target_for_overlap
        qc = output_circuit

    qc.global_phase = -np.log(global_phase).imag
    return qc


#
# some other routines for random circuit generation
#


# TODO: In rare cases this generates an empty range, and I'm not sure why.
# TODO: This doesn't sample uniformly; it treats the pushed-forward uniform
#       distribution along a projection as uniform, which is false.
def random_alcove_coordinate(denominator=100):
    first_numerator = randint(0, denominator // 2)
    second_numerator = randint(
        max(-first_numerator // 3, -(denominator - 2 * first_numerator) // 2),
        min(first_numerator, (3 * denominator - 6 * first_numerator) // 2)
    )
    third_numerator = randint(
        max(-(first_numerator + second_numerator) // 2 + 1,
            -(denominator - 2 * first_numerator) // 2),
        min(second_numerator + 1,
            denominator - 2 * first_numerator - second_numerator)
    )

    return (
        Fraction(first_numerator, denominator),
        Fraction(second_numerator, denominator),
        Fraction(third_numerator, denominator),
    )


def sample_irreducible_circuit(coverage_set, operations, target_gate_polytope):
    """
    Produces a randomly generated circuit of the prescribed type which cannot
    be rewritten into a circuit of lower cost.
    """

    operation_gates = {
        operation.operations[0]:
            qiskit.extensions.UnitaryGate(
                canonical_matrix(*alcove_to_positive_canonical_coordinate(
                    *operation.convex_subpolytopes[0].vertices[0])),
                label=f"CAN({operation.operations[0]})"
            )
        for operation in operations
    }

    q = qiskit.QuantumRegister(2, 'q')

    while True:
        qc = qiskit.QuantumCircuit(q)

        qc.u3(qubit=0,
              theta=np.random.uniform(2 * np.pi),
              phi=np.random.uniform(2 * np.pi),
              lam=np.random.uniform(2 * np.pi))
        qc.u3(qubit=1,
              theta=np.random.uniform(2 * np.pi),
              phi=np.random.uniform(2 * np.pi),
              lam=np.random.uniform(2 * np.pi))
        for operation in target_gate_polytope.operations:
            qc.append(operation_gates[operation], q)
            qc.u3(qubit=0,
                  theta=np.random.uniform(2 * np.pi),
                  phi=np.random.uniform(2 * np.pi),
                  lam=np.random.uniform(2 * np.pi))
            qc.u3(qubit=1,
                  theta=np.random.uniform(2 * np.pi),
                  phi=np.random.uniform(2 * np.pi),
                  lam=np.random.uniform(2 * np.pi))

        point = unitary_to_alcove_coordinate(
            qiskit.quantum_info.Operator(qc).data
        )
        computed_depth = cheapest_container(
            coverage_set, exactly(*point[:-1])
        ).cost

        if computed_depth >= target_gate_polytope.cost:
            break

    return qc
