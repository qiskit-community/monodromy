"""
monodromy/src/circuits.py

Tools for designing circuits which implement a particular canonical gate.

NOTE: This is _not_ known to be an exactly solvable problem in general.  The
      monodromy polytope techniques tell us _when_ a circuit is available, but
      not how to produce it.  Accordingly, this file is more of a grab-bag of
      techniques for specific native gates.
"""

import numpy as np
from typing import List

import qiskit
import qiskit.quantum_info

from .coordinates import alcove_to_canonical_coordinate, unitary_to_alcove_coordinate
from .coverage import intersect_and_project, GatePolytope
from .decompose import decompose_xxyy_into_xxyy_xx
from .examples import exactly, canonical_matrix
from .polytopes import Polytope


def l1_distance(x, y):
    """

    """
    return sum([abs(xx - yy) for xx, yy in zip(x, y)])


def cheapest_container(coverage_set, point_polytope):
    """
    Finds the least costly coverage polytope in `coverage_set` which intersects
    `point_polytope` nontrivially.
    """
    best_cost = float("inf")
    working_polytope = None

    for polytope in coverage_set:
        if polytope.cost < best_cost:
            intersected_polytope = polytope.intersect(point_polytope)
            intersected_polytope = intersected_polytope.reduce()
            if 0 != len(intersected_polytope.convex_subpolytopes):
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
    # then project to the first tuple
    backsolution_polytope = intersect_and_project(
        target="a",
        a_polytope=ancestor_polytope,
        b_polytope=operation_polytope,
        c_polytope=target_polytope,
    )

    # pick any nonzero point in the backsolution polytope,
    # then recurse on that point and the ancestor polytope
    for convex_polytope in backsolution_polytope.convex_subpolytopes:
        vertices = convex_polytope.vertices
        if 0 != len(vertices):
            return (
                vertices[0],
                operation_polytope.operations[0],
                target_polytope.convex_subpolytopes[0].vertices[0],
                ancestor_polytope
            )

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
        operation.operations[0]: alcove_to_canonical_coordinate(
            *operation.convex_subpolytopes[0].vertices[0])
        for operation in operations
    }

    # make sure that all the operations are of XX-interaction
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

    # from here, we have to work.
    for input_alcove_coord, operation, output_alcove_coord in decomposition[1:]:
        input_canonical_coord = alcove_to_canonical_coordinate(
            *input_alcove_coord)
        output_canonical_coord = alcove_to_canonical_coordinate(
            *output_alcove_coord)

        # NOTE: nonzero_p is guaranteed to be nonzero somewhere
        delta_p = [x != y for x, y in
                   zip(input_canonical_coord, output_canonical_coord)]
        nonzero_p = [x != 0 for x in input_canonical_coord]
        delta_count = sum(delta_p)

        if 3 == delta_count:
            raise NotImplementedError("Three coordinates changed at once, so "
                                      "the XX+XY trick doesn't apply.")
        elif 2 == delta_count:
            if 0 == sum([x and y for x, y in zip(delta_p, nonzero_p)]):
                raise NotImplementedError(
                    "Two vanishing coordinates changed at once, so "
                    "the XX+XY trick doesn't apply.")
            first_index, second_index = [index for index, value in
                                         enumerate(delta_p) if value]
        elif 1 == delta_count:
            first_index = delta_p.index(True)
            if nonzero_p[first_index]:
                second_index = 0 if first_index == 1 else 1
            else:
                second_index = nonzero_p.index(True)
        else:
            # no delta means no operation needed
            continue

        # calculate the ZI and IZ gates used in the (XX + YY) * XX decomposition
        r, s, u, v, x, y = decompose_xxyy_into_xxyy_xx(
            float(output_canonical_coord[first_index]),
            float(output_canonical_coord[second_index]),
            float(input_canonical_coord[first_index]),
            float(input_canonical_coord[second_index]),
            float(canonical_coordinate_table[operation][0]),
        )

        # calculate the local gates used to permute the canonical coordinates
        conj = canonical_rotation_circuit(first_index, second_index, q)

        # (zrzs + (input circuit)^conj + zuzv + operation + zxzy)^conj*
        output_circuit = qiskit.QuantumCircuit(q)
        output_circuit += conj.inverse()
        output_circuit.rz(x * 2, q[0])
        output_circuit.rz(y * 2, q[1])
        output_circuit.append(canonical_gate_table[operation], q)
        output_circuit.rz(u * 2, q[0])
        output_circuit.rz(v * 2, q[1])
        output_circuit += conj
        output_circuit.compose(qc, inplace=True)
        output_circuit += conj.inverse()
        output_circuit.rz(r * 2, q[0])
        output_circuit.rz(s * 2, q[1])
        output_circuit += conj

        qc = output_circuit

    return qc


#
# some other routines for random circuit generation
#


def sample_irreducible_circuit(coverage_set, operations, target_gate_polytope):
    """
    Produces a randomly generated circuit of the prescribed type which cannot
    be rewritten into a circuit of lower cost.
    """

    operation_gates = {
        operation.operations[0]:
            qiskit.extensions.UnitaryGate(
                canonical_matrix(*alcove_to_canonical_coordinate(
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
