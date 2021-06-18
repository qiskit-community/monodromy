"""
monodromy/xx_decompose/circuits.py

Tools for building optimal circuits out of XX interactions.

Inputs:
 + A set of native operations, described as `OperationPolytope`s.
 + A right-angled path, computed using the methods in `xx_decompose/paths.py`.

Output:
 + A circuit which implements the target operation (expressed exactly as the
   exponential of `a XX + b YY + c ZZ`) using the native operations and local
   gates.

These routines make a variety of assumptions, not yet proven:
 + CircuitPolytopes for XX interactions are permutation invariant.
 + Right-angled paths always exist.
"""

from dataclasses import dataclass
from functools import reduce
import math
import numpy as np
from operator import itemgetter
import warnings

import qiskit
from qiskit.circuit.library import RXGate, RYGate, RZGate
import qiskit.quantum_info

from ..coordinates import alcove_to_positive_canonical_coordinate
from ..coverage import CircuitPolytope
from ..exceptions import NoBacksolution
from ..io.base import OperationPolytopeData
from .paths import single_unordered_decomposition_hop
from .scipy import polyhedron_has_element
from ..static.matrices import canonical_matrix, rz_matrix
from ..utilities import epsilon


@dataclass
class OperationPolytope(OperationPolytopeData, CircuitPolytope):
    """
    See OperationPolytopeData.
    """
    pass


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


# TODO: THIS IS A STOPGAP!!!
def safe_arccos(numerator, denominator):
    """
    Computes arccos(n/d) with different (better?) numerical stability.
    """
    threshold = 0.005

    if abs(numerator) > abs(denominator) and \
            abs(numerator - denominator) < threshold:
        return 0.0
    elif abs(numerator) > abs(denominator) and \
            abs(numerator + denominator) < threshold:
        return np.pi
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            return np.arccos(numerator / denominator)


def decompose_xxyy_into_xxyy_xx(a_target, b_target, a1, b1, a2):
    """
    Consumes a target canonical interaction CAN(a_target, b_target) and
    source interactions CAN(a1, b1), CAN(a2), then manufactures a
    circuit identity of the form

    CAN(a_target, b_target) = (Zr, Zs) CAN(a1, b1) (Zu, Zv) CAN(a2) (Zx, Zy).

    Returns the 6-tuple (r, s, u, v, x, y).
    """

    cplus, cminus = np.cos(a1 + b1), np.cos(a1 - b1)
    splus, sminus = np.sin(a1 + b1), np.sin(a1 - b1)
    ca, sa = np.cos(a2), np.sin(a2)

    uplusv = 1 / 2 * safe_arccos(
        cminus ** 2 * ca ** 2 + sminus ** 2 * sa ** 2 - np.cos(a_target - b_target) ** 2,
        2 * cminus * ca * sminus * sa
    )
    uminusv = 1 / 2 * safe_arccos(
        cplus ** 2 * ca ** 2 + splus ** 2 * sa ** 2 - np.cos(a_target + b_target) ** 2,
        2 * cplus * ca * splus * sa
    )

    u, v = (uplusv + uminusv) / 2, (uplusv - uminusv) / 2

    # NOTE: the target matrix is phase-free
    middle_matrix = reduce(np.dot, [
        canonical_matrix(a1, b1),
        np.kron(rz_matrix(u), rz_matrix(v)),
        canonical_matrix(a2),
    ])

    phase_solver = np.array([
        [1 / 4, 1 / 4, 1 / 4, 1 / 4, ],
        [1 / 4, -1 / 4, -1 / 4, 1 / 4, ],
        [1 / 4, 1 / 4, -1 / 4, -1 / 4, ],
        [1 / 4, -1 / 4, 1 / 4, -1 / 4, ],
    ])
    inner_phases = [
        np.angle(middle_matrix[0, 0]),
        np.angle(middle_matrix[1, 1]),
        np.angle(middle_matrix[1, 2]) + np.pi / 2,
        np.angle(middle_matrix[0, 3]) + np.pi / 2,
    ]
    r, s, x, y = np.dot(phase_solver, inner_phases)

    # If there's a phase discrepancy, need to conjugate by an extra Z/2 (x) Z/2.
    generated_matrix = reduce(np.dot, [
        np.kron(rz_matrix(r), rz_matrix(s)),
        middle_matrix,
        np.kron(rz_matrix(x), rz_matrix(y)),
    ])
    if ((abs(np.angle(generated_matrix[3, 0]) - np.pi / 2) < 0.01 and a_target > b_target) or
            (abs(np.angle(generated_matrix[3, 0]) + np.pi / 2) < 0.01 and a_target < b_target)):
        x += np.pi / 4
        y += np.pi / 4
        r -= np.pi / 4
        s -= np.pi / 4

    return r, s, u, v, x, y


reflection_options = {
    "no reflection":  ([ 1,  1,  1],  1, []),        # we checked this phase
    "reflect XX, YY": ([-1, -1,  1],  1, [RZGate]),  # we checked this phase
    "reflect XX, ZZ": ([-1,  1, -1],  1, [RYGate]),  # we checked this phase, but only in a pair with Y shift
    "reflect YY, ZZ": ([ 1, -1, -1], -1, [RXGate]),  # unchecked
}

shift_options = {
    "no shift":    ([0, 0, 0],   1, []),                # we checked this phase
    "Z shift":     ([0, 0, 1],  1j, [RZGate]),          # we checked this phase
    "Y shift":     ([0, 1, 0], -1j, [RYGate]),          # we checked this phase, but only in a pair with reflect XX, ZZ
    "Y,Z shift":   ([0, 1, 1],  -1, [RYGate, RZGate]),  # unchecked
    "X shift":     ([1, 0, 0], -1j, [RXGate]),          # we checked this phase
    "X,Z shift":   ([1, 0, 1],   1, [RXGate, RZGate]),  # we checked this phase
    "X,Y shift":   ([1, 1, 0],  -1, [RXGate, RYGate]),  # unchecked
    "X,Y,Z shift": ([1, 1, 1], -1j, [RXGate, RYGate, RZGate]),  # unchecked
}


def apply_reflection(reflection_name, coordinate):
    """
    Given a reflection type and a canonical coordinate, applies the reflection
    and describes a circuit which enacts the reflection + a global phase shift.
    """
    reflection_scalars, reflection_phase_shift, source_reflection_gates = \
        reflection_options[reflection_name]
    reflected_coord = [x * y for x, y in zip(reflection_scalars, coordinate)]
    source_reflection = qiskit.QuantumCircuit(2)
    for gate in source_reflection_gates:
        source_reflection.append(gate(np.pi), [0])

    return reflected_coord, source_reflection, reflection_phase_shift


# TODO: I wonder if the global phase shift can be attached to the circuit...
def apply_shift(shift_name, coordinate):
    """
    Given a shift type and a canonical coordinate, applies the shift and
    describes a circuit which enacts the shift + a global phase shift.
    """
    shift_scalars, shift_phase_shift, source_shift_gates = \
        shift_options[shift_name]
    shifted_coord = [np.pi / 2 * x + y for x, y in zip(shift_scalars, coordinate)]

    source_shift = qiskit.QuantumCircuit(2)
    for gate in source_shift_gates:
        source_shift.append(gate(np.pi), [0])
        source_shift.append(gate(np.pi), [1])

    return shifted_coord, source_shift, shift_phase_shift


def canonical_rotation_circuit(first_index, second_index):
    """
    Given a pair of distinct indices 0 ≤ (first_index, second_index) ≤ 2,
    produces a two-qubit circuit which rotates a canonical gate

        a0 XX + a1 YY + a2 ZZ

    into

        a[first] XX + a[second] YY + a[other] ZZ .
    """
    conj = qiskit.QuantumCircuit(2)

    if (0, 1) == (first_index, second_index):
        pass  # no need to do anything
    elif (0, 2) == (first_index, second_index):
        conj.rx(-np.pi / 2, [0])
        conj.rx(np.pi / 2, [1])
    elif (1, 0) == (first_index, second_index):
        conj.rz(-np.pi / 2, [0])
        conj.rz(-np.pi / 2, [1])
    elif (1, 2) == (first_index, second_index):
        conj.rz(np.pi / 2, [0])
        conj.rz(np.pi / 2, [1])
        conj.ry(np.pi / 2, [0])
        conj.ry(-np.pi / 2, [1])
    elif (2, 0) == (first_index, second_index):
        conj.rz(np.pi / 2, [0])
        conj.rz(np.pi / 2, [1])
        conj.rx(np.pi / 2, [0])
        conj.rx(-np.pi / 2, [1])
    elif (2, 1) == (first_index, second_index):
        conj.ry(np.pi / 2, [0])
        conj.ry(-np.pi / 2, [1])

    return conj


def xx_circuit_step(
        source_monodromy_coord, operation, target_monodromy_coord,
        canonical_coordinate_table,
        canonical_gate_table
):
    """
    Builds a single step in an XX-based circuit.
    """
    source_canonical_coord, target_canonical_coord = [
        alcove_to_positive_canonical_coordinate(*x)
        for x in [source_monodromy_coord, target_monodromy_coord]
    ]

    permute_source_for_overlap, permute_target_for_overlap = None, None

    # apply all possible reflections, shifts to the source
    for source_reflection_name in reflection_options.keys():
        reflected_source_coord, source_reflection, reflection_phase_shift = \
            apply_reflection(source_reflection_name, source_canonical_coord)
        for source_shift_name in shift_options.keys():
            shifted_source_coord, source_shift, shift_phase_shift = \
                apply_shift(source_shift_name, reflected_source_coord)

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
            source_first, source_second = [x for x in [0, 1, 2]
                                           if x != source_shared]
            target_first, target_second = [x for x in [0, 1, 2]
                                           if x != target_shared]

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
                source_first, source_second
            )
            permute_target_for_overlap = canonical_rotation_circuit(
                target_first, target_second
            )
            break

        if permute_source_for_overlap is not None:
            break

    if permute_source_for_overlap is None:
        raise NoBacksolution()

    prefix_circuit, affix_circuit = \
        qiskit.QuantumCircuit(2), qiskit.QuantumCircuit(2)

    # the basic formula we're trying to work with is:
    # target^p_t_f_o =
    #     rs * (source^s_reflection * s_shift)^p_s_f_o * uv * operation * xy
    # but we're rearranging it into the form
    #   target = affix source prefix
    # and computing just the prefix / affix circuits.

    # the outermost prefix layer comes from the (inverse) target permutation.
    prefix_circuit += permute_target_for_overlap.inverse()
    # the middle prefix layer comes from the local Z rolls.
    prefix_circuit.rz(2 * x, [0])
    prefix_circuit.rz(2 * y, [1])
    prefix_circuit.compose(canonical_gate_table[operation], inplace=True)
    prefix_circuit.rz(2 * u, [0])
    prefix_circuit.rz(2 * v, [1])
    # the innermost prefix layer is source_reflection, shifted by source_shift,
    # finally conjugated by p_s_f_o.
    prefix_circuit += permute_source_for_overlap
    prefix_circuit += source_reflection
    prefix_circuit.global_phase += -np.log(reflection_phase_shift).imag
    prefix_circuit.global_phase += -np.log(shift_phase_shift).imag

    # the affix circuit is constructed in reverse.
    # first (i.e., innermost), we install the other half of the source
    # transformations and p_s_f_o.
    affix_circuit += source_reflection.inverse()
    affix_circuit += source_shift
    affix_circuit += permute_source_for_overlap.inverse()
    # then, the other local rolls in the middle.
    affix_circuit.rz(2 * r, [0])
    affix_circuit.rz(2 * s, [1])
    # finally, the other half of the p_t_f_o conjugation.
    affix_circuit += permute_target_for_overlap

    return {
        "prefix_circuit": prefix_circuit,
        "affix_circuit": affix_circuit
    }


def canonical_xx_circuit(
        target, coverage_set, precomputed_backsolutions, operations
):
    """
    Assembles a QISKit circuit from XX-type interactions, as enumerated in
    `operations`, which emulates the canonical gate at monodromy coordinates
    `target`.  `coverage_set` and `precomputed_backsolutions` are calculated
    as in `defaults.py`.
    """

    canonical_coordinate_table = {
        operation.operations[0]: alcove_to_positive_canonical_coordinate(
            *[next(-eq[0] / eq[1 + j]
                   for eq in operation.convex_subpolytopes[0].equalities
                   if eq[1 + j] != 0)
              for j in range(3)]
        )
        for operation in operations
    }

    # make sure that all the operations are of XX-interaction type
    assert all([abs(c[1]) < epsilon and abs(c[2]) < epsilon
                for c in canonical_coordinate_table.values()])

    canonical_gate_table = {
        operation.operations[0]: operation.canonical_circuit
        for operation in operations
    }

    # find outermost polytope.
    # NOTE: In practice, this computation has already been done.
    target_polytope = None
    best_cost = float("inf")
    for polytope in coverage_set:
        if polytope.cost < best_cost and polyhedron_has_element(
                polytope, target):
            target_polytope = polytope
            best_cost = polytope.cost

    if target_polytope is None:
        raise ValueError(f"{target} not contained in coverage set.")

    operations_remaining = target_polytope.operations

    # empty decompositions are easy!
    if 0 == len(operations_remaining):
        return qiskit.QuantumCircuit(2)

    # assemble the prefix / affix circuits
    prefix_circuit, affix_circuit = \
        qiskit.QuantumCircuit(2), qiskit.QuantumCircuit(2)
    while 1 < len(operations_remaining):
        try:
            next_target, next_operations_remaining, hop = \
                itemgetter("ancestor", "operations_remaining", "hop")(
                    single_unordered_decomposition_hop(
                        target, operations_remaining, precomputed_backsolutions
                    )
                )

            preceding_prefix_circuit, preceding_affix_circuit = \
                itemgetter("prefix_circuit", "affix_circuit")(xx_circuit_step(
                    *hop,
                    canonical_coordinate_table,
                    canonical_gate_table
                ))

            prefix_circuit.compose(preceding_prefix_circuit, inplace=True)
            affix_circuit.compose(preceding_affix_circuit, inplace=True,
                                  front=True)

            target = next_target
            operations_remaining = next_operations_remaining
        except NoBacksolution:
            pass

    circuit = prefix_circuit

    # lastly, deal with the "leading" gate.
    if target[0] <= 1 / 4:
        circuit.compose(
            canonical_gate_table[operations_remaining[0]],
            inplace=True
        )
    else:
        _, source_reflection, reflection_phase_shift = \
            apply_reflection("reflect XX, YY", [0, 0, 0])
        _, source_shift, shift_phase_shift = \
            apply_shift("X shift", [0, 0, 0])

        circuit += source_reflection
        circuit.compose(canonical_gate_table[operations_remaining[0]],
                        inplace=True)
        circuit += source_reflection.inverse()
        circuit += source_shift
        circuit.global_phase += -np.log(shift_phase_shift).imag
        circuit.global_phase += -np.log(reflection_phase_shift).imag

    circuit.compose(affix_circuit, inplace=True)

    return circuit
