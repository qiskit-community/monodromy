"""
monodromy/xx_decompose/qiskit.py

Staging ground for a QISKit Terra compilation pass which emits ZX circuits.
"""

import heapq
from math import cos, sin, sqrt
from operator import itemgetter
from typing import Dict, List, Optional

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit.library import RZXGate
from qiskit.extensions import UnitaryGate
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info.synthesis import OneQubitEulerDecomposer
from qiskit.quantum_info.synthesis.two_qubit_decompose import \
    TwoQubitWeylDecomposition

from ..coordinates import fidelity_distance, \
    monodromy_to_positive_canonical_coordinate, unitary_to_monodromy_coordinate
from ..utilities import epsilon

from .circuits import apply_reflection, apply_shift, canonical_xx_circuit
from ..static.interference import polytope_from_strengths
from .scipy import nearest_point_polyhedron, polyhedron_has_element, \
    optimize_over_polytope


class MonodromyZXDecomposer:
    """
    A class for optimal decomposition of 2-qubit unitaries into 2-qubit basis
    gates of XX type (i.e., each locally equivalent to CAN(alpha, 0, 0) for a
    possibly varying alpha).

    Args:
        euler_basis: Basis string provided to OneQubitEulerDecomposer for 1Q
            synthesis.  Defaults to "U3".
        top_k: Retain the top k Euclidean solutions when searching for
            approximations.  Zero means retain all the Euclidean solutions;
            negative values cause undefined behavior.
        embodiments: An dictionary mapping interaction strengths alpha to native
            circuits which embody the gate CAN(alpha, 0, 0). Strengths are taken
            to be normalized, so that 1/2 represents the class of a full CX.

    NOTE: If embodiments is not passed, or if an entry is missing, it will
        be populated as needed using the method _default_embodiment.
    """

    def __init__(
            self,
            euler_basis: str = "U3",
            top_k: int = 4,
            embodiments: Optional[dict] = None,
    ):
        self._decomposer1q = OneQubitEulerDecomposer(euler_basis)
        self.gate = RZXGate(np.pi/2)
        self.top_k = top_k
        self.embodiments = embodiments if embodiments is not None else {}

    @staticmethod
    def _default_embodiment(strength):
        """
        If the user does not provide a custom implementation of XX(strength),
        then this routine defines a default implementation using RZX or CX.
        """
        xx_circuit = QuantumCircuit(2)

        if strength == np.pi/2:
            xx_circuit.h(0)
            xx_circuit.cx(0, 1)
            xx_circuit.h(1)
            xx_circuit.rz(np.pi / 2, 0)
            xx_circuit.rz(np.pi / 2, 1)
            xx_circuit.h(1)
            xx_circuit.h(0)
            xx_circuit.global_phase += np.pi / 4
        else:
            xx_circuit.h(0)
            xx_circuit.rzx(strength, 0, 1)
            xx_circuit.h(0)

        return xx_circuit

    # TODO: unify this with `_cheapest_container`
    def _rank_euclidean_sequences(
            self, target: List[float], strengths: Dict[float, float]
    ):
        """
        Returns the closest polytopes to `target` from the coverage set, in
        anti-proximity order: the nearest polytope is last.

        NOTE: `target` is to be provided in positive canonical coordinates.
            `strengths` is a dictionary mapping ZX strengths (that is, pi/2
            corresponds to a full CX) to costs.
        NOTE: "Proximity" has a funny meaning.  In each polytope, we find the
            nearest point in Euclidean distance, then sort the polytopes by the
            trace distances from those points to the target (+ polytope cost).
        """
        priority_queue = []
        polytope_costs = []
        heapq.heappush(priority_queue, (0, []))

        while True:
            sequence_cost, sequence = heapq.heappop(priority_queue)

            gate_polytope = polytope_from_strengths(
                [x / 2 for x in sequence], scale_factor=np.pi / 2
            )
            candidate_point = nearest_point_polyhedron(target, gate_polytope)
            candidate_cost = 1 - fidelity_distance(target, candidate_point)
            polytope_costs.append((sequence, candidate_cost + sequence_cost))

            if polyhedron_has_element(gate_polytope, target):
                break

            for strength, extra_cost in strengths.items():
                if len(sequence) == 0 or strength <= sequence[-1]:
                    heapq.heappush(
                        priority_queue,
                        (sequence_cost + extra_cost, sequence + [strength])
                    )

        polytope_costs = sorted(polytope_costs, key=lambda x: x[1],
                                reverse=True)

        return polytope_costs

    @staticmethod
    def _cheapest_container(available_strengths, canonical_coordinate):
        """
        Finds the cheapest sequence of `available_strengths` for which
        `canonical_coordinate` belongs to the associated interaction polytope.

        `canonical_coordinate` is a positive canonical coordinate. `strengths`
        is a dictionary mapping the available strengths, normalized so that pi/2
        represents CX = RZX(pi/2), to their (infidelity) costs.
        """
        priority_queue = []
        heapq.heappush(priority_queue, (0, []))

        while True:
            cost, sequence = heapq.heappop(priority_queue)

            strength_polytope = polytope_from_strengths(
                [x / 2 for x in sequence], scale_factor=np.pi / 2
            )
            if polyhedron_has_element(strength_polytope, canonical_coordinate):
                return sequence

            for strength, extra_cost in available_strengths.items():
                if len(sequence) == 0 or strength <= sequence[-1]:
                    heapq.heappush(priority_queue,
                                   (cost + extra_cost, sequence + [strength]))

    @staticmethod
    def _build_objective(target):
        """
        Constructs the functional which measures the trace distance to `target`.

        NOTE: `target` is in unnormalized positive canonical coordinates.
        """
        a0, b0, c0 = target

        def objective(array):
            return -fidelity_distance(target, array)

        def jacobian(array):
            a, b, c = array

            # squares
            ca2, sa2 = cos(a0 - a) ** 2, sin(a0 - a) ** 2
            cb2, sb2 = cos(b0 - b) ** 2, sin(b0 - b) ** 2
            cc2, sc2 = cos(c0 - c) ** 2, sin(c0 - c) ** 2
            # double angles
            c2a, s2a = cos(2 * a0 - 2 * a), sin(2 * a0 - 2 * a)
            c2b, s2b = cos(2 * b0 - 2 * b), sin(2 * b0 - 2 * b)
            c2c, s2c = cos(2 * c0 - 2 * c), sin(2 * c0 - 2 * c)

            # gradient in canonical coordinates
            sqrt_sum = sqrt(ca2 * cb2 * cc2 + sa2 * sb2 * sc2)
            da = -(c2b + c2c) * s2a / (5 * sqrt_sum)
            db = -(c2a + c2c) * s2b / (5 * sqrt_sum)
            dc = -(c2a + c2b) * s2c / (5 * sqrt_sum)

            return np.array([da, db, dc])

        return {
            "objective": objective,
            "jacobian": jacobian,
            # "hessian": hessian,
        }

    def _best_decomposition(self, target, strengths):
        """
        Searches over different circuit templates for the least costly
        embodiment of the canonical coordinate `target`.  Returns a dictionary
        with keys "cost", "point", and "operations".

        NOTE: Expects `target` in positive canonical coordinates.
        NOTE: `strengths` is a dictionary mapping normalized XX strengths (that
              is, 1/2 corresponds to a full CX) to costs.
        """

        ranked_sequences = self._rank_euclidean_sequences(target, strengths)
        ranked_sequences = ranked_sequences[-self.top_k:]

        best_cost = 0.8  # a/k/a np.inf
        best_point = [0, 0, 0]
        best_sequence = []
        objective, jacobian = itemgetter("objective", "jacobian")(
            self._build_objective(target)
        )

        for sequence, cost in sorted(ranked_sequences, key=lambda x: x[1]):
            # short-circuit in the case of exact membership
            gate_polytope = polytope_from_strengths(
                [x / 2 for x in sequence], scale_factor=np.pi / 2
            )
            if polyhedron_has_element(gate_polytope, target):
                if cost < best_cost:
                    best_cost, best_point, best_sequence = \
                        cost, target, sequence
                break

            # otherwise, numerically optimize the trace distance
            for convex_polytope in gate_polytope.convex_subpolytopes:
                solution = optimize_over_polytope(
                    objective, convex_polytope,
                    jacobian=jacobian,
                    # hessian=hessian,
                )

                if (solution.fun + 1) + cost < best_cost:
                    best_cost = (solution.fun + 1) + cost
                    best_point = solution.x
                    best_sequence = sequence

        return {
            "point": best_point,
            "cost": best_cost,
            "sequence": best_sequence
        }

    def num_basis_gates(self, unitary):
        """
        Counts the number of gates that would be emitted during re-synthesis.

        NOTE: Used by ConsolidateBlocks.
        """
        strengths = self._strength_to_infidelity(1.0)
        target = unitary_to_monodromy_coordinate(unitary)[:3]
        target = monodromy_to_positive_canonical_coordinate(*target)
        best_sequence, _ = self._rank_euclidean_sequences(target, strengths)[-1]
        return len(best_sequence)

    def _strength_to_infidelity(self, basis_fidelity):
        """
        Converts a dictionary mapping ZX strengths to fidelities to a dictionary
        mapping ZX strengths to infidelities. Also supports some of the other
        formats QISKit uses: injects a default set of infidelities for CX, CX/2,
        and CX/3 if None is supplied, or extends a single float infidelity over
        CX, CX/2, and CX/3 if only a single float is supplied.
        """

        if basis_fidelity is None or isinstance(basis_fidelity, float):
            if isinstance(basis_fidelity, float):
                slope, offset = 1 - basis_fidelity, 1e-10
            else:
                # some reasonable default values
                slope, offset = (64 * 90) / 1000000, 909 / 1000000 + 1 / 1000
            return {
                strength: slope * strength / (np.pi / 2) + offset
                for strength in [np.pi / 2, np.pi / 4, np.pi / 6]
            }
        elif isinstance(basis_fidelity, dict):
            return {
                strength: 1 - fidelity
                for (strength, fidelity) in basis_fidelity.items()
            }

        raise TypeError("Unknown basis_fidelity payload.")

    def __call__(self, u, basis_fidelity=None, approximate=True, chatty=False):
        """
        Fashions a circuit which (perhaps `approximate`ly) models the special
        unitary operation `u`, using the circuit templates supplied at
        initialization.  The routine uses `basis_fidelity` to select the optimal
        circuit template, including when performing exact synthesis; the
        contents of `basis_fidelity` is a dictionary mapping interaction
        strengths (scaled so that CX = RZX(pi/2) corresponds to pi/2) to circuit
        fidelities.
        """
        strength_to_infidelity = self._strength_to_infidelity(basis_fidelity)

        # get the associated _positive_ canonical coordinate
        weyl_decomposition = TwoQubitWeylDecomposition(u)
        target = [getattr(weyl_decomposition, x) for x in ("a", "b", "c")]
        if target[-1] < -epsilon:
            target = [np.pi / 2 - target[0], target[1], -target[2]]

        # scan for the best point
        if approximate:
            best_point, best_sequence = \
                itemgetter("point", "sequence")(self._best_decomposition(
                    target, strengths=strength_to_infidelity
                ))
        else:
            best_point, best_sequence = target, self._cheapest_container(strength_to_infidelity, target)
        # build the circuit building this canonical gate
        embodiments = {
            k: self.embodiments.get(k, self._default_embodiment(k))
            for k, v in strength_to_infidelity.items()
        }
        circuit = canonical_xx_circuit(best_point, best_sequence, embodiments)

        # change to positive canonical coordinates
        if weyl_decomposition.c >= -epsilon:
            # if they're the same...
            corrected_circuit = QuantumCircuit(2)
            corrected_circuit.rz(np.pi, [0])
            corrected_circuit.compose(circuit, [0, 1], inplace=True)
            corrected_circuit.rz(-np.pi, [0])
            circuit = corrected_circuit
        else:
            # else they're in the "positive" scissors part...
            corrected_circuit = QuantumCircuit(2)
            _, source_reflection, reflection_phase_shift = apply_reflection(
                "reflect XX, ZZ", [0, 0, 0]
            )
            _, source_shift, shift_phase_shift = apply_shift(
                "X shift", [0, 0, 0]
            )

            corrected_circuit.compose(source_reflection.inverse(), inplace=True)
            corrected_circuit.rz(np.pi, [0])
            corrected_circuit.compose(circuit, [0, 1], inplace=True)
            corrected_circuit.rz(-np.pi, [0])
            corrected_circuit.compose(source_shift.inverse(), inplace=True)
            corrected_circuit.compose(source_reflection, inplace=True)
            corrected_circuit.global_phase += np.pi / 2

            circuit = corrected_circuit

        circ = QuantumCircuit(2, global_phase=weyl_decomposition.global_phase)

        circ.append(UnitaryGate(weyl_decomposition.K2r), [0])
        circ.append(UnitaryGate(weyl_decomposition.K2l), [1])
        circ.compose(circuit, [0, 1], inplace=True)
        circ.append(UnitaryGate(weyl_decomposition.K1r), [0])
        circ.append(UnitaryGate(weyl_decomposition.K1l), [1])

        # return self._decompose_1q(circ)

        return circ

    # TODO: remit this to `optimize_1q_decomposition.py` in qiskit
    def _decompose_1q(self, circuit):
        """
        Gather the one-qubit substrings in a two-qubit circuit and apply the
        local decomposer.
        """
        circ_0 = QuantumCircuit(1)
        circ_1 = QuantumCircuit(1)
        output_circuit = QuantumCircuit(2, global_phase=circuit.global_phase)

        for gate, q, _ in circuit:
            if q == [circuit.qregs[0][0]]:
                circ_0.append(gate, [0])
            elif q == [circuit.qregs[0][1]]:
                circ_1.append(gate, [0])
            else:
                circ_0 = self._decomposer1q(Operator(circ_0).data)
                circ_1 = self._decomposer1q(Operator(circ_1).data)
                output_circuit.compose(circ_0, [0], inplace=True)
                output_circuit.compose(circ_1, [1], inplace=True)
                output_circuit.append(gate, [0, 1])
                circ_0 = QuantumCircuit(1)
                circ_1 = QuantumCircuit(1)

        circ_0 = self._decomposer1q(Operator(circ_0).data)
        circ_1 = self._decomposer1q(Operator(circ_1).data)
        output_circuit.compose(circ_0, [0], inplace=True)
        output_circuit.compose(circ_1, [1], inplace=True)

        return output_circuit
