"""
monodromy/xx_decompose/qiskit.py

Staging ground for a QISKit Terra compilation pass which emits ZX circuits.
"""

from fractions import Fraction
from math import cos, sin, sqrt
from operator import itemgetter

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit.library import RZXGate
from qiskit.extensions import UnitaryGate
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info.synthesis import OneQubitEulerDecomposer
from qiskit.quantum_info.synthesis.two_qubit_decompose import \
    TwoQubitWeylDecomposition

from .circuits import apply_reflection, apply_shift, canonical_xx_circuit
from ..coordinates import alcove_to_positive_canonical_coordinate, \
    fidelity_distance, unitary_to_alcove_coordinate
from .defaults import get_zx_operations, default_data
from ..exceptions import NoBacksolution
from ..io.inflate import filter_scipy_data
from .scipy import nearest_point_polyhedron, polyhedron_has_element, \
    optimize_over_polytope
from ..utilities import epsilon


# TODO: stick to working with canonical coordinates everywhere, rather than
#       flipping with monodromy coordinates.


class MonodromyZXDecomposer:
    """
    A class for decomposing 2-qubit unitaries into minimal number of uses of a
    2-qubit basis gate.

    Args:
        operations (List[Polytope]): Bases gates.
        coverage_set (List[CircuitPolytope]): Coverage regions for different
            circuit shapes.
        precomputed_backsolutions (List[CircuitPolytope]): Precomputed solution
            spaces for peeling off one multiqubit interaction from a circuit
            template.  Use `calculate_scipy_coverage_set` to generate.
        euler_basis (str): Basis string provided to OneQubitEulerDecomposer for
            1Q synthesis.  Defaults to "U3".
        top_k (int): Retain the top k Euclidean solutions when searching for
            approximations.  Zero means retain all the Euclidean solutions;
            negative values cause undefined behavior.
    """

    def __init__(self, operations, coverage_set, precomputed_backsolutions,
                 euler_basis="U3", top_k=4):
        self.operations, self.coverage_set, self.precomputed_backsolutions = \
            operations, coverage_set, precomputed_backsolutions

        self._decomposer1q = OneQubitEulerDecomposer(euler_basis)
        self.gate = RZXGate(np.pi/2)
        self.top_k = top_k

    def _rank_euclidean_polytopes(self, target):
        """
        Returns the closest polytopes to `target` from the coverage set, in
        anti-proximity order: the nearest polytope is last.

        NOTE: `target` is to be provided in monodromy coordinates.
        NOTE: "Proximity" has a funny meaning.  In each polytope, we find the
            nearest point in Euclidean distance, then sort the polytopes by the
            trace distances from those points to the target (+ polytope cost).
        """
        polytope_costs = []
        for gate_polytope in self.coverage_set:
            candidate_point = nearest_point_polyhedron(target, gate_polytope)
            candidate_cost = 1 - fidelity_distance(target, candidate_point)
            polytope_costs.append(
                (gate_polytope, candidate_cost + gate_polytope.cost)
            )
            if polyhedron_has_element(gate_polytope, target):
                break

        polytope_costs = sorted(polytope_costs, key=lambda x: x[1],
                                reverse=True)

        return [x[0] for x in polytope_costs]

    def _first_containing_polytope(self, target):
        """
        Finds the cheapest coverage polytope to which the `target` belongs.

        NOTE: `target` is to be provided in monodromy coordinates.
        """
        for gate_polytope in self.coverage_set:
            if polyhedron_has_element(gate_polytope, target):
                return gate_polytope

    @staticmethod
    def _build_objective(target):
        """
        Constructs the functional which measures the trace distance to `target`.
        """
        a0, b0, c0 = alcove_to_positive_canonical_coordinate(*target)

        def objective(array):
            return -fidelity_distance(target, array)

        def jacobian(array):
            a, b, c = alcove_to_positive_canonical_coordinate(*array)

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

            # gradient in monodromy coordinates
            return np.pi / 2 * np.array([da + db, da + dc, db + dc])

        return {
            "objective": objective,
            "jacobian": jacobian,
            # "hessian": hessian,
        }

    def _best_decomposition(self, target):
        """
        Searches over different circuit templates for the least costly
        embodiment of the canonical coordinate `target`.  Returns a dictionary
        with keys "cost", "point", and "operations".

        NOTE: Expects `target` to be supplied in monodromy coordinates.
        """

        ranked_polytopes = self._rank_euclidean_polytopes(target)
        ranked_polytopes = ranked_polytopes[-self.top_k:]

        best_cost = 0.8
        best_point = [0, 0, 0]
        best_polytope = None
        objective, jacobian = itemgetter("objective", "jacobian")(
            self._build_objective(target)
        )

        for gate_polytope, _ in sorted([(p, p.cost) for p in ranked_polytopes],
                                       key=lambda x: x[1]):
            # short-circuit in the case of exact membership
            if polyhedron_has_element(gate_polytope, target):
                if gate_polytope.cost < best_cost:
                    best_cost, best_point, best_polytope = \
                        gate_polytope.cost, target, gate_polytope
                break

            # otherwise, numerically optimize the trace distance
            for convex_polytope in gate_polytope.convex_subpolytopes:
                solution = optimize_over_polytope(
                    objective, convex_polytope,
                    jacobian=jacobian,
                    # hessian=hessian,
                )

                if solution.fun + 1 + gate_polytope.cost < best_cost:
                    best_cost = solution.fun + 1 + gate_polytope.cost
                    best_point = solution.x
                    best_polytope = gate_polytope

        if best_polytope is None:
            raise ValueError("Failed to find a match.")

        return {
            "point": best_point,
            "cost": best_cost,
            "operations": best_polytope.operations
        }

    def num_basis_gates(self, unitary):
        """
        Counts the number of gates that would be emitted during re-synthesis.

        NOTE: Used by ConsolidateBlocks.
        """
        target = unitary_to_alcove_coordinate(unitary)[:3]
        best_polytope = self._rank_euclidean_polytopes(target)[-1]
        return len(best_polytope.operations)

    # TODO: remit this to `optimize_1q_decomposition.py` in qiskit
    def decompose_1q(self, circuit):
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

    def __call__(self, u, basis_fidelity=None, approximate=True, chatty=False):
        """
        Fashions a circuit which (perhaps `approximate`ly) models the special
        unitary operation `u`, using the circuit templates supplied at
        initialization.

        NOTE: Ignores `basis_fidelity` in favor of the operation tables loaded
              at initialization time.
        """
        target = unitary_to_alcove_coordinate(u)[:3]
        best_point, best_cost, best_operations = \
            itemgetter("point", "cost", "operations")(
                self._best_decomposition(target)
            )

        if chatty:
            print(f"Overall best: {best_point} hits {best_cost} via "
                  f"{'.'.join(best_operations)}")
            print(f"In canonical coordinates: "
                  f"{alcove_to_positive_canonical_coordinate(*best_point)} "
                  f"from {alcove_to_positive_canonical_coordinate(*target)}")

        circuit = canonical_xx_circuit(
            best_point,
            self.coverage_set, self.precomputed_backsolutions, self.operations
        )

        weyl_decomposition = TwoQubitWeylDecomposition(u)

        if chatty:
            print([weyl_decomposition.a,
                   weyl_decomposition.b,
                   weyl_decomposition.c])
            print(alcove_to_positive_canonical_coordinate(*target))

        # change to positive canonical coordinates
        if abs(weyl_decomposition.c -
               alcove_to_positive_canonical_coordinate(*target)[2]) < epsilon:
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

            if chatty:
                import qiskit.quantum_info
                with np.printoptions(precision=3, suppress=True):
                        weyl_circuit = QuantumCircuit(q)
                        weyl_decomposition._weyl_gate(False, weyl_circuit, 1e-10)
                        print(qiskit.quantum_info.Operator(weyl_circuit).data)
                        print(qiskit.quantum_info.Operator(circuit).data)
                        print("====")

        circ = QuantumCircuit(2, global_phase=weyl_decomposition.global_phase)

        circ.append(UnitaryGate(weyl_decomposition.K2r), [0])
        circ.append(UnitaryGate(weyl_decomposition.K2l), [1])
        circ.compose(circuit, [0, 1], inplace=True)
        circ.append(UnitaryGate(weyl_decomposition.K1r), [0])
        circ.append(UnitaryGate(weyl_decomposition.K1l), [1])

        return self.decompose_1q(circ)


def monodromy_decomposer_from_approximation_degree(
        euler_basis="U3",
        approximation_degree=1.0,
        flat_rate=1e-10,
):
    """
    `euler_basis`: Basis into which to decompose single-qubit gates.
    `approximation_degree`: Linear fidelity rate.  The cost of a full ZX gate
        is taken to be (1 - approximation_degree) + flat_rate.
    `flat_rate`: Affine offset in a linear error model. Undefined behavior when
        set to zero; OK to use a small value like 1e-10.
    """
    cost_table = {
        Fraction(1, k): flat_rate + 1 / k * (1 - approximation_degree)
        for k in [1, 2, 3]
    }
    operations = get_zx_operations(cost_table)
    inflated_data = filter_scipy_data(operations, **default_data, chatty=False)
    return MonodromyZXDecomposer(**inflated_data, euler_basis=euler_basis)
