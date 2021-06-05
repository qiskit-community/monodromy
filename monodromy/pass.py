"""
monodromy/pass.py

Staging ground for a QISKit Terra compilation pass which emits circuits in the
style of monodromy/circuits.py .
"""

from math import cos, sin, sqrt

import numpy as np
import scipy.optimize

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info.synthesis import OneQubitEulerDecomposer
from qiskit.quantum_info.synthesis.two_qubit_decompose import \
    TwoQubitWeylDecomposition

from .circuits import apply_reflection, apply_shift, \
    xx_circuit_from_decomposition
from .coordinates import alcove_to_positive_canonical_coordinate, \
    unitary_to_alcove_coordinate
from .polytopes import ConvexPolytopeData
from .scipy import nearly, NoBacksolution, scipy_unordered_decomposition_hops
from .utilities import epsilon


def optimize_over_polytope(
        fn,
        convex_polytope: ConvexPolytopeData
) -> scipy.optimize.OptimizeResult:
    """
    Optimizes the function `fn`: array --> reals over `convex_polytope`.
    """
    dimension = None

    constraints = []

    if 0 < len(convex_polytope.inequalities):
        dimension = -1 + len(convex_polytope.inequalities[0])
        A_ub = np.array([[float(x) for x in ineq[1:]]
                         for ineq in convex_polytope.inequalities])
        b_ub = np.array([float(ineq[0])
                         for ineq in convex_polytope.inequalities])
        constraints.append(dict(
            type='ineq',
            fun=lambda x: A_ub @ x + b_ub
        ))

    if 0 < len(convex_polytope.equalities):
        dimension = -1 + len(convex_polytope.equalities[0])
        A_eq = np.array([[float(x) for x in eq[1:]]
                         for eq in convex_polytope.equalities])
        b_eq = np.array([float(eq[0]) for eq in convex_polytope.equalities])
        constraints.append(dict(
            type='ineq',
            fun=lambda x: A_eq @ x + b_eq
        ))
        constraints.append(dict(
            type='ineq',
            fun=lambda x: -A_eq @ x - b_eq
        ))

    return scipy.optimize.minimize(
        fun=fn,
        x0=np.array([1 / 4] * dimension),
        constraints=constraints
    )


def has_element(polytope, point):
    """
    A standalone variant of Polytope.has_element.
    """
    return any([(all([-epsilon <= inequality[0] +
                      sum(x * y for x, y in
                          zip(point, inequality[1:]))
                      for inequality in cp.inequalities]) and
                 all([abs(equality[0] + sum(x * y for x, y in
                                            zip(point, equality[1:])))
                      <= epsilon
                      for equality in cp.equalities]))
                for cp in polytope.convex_subpolytopes])


# TODO: stick to working with canonical coordinates everywhere, rather than
#       flipping with monodromy coordinates.


class MonodromyZXDecomposer():
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
    """

    def __init__(self, operations, coverage_set, precomputed_backsolutions,
                 euler_basis="U3"):
        self.operations, self.coverage_set, self.precomputed_backsolutions = \
            operations, coverage_set, precomputed_backsolutions

        self._decomposer1q = OneQubitEulerDecomposer(euler_basis)

    @staticmethod
    def _build_objective(target):
        """
        Constructs the functional which measures the trace distance to `target`.
        """
        a0, b0, c0 = alcove_to_positive_canonical_coordinate(*target)

        def internal_objective(array):
            a, b, c = alcove_to_positive_canonical_coordinate(*array)

            return -1 / 20 * (4 + 16 * sqrt(
                cos(a0 - a) ** 2 * cos(b0 - b) ** 2 * cos(c0 - c) ** 2 +
                sin(a0 - a) ** 2 * sin(b0 - b) ** 2 * sin(c0 - c) ** 2))

        return internal_objective

    def _best_decomposition(self, target,
                            approximate=True,
                            chatty=True):  # -> (coord, cost, op'ns)
        """
        Searches over different circuit templates for the least coatly
        embodiment of the canonical coordinate `target`.  If `approximate` is
        flagged, this permits approximate solutions whose trace distance is less
        than the cost of using more gates to model the target exactly.

        NOTE: Expects `target` to be supplied in monodromy coordinates.
        """
        if chatty:
            print(f"Aiming for {target} (monodromy).")

        overall_best_cost = 1
        overall_exact_cost = None
        overall_best_point = [0, 0, 0]
        overall_best_operations = []

        for gate_polytope in self.coverage_set:
            if chatty:
                print(f"Working on {'.'.join(gate_polytope.operations)}: ",
                      end="")
            best_distance = 1
            best_point = [0, 0, 0]

            for convex_polytope in gate_polytope.convex_subpolytopes:
                solution = optimize_over_polytope(self._build_objective(target),
                                                  convex_polytope)
                if solution.fun + 1 < best_distance:
                    best_distance = solution.fun + 1
                    best_point = solution.x

            if chatty:
                print(f"{gate_polytope.cost} + {best_distance} = "
                      f"{gate_polytope.cost + best_distance}")

            if gate_polytope.cost + best_distance < overall_best_cost:
                overall_best_cost = gate_polytope.cost + best_distance
                overall_best_point = best_point
                overall_best_operations = gate_polytope.operations

            # stash the first polytope we belong to
            if overall_exact_cost is None and has_element(gate_polytope,
                                                          target):
                overall_exact_cost = gate_polytope.cost
                overall_exact_operations = gate_polytope.operations
                if not approximate:
                    return (
                    target, overall_exact_cost, overall_exact_operations)
                break

        if approximate:
            return overall_best_point, overall_best_cost,overall_best_operations
        else:
            raise ValueError("Failed to find a match.")

    def __call__(self, u, approximate=True, chatty=True):
        """
        Fashions a circuit which (perhaps `approximate`ly) models the special
        unitary operation `u`, using the circuit templates supplied at
        initialization.
        """
        target = unitary_to_alcove_coordinate(u)[:3]
        overall_best_point, overall_best_cost, overall_best_operations = \
            self._best_decomposition(target, approximate=approximate)

        if chatty:
            print(f"Overall best: {overall_best_point} hits "
                  f"{overall_best_cost} via "
                  f"{'.'.join(overall_best_operations)}")
            print(f"In canonical coordinates: "
                  f"{alcove_to_positive_canonical_coordinate(*overall_best_point)} "
                  f"from {alcove_to_positive_canonical_coordinate(*target)}")

        circuit = None
        while circuit is None:
            try:
                decomposition = scipy_unordered_decomposition_hops(
                    self.coverage_set,
                    self.precomputed_backsolutions,
                    nearly(*overall_best_point)
                )

                if chatty:
                    print("Trying decomposition:")
                    for input_alcove_coord, operation, output_alcove_coord \
                            in decomposition:
                        line = "("
                        line += " π/4, ".join([f"{float(x / (np.pi / 4)):+.3f}"
                                               for x in alcove_to_positive_canonical_coordinate(*input_alcove_coord)])
                        line += f" π/4) --{operation:-<7}-> ("
                        line += " π/4, ".join([f"{float(x / (np.pi / 4)):+.3f}"
                                               for x in alcove_to_positive_canonical_coordinate(*output_alcove_coord)])
                        line += " π/4)"
                        print(line)

                circuit = xx_circuit_from_decomposition(
                    decomposition,
                    self.operations
                )
            except NoBacksolution:
                if chatty:
                    print(f"Nerts!")
                circuit = None
                pass

        weyl_decomposition = TwoQubitWeylDecomposition(u)
        q = QuantumRegister(2)

        if chatty:
            print([weyl_decomposition.a,
                   weyl_decomposition.b,
                   weyl_decomposition.c])
            print(alcove_to_positive_canonical_coordinate(*target))

        # change to positive canonical coordinates
        if abs(weyl_decomposition.c -
               alcove_to_positive_canonical_coordinate(*target)[2]) < epsilon:
            # if they're the same...
            corrected_circuit = QuantumCircuit(q)
            corrected_circuit.rz(np.pi, [0])
            corrected_circuit.compose(circuit, [0, 1], inplace=True)
            corrected_circuit.rz(-np.pi, [0])
            circuit = corrected_circuit
        else:
            # else they're in the "positive" scissors part...
            corrected_circuit = QuantumCircuit(q)
            _, source_reflection, reflection_phase_shift = apply_reflection(
                "reflect XX, ZZ", [0, 0, 0], q)
            _, source_shift, shift_phase_shift = apply_shift("X shift",
                                                             [0, 0, 0], q)

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

        q = circuit.qubits[0].register
        circ = QuantumCircuit(q, global_phase=weyl_decomposition.global_phase)
        # Q: Why am I calling oneq_synth at all?
        circ.compose(self._decomposer1q(weyl_decomposition.K2r), [0],
                     inplace=True)
        circ.compose(self._decomposer1q(weyl_decomposition.K2l), [1],
                     inplace=True)
        circ.compose(circuit, [0, 1], inplace=True)
        circ.compose(self._decomposer1q(weyl_decomposition.K1r), [0],
                     inplace=True)
        circ.compose(self._decomposer1q(weyl_decomposition.K1l), [1],
                     inplace=True)

        return circ
