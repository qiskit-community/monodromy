"""
monodromy/xx_decompose/qiskit.py

Staging ground for a QISKit Terra compilation pass which emits ZX circuits.
"""

from fractions import Fraction
from math import cos, sin, sqrt

import numpy as np

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import RZXGate
from qiskit.extensions import UnitaryGate
from qiskit.transpiler.passes.optimization import Optimize1qGatesDecomposition
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info.synthesis import OneQubitEulerDecomposer
from qiskit.quantum_info.synthesis.two_qubit_decompose import \
    TwoQubitWeylDecomposition

from .circuits import apply_reflection, apply_shift, \
    xx_circuit_from_decomposition
from ..coordinates import alcove_to_positive_canonical_coordinate, \
    unitary_to_alcove_coordinate
from .defaults import get_zx_operations, default_data
from ..io.inflate import filter_scipy_data
from .paths import NoBacksolution, scipy_unordered_decomposition_hops
from .scipy import polyhedron_has_element, nearly, optimize_over_polytope
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
    """

    def __init__(self, operations, coverage_set, precomputed_backsolutions,
                 euler_basis="U3"):
        self.operations, self.coverage_set, self.precomputed_backsolutions = \
            operations, coverage_set, precomputed_backsolutions

        self._decomposer1q = OneQubitEulerDecomposer(euler_basis)
        self._optimize1q = Optimize1qGatesDecomposition(euler_basis)

        self.gate = RZXGate(np.pi/2)

    @staticmethod
    def _build_objective(target):
        """
        Constructs the functional which measures the trace distance to `target`.
        """
        a0, b0, c0 = alcove_to_positive_canonical_coordinate(*target)

        def objective(array):
            a, b, c = alcove_to_positive_canonical_coordinate(*array)
            return -1 / 20 * (4 + 16 * sqrt(
                cos(a0 - a) ** 2 * cos(b0 - b) ** 2 * cos(c0 - c) ** 2 +
                sin(a0 - a) ** 2 * sin(b0 - b) ** 2 * sin(c0 - c) ** 2
            ))

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

    def _best_decomposition(self, target,
                            approximate=True,
                            chatty=False):  # -> (coord, cost, op'ns)
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
                objective_dict = self._build_objective(target)
                solution = optimize_over_polytope(
                    objective_dict["objective"], convex_polytope,
                    jacobian=objective_dict["jacobian"],
                    # hessian=objective_dict["hessian"]
                )
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
            if overall_exact_cost is None and polyhedron_has_element(
                    gate_polytope, target
            ):
                overall_exact_cost = gate_polytope.cost
                overall_exact_operations = gate_polytope.operations
                if not approximate:
                    return target, overall_exact_cost, overall_exact_operations
                break

        if approximate:
            return overall_best_point, overall_best_cost, overall_best_operations
        else:
            raise ValueError("Failed to find a match.")

    def num_basis_gates(self, unitary, approximate=True, chatty=False):
        """
        Counts the number of gates that would be emitted during re-synthesis.

        Used by ConsolidateBlocks.
        """
        target = unitary_to_alcove_coordinate(unitary)[:3]
        _, _, overall_best_operations = \
            self._best_decomposition(target,
                                     approximate=approximate,
                                     chatty=chatty)
        return len(overall_best_operations)

    def decompose_1q(self, circuit):
        """
        Gather the one-qubit substrings in a two-qubit circuit and apply the
        local decomposer.
        """
        circ_0 = QuantumCircuit(1)
        circ_1 = QuantumCircuit(1)
        output_circuit = QuantumCircuit(2)

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
        overall_best_point, overall_best_cost, overall_best_operations = \
            self._best_decomposition(target, approximate=approximate, chatty=chatty)

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
                    overall_best_point
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

        circ.append(UnitaryGate(weyl_decomposition.K2r), [0])
        circ.append(UnitaryGate(weyl_decomposition.K2l), [1])
        circ.compose(circuit, [0, 1], inplace=True)
        circ.append(UnitaryGate(weyl_decomposition.K1r), [0])
        circ.append(UnitaryGate(weyl_decomposition.K1l), [1])

        return self.decompose_1q(circ)


def monodromy_decomposer_from_approximation_degree(
        euler_basis="U3",
        approximation_degree=1.0,
):
    operations = get_zx_operations({
        # This is kind of gross.
        # * 1.e-10 is a small, below-hardware-granularity threshold that permits
        #   the sorting mechanism in inflate_scipy_data to still function.
        # * 12/5 * 1/k means that CX^1/3 is taken to have 0.8 error, which is
        #   the maximum amount possible with gate average fidelity.
        # * (1 - approximation_degree) interpolates between minimum assumed
        #   error (80% at 1.0) and maximum assumed error (0% at 0.0).
        Fraction(1, k): 1.e-10 + 12 / 5 * 1 / k * \
                                (1.e-10 + (1 - 1.e-10) * (1 - approximation_degree))
        for k in [1, 2, 3]
    })
    inflated_data = filter_scipy_data(operations, **default_data, chatty=False)
    return MonodromyZXDecomposer(**inflated_data, euler_basis=euler_basis)
