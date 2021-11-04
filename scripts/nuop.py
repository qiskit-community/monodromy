"""
scripts/nuop.py

This script compares the performance of our synthesis techniques with Lao et
al.'s `NuOp` package:

    https://github.com/prakashmurali/NuOp .

Unhappily, NuOp is not presently formulated as a package, so it's not so easy
to include.  Additionally, it has some artificial constraints written into
constants with function-local definitions — for instance, it won't synthesize
circuits of depth greater than 4 — which we want to override in our comparison.
Accordingly, we've copy/pasted the entire body of their package below.  Search
for "# COMPARISON" to hop to where the script actually begins.
"""

import qiskit

from time import perf_counter

import numpy as np
from scipy.optimize import minimize
from scipy.stats import unitary_group

from qiskit.circuit.library.standard_gates import RZZGate
from qiskit.extensions.unitary import UnitaryGate
from qiskit.quantum_info.synthesis.xx_decompose import XXDecomposer

from monodromy.utilities import epsilon


results_filename = "nuop_stats.dat"
gate_strength = np.pi / 4


#
# nuop/gates_numpy.py
#


def cphase_gate(theta):
    return np.matrix([
        [
            1, 0, 0, 0
        ],
        [
            0, 1, 0, 0
        ],
        [
            0, 0, 1, 0
        ],
        [
            0, 0, 0, np.cos(theta) + 1j * np.sin(theta)
        ]])


def cnot_gate():
    return np.matrix([
        [
            1, 0, 0, 0
        ],
        [
            0, 1, 0, 0
        ],
        [
            0, 0, 0, 1
        ],
        [
            0, 0, 1, 0
        ]])


def fsim_gate(theta, phi):
    return np.matrix([
        [
            1, 0, 0, 0
        ],
        [
            0,
            np.cos(theta),
            -1j * np.sin(theta),
            0
        ],
        [
            0,
            -1j * np.sin(theta),
            np.cos(theta),
            0
        ],
        [
            0, 0, 0, np.cos(phi) - 1j * np.sin(phi)
        ]])


def xy_gate(theta):
    return np.matrix([
        [
            1, 0, 0, 0
        ],
        [
            0,
            np.cos(theta / 2),
            1j * np.sin(theta / 2),
            0
        ],
        [
            0,
            1j * np.sin(theta / 2),
            np.cos(theta / 2),
            0
        ],
        [
            0, 0, 0, 1
        ]
    ])


def cz_gate():
    return np.matrix([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, -1]])


def rzz_unitary(theta):
    return np.array([[np.exp(-1j * theta / 2), 0, 0, 0],
                     [0, np.exp(1j * theta / 2), 0, 0],
                     [0, 0, np.exp(1j * theta / 2), 0],
                     [0, 0, 0, np.exp(-1j * theta / 2)]], dtype=complex)


def get_gate_unitary_qiskit(gate_op):
    # Let's assume all the default unitary matrices in qiskit, which has
    # different endianness from our convention, so we will need to reverse the
    # qubit order when we apply our decomposition pass.
    if isinstance(gate_op, UnitaryGate):
        return gate_op.to_matrix()
    elif isinstance(gate_op, RZZGate):
        return rzz_unitary(gate_op.params[0])
    else:
        return gate_op.to_matrix()


#
# nuop/parallel_two_qubit_gate_decomposition.py
#


class GateTemplate:
    """
    Creates a unitary matrix using a specified two-qubit gate, number of layers
    and single-qubit rotation parameters
    """

    def __init__(self, two_qubit_gate, two_qubit_gate_params):
        """
        two_qubit_gate: a function that returns the numpy matrix for the desired
            gate
        two_qubit_gate_params: inputs to the function e.g., for fsim gates,
            params has fixed theta, phi values
        """
        self.two_qubit_gate = two_qubit_gate
        self.two_qubit_gate_params = two_qubit_gate_params

    def u3_gate(self, theta, phi, lam):
        return np.matrix([
            [
                np.cos(theta / 2),
                -np.exp(1j * lam) * np.sin(theta / 2)
            ],
            [
                np.exp(1j * phi) * np.sin(theta / 2),
                np.exp(1j * (phi + lam)) * np.cos(theta / 2)
            ]])

    def multiply_all(self, matrices):
        product = np.eye(4)
        for i in range(len(matrices)):
            product = np.matmul(matrices[i], product)
        return product

    def u3_layer(self, x):
        u3_1 = self.u3_gate(*x[0:3])
        u3_2 = self.u3_gate(*x[3:6])
        t1 = np.kron(u3_1, u3_2)
        return t1

    def n_layer_unitary(self, n_layers, params):
        """
        n_layers: number of layers desired in the template
        params: list of 1Q gate rotation parameters, specified by the optimizer
        """
        gate_list = []
        idx = 0
        gate_list.append(self.u3_layer(params[idx:idx + 6]))
        idx += 6
        for i in range(n_layers):
            if len(self.two_qubit_gate_params):
                gate_list.append(
                    self.two_qubit_gate(*self.two_qubit_gate_params))
            else:
                gate_list.append(self.two_qubit_gate())
            gate_list.append(self.u3_layer(params[idx:idx + 6]))
            idx += 6
        return self.multiply_all(gate_list)

    def get_num_params(self, n_layers):
        return 6 * (n_layers + 1)


class TwoQubitGateSynthesizer:
    """
    Synthesises a gate implementation for a target unitary, using a specified
    gate template
    """

    def __init__(self, target_unitary, gate_template_obj):
        self.target_unitary = target_unitary
        self.gate_template_obj = gate_template_obj

    def unitary_distance_function(self, A, B):
        # return (1 - np.abs(np.sum(np.multiply(B,np.conj(np.transpose(A))))) / 4)
        # return (1 - (np.abs(np.sum(np.multiply(B,np.conj(A)))))**2+4 / 20)  # quantum volume paper
        return (1 - np.abs(np.sum(np.multiply(B, np.conj(A)))) / 4)

    def make_cost_function(self, n_layers):
        target_unitary = self.target_unitary

        def cost_function(x):
            A = self.gate_template_obj.n_layer_unitary(n_layers, x)
            B = target_unitary
            return self.unitary_distance_function(A, B)

        return cost_function

    def get_num_params(self, n_layers):
        return self.gate_template_obj.get_num_params(n_layers)

    def rand_initialize(self, n_layers):
        params = self.get_num_params(n_layers)
        return [np.pi * 2 * np.random.random() for i in range(params)]

    def solve_instance(self, n_layers, trials):
        self.cost_function = self.make_cost_function(n_layers)
        results = []
        best_idx = 0
        best_val = float('inf')
        for i in range(trials):
            init = self.rand_initialize(n_layers)
            res = minimize(self.cost_function, init, method='BFGS',
                           options={'maxiter': 1000 * 30})
            results.append(res)
            if best_val > res.fun:
                best_val = res.fun
                best_idx = i
        return results[best_idx]

    def optimal_decomposition(self, tol=1e-3, fidelity_2q_gate=1.0,
                              fidelity_1q_gate=[1.0, 1.0]):
        max_num_layers = 10
        cutoff_with_tol = True
        results = []
        best_idx = 0
        best_fidelity = 0

        for i in range(max_num_layers):
            if cutoff_with_tol and best_fidelity > 1.0 - tol:
                break

            # Solve an instance with i+1 layers, doing 1 random trial
            res = self.solve_instance(n_layers=i + 1, trials=1)
            results.append(res)

            # Evaluate the fidelity after adding one layer
            hw_fidelity = ((fidelity_1q_gate[0] * fidelity_1q_gate[1]) ** (2 + i)) * \
                          (fidelity_2q_gate ** (i + 1))
            unitary_fidelity = 1.0 - res.fun
            current_fidelity = hw_fidelity * unitary_fidelity

            # Update if the best_fidelity so far has been 0 (initial case)
            if best_fidelity == 0:
                best_idx = i
                best_fidelity = current_fidelity

            # Update if the current value is smaller than the previous minimum
            if current_fidelity - best_fidelity > tol * 0.1:
                best_idx = i
                best_fidelity = current_fidelity

        return best_idx + 1, results[best_idx], best_fidelity


#
# COMPARISON
#

monodromy_decomposer = XXDecomposer(euler_basis="PSX")
gate_template = GateTemplate(qiskit.circuit.library.RZXGate, [gate_strength])

with open(results_filename, "w") as fh:
    print("monodromy_depth monodromy_time nuop_depth nuop_time")
    fh.write("monodromy_depth monodromy_time nuop_depth nuop_time\n")

for _ in range(1000):
    # generate a random special unitary
    u = unitary_group.rvs(4)
    u /= np.linalg.det(u) ** (1/4)

    # find the best exact and approximate points
    monodromy_time = perf_counter()
    circuit = monodromy_decomposer(
        u, approximate=False, basis_fidelity={gate_strength: 1.0}
    )
    monodromy_time = perf_counter() - monodromy_time
    monodromy_depth = sum([isinstance(datum[0], qiskit.circuit.library.RZXGate)
                           for datum in circuit.data])

    nuop_time = perf_counter()
    nuop_depth, _, nuop_fidelity = TwoQubitGateSynthesizer(
        u, gate_template
    ).optimal_decomposition(
        tol=epsilon, fidelity_2q_gate=1.0, fidelity_1q_gate=[1.0, 1.0]
    )
    nuop_time = perf_counter() - nuop_time

    with open(results_filename, "a") as fh:
        fh.write(f"{monodromy_depth} {monodromy_time} "
                 f"{nuop_depth} {nuop_time}\n")
    print(monodromy_time, monodromy_depth, nuop_time, nuop_depth)
