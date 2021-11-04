import qiskit.quantum_info
from qiskit.quantum_info.synthesis.xx_decompose import XXDecomposer

import numpy as np
from scipy.stats import unitary_group

from monodromy.coverage import *
from monodromy.static.examples import *
from monodromy.haar import expected_cost
import monodromy.render


def default_zx_operation_cost(
        strength: Fraction,
        # note: Isaac reports this value in percent per degree
        scale_factor: float = (64 * 90) / (10000 * 100),
        # first component: 2Q invocation cost; second component: local cost
        offset: float = 909 / (10000 * 100) + 1 / 1000,
):
    """
    A sample fidelity cost model, extracted from experiment, for ZX operations.
    """
    return strength * scale_factor + offset


def get_zx_operations(strengths: Dict[Fraction, float]) \
        -> List[CircuitPolytope]:
    """
    Converts a dictionary mapping fractional CX `strengths` to fidelities to the
    corresponding list of `OperationPolytope`s.
    """
    operations = []

    for strength, fidelity in strengths.items():
        operations.append(CircuitPolytope(
            operations=[f"rzx(pi/2 * {strength})"],
            cost=fidelity,
            convex_subpolytopes=exactly(
                strength / 4, strength / 4, -strength / 4,
            ).convex_subpolytopes,
        ))

    return operations


operations = get_zx_operations({
    frac: default_zx_operation_cost(frac)
    for frac in [Fraction(1), Fraction(1, 2), Fraction(1, 3)]
})

# build the set of covering polytopes
print("==== Working to build a set of covering polytopes ====")
coverage_set = build_coverage_set(operations, chatty=True)

# print it out for user inspection
print("==== Done. Here's what we found: ====")
print_coverage_set(coverage_set)

print("==== Haar volumes ====")
print(f"Haar-expectation cost: {expected_cost(coverage_set, chatty=True)}")

# flex the rendering code
print("==== Render these in Mathematica: =====")
print(monodromy.render.polytopes_to_mathematica(coverage_set))

# perform a gate decomposition
print("==== Compiling a single Haar-random gate into CX, CX/2, CX/3 ====")
# generate a random special unitary
u = unitary_group.rvs(4)
u /= np.linalg.det(u) ** (1 / 4)

# decompose into CX, CX/2, and CX/3
monodromy_decomposer = XXDecomposer(euler_basis="PSX")
circuit = monodromy_decomposer(u, approximate=False)

with np.printoptions(precision=4, suppress=True):
    print(u)
    print(qiskit.quantum_info.Operator(circuit).data)
    print(f"=== {(abs(u - qiskit.quantum_info.Operator(circuit).data) < 1e-1).all()} ===")
    print(circuit)
