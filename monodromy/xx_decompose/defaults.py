"""
monodromy/xx_decompose/defaults.py

Contains some default data which one can use to prime the other methods in this
submodule.
"""

from fractions import Fraction
from functools import reduce
from typing import Dict, List

import numpy as np
import qiskit
from qiskit.circuit.library import HGate

from .circuits import OperationPolytope
from ..static.matrices import canonical_matrix
from ..static.examples import exactly


default_zx_names = {
    Fraction(1, 1): "cx",
    Fraction(1, 2): "sqrtCX",
    Fraction(1, 3): "cbrtCX",
}


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
        -> List[OperationPolytope]:
    """
    Converts a dictionary mapping fractional CX `strengths` to fidelities to the
    corresponding list of `OperationPolytope`s.
    """

    q = qiskit.QuantumRegister(2)
    operations = []

    for strength, fidelity in strengths.items():
        label = default_zx_names.get(strength, f"ZX^({strength})")

        cx_circuit = qiskit.QuantumCircuit(q)
        cx_circuit.h(q[0])
        cx_circuit.append(qiskit.extensions.UnitaryGate(
            # sandwich with Hs on one coordinate
            reduce(np.dot, [
                np.kron(np.eye(2), HGate().to_matrix()),
                canonical_matrix(strength * np.pi/4),
                np.kron(np.eye(2), HGate().to_matrix()),
            ]),
            label=label
        ), q)
        cx_circuit.h(q[0])

        operations.append(OperationPolytope(
            operations=[label],
            cost=fidelity,
            convex_subpolytopes=exactly(
                    strength / 4, strength / 4, -strength / 4,
                ).convex_subpolytopes,
            canonical_circuit=cx_circuit,
        ))

    return operations
