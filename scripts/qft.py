"""
scripts/qft.py

Count the XX interactions in synthesized QFT circuits of various sizes.

NOTE: Runs indefinitely.
"""

import qiskit
import numpy as np
from collections import defaultdict
from itertools import count

for qubit_count in count(2):
    qc = qiskit.circuit.library.QFT(qubit_count)
    cx_counts = defaultdict(lambda: 0)
    for gate, _, _ in qiskit.transpile(
            qc, basis_gates=['u3', 'cx'], translation_method='synthesis'
    ).data:
        if isinstance(gate, qiskit.circuit.library.CXGate):
            cx_counts[np.pi] += 1
        elif isinstance(gate, qiskit.circuit.library.RZXGate):
            cx_counts[gate.params[0]] += 1

    rzx_counts = defaultdict(lambda: 0)
    for gate, _, _ in qiskit.transpile(
            qc, basis_gates=['u3', 'rzx'], translation_method='synthesis'
    ).data:
        if isinstance(gate, qiskit.circuit.library.CXGate):
            rzx_counts[np.pi] += 1
        elif isinstance(gate, qiskit.circuit.library.RZXGate):
            rzx_counts[gate.params[0]] += 1

    print(f"At qubit count {qubit_count}:")
    print(f"CX counts: {cx_counts}")
    print(f"CX, CX/2, CX/3 counts: {rzx_counts}")
