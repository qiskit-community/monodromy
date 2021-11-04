"""
scripts/single_circuit.py

Calculate the polytope of canonical coordinates accessible to a fixed circuit
template of the form

    local0 * gates[0] * local1 * ... * localn * gates[n] * local(n+1).
"""

from qiskit.circuit.library import RZXGate
import numpy as np

gates = [RZXGate(np.pi/3), RZXGate(np.pi/3), RZXGate(np.pi/3), RZXGate(np.pi/3)]


from fractions import Fraction
from monodromy.coordinates import monodromy_to_positive_canonical_polytope, \
    positive_canonical_alcove_c2, unitary_to_monodromy_coordinate
from monodromy.coverage import deduce_qlr_consequences
from monodromy.static.examples import exactly, identity_polytope, \
    everything_polytope


circuit_polytope = identity_polytope


for gate in gates:
    b_polytope = exactly(
        *(Fraction(x).limit_denominator(10_000)
          for x in unitary_to_monodromy_coordinate(gate.to_matrix())[:-1])
    )
    circuit_polytope = deduce_qlr_consequences(
        target="c",
        a_polytope=circuit_polytope,
        b_polytope=b_polytope,
        c_polytope=everything_polytope
    )

print(monodromy_to_positive_canonical_polytope(circuit_polytope))
print(f"{monodromy_to_positive_canonical_polytope(circuit_polytope).volume} vs "
      f"{positive_canonical_alcove_c2.volume}")
