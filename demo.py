import qiskit

from monodromy.xx_decompose.circuits import *
from monodromy.xx_decompose.defaults import *
from monodromy.xx_decompose.precalculate import calculate_unordered_scipy_coverage_set
from monodromy.coordinates import monodromy_to_positive_canonical_coordinate
from monodromy.coverage import *
from monodromy.static.examples import *
from monodromy.haar import expected_cost
import monodromy.render

import qiskit.quantum_info

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

# use coverage_set to perform a gate decomposition
print("Precomputation...")

precomputed_backsolutions = calculate_unordered_scipy_coverage_set(
    coverage_set, operations, chatty=True
)
alcove_coordinate = [Fraction(3, 8), Fraction(3, 8), Fraction(-1, 8)]
point_polytope = exactly(*alcove_coordinate)
qc = canonical_xx_circuit(
    alcove_coordinate, coverage_set, precomputed_backsolutions, operations
)

# print the circuit for the user to marvel at
print("The following circuit implements the canonical gate at "
      f"{[str(x) for x in alcove_coordinate]}: ", end="")
print((abs(qiskit.quantum_info.Operator(qc).data -
           canonical_matrix(*monodromy_to_positive_canonical_coordinate(*alcove_coordinate)))
       < 0.01).all())

print(qc)

