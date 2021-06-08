"""
monodromy/attic/sample.py


"""


# TODO: In rare cases this generates an empty range, and I'm not sure why.
# TODO: This doesn't sample uniformly; it treats the pushed-forward uniform
#       distribution along a projection as uniform, which is false.
def random_alcove_coordinate(denominator=100):
    first_numerator = randint(0, denominator // 2)
    second_numerator = randint(
        max(-first_numerator // 3, -(denominator - 2 * first_numerator) // 2),
        min(first_numerator, (3 * denominator - 6 * first_numerator) // 2)
    )
    third_numerator = randint(
        max(-(first_numerator + second_numerator) // 2 + 1,
            -(denominator - 2 * first_numerator) // 2),
        min(second_numerator + 1,
            denominator - 2 * first_numerator - second_numerator)
    )

    return (
        Fraction(first_numerator, denominator),
        Fraction(second_numerator, denominator),
        Fraction(third_numerator, denominator),
    )


def sample_irreducible_circuit(coverage_set, operations, target_gate_polytope):
    """
    Produces a randomly generated circuit of the prescribed type which cannot
    be rewritten into a circuit of lower cost.
    """

    operation_gates = {
        operation.operations[0]:
            qiskit.extensions.UnitaryGate(
                canonical_matrix(*alcove_to_positive_canonical_coordinate(
                    *operation.convex_subpolytopes[0].vertices[0])),
                label=f"CAN({operation.operations[0]})"
            )
        for operation in operations
    }

    q = qiskit.QuantumRegister(2, 'q')

    while True:
        qc = qiskit.QuantumCircuit(q)

        qc.u3(qubit=0,
              theta=np.random.uniform(2 * np.pi),
              phi=np.random.uniform(2 * np.pi),
              lam=np.random.uniform(2 * np.pi))
        qc.u3(qubit=1,
              theta=np.random.uniform(2 * np.pi),
              phi=np.random.uniform(2 * np.pi),
              lam=np.random.uniform(2 * np.pi))
        for operation in target_gate_polytope.operations:
            qc.append(operation_gates[operation], q)
            qc.u3(qubit=0,
                  theta=np.random.uniform(2 * np.pi),
                  phi=np.random.uniform(2 * np.pi),
                  lam=np.random.uniform(2 * np.pi))
            qc.u3(qubit=1,
                  theta=np.random.uniform(2 * np.pi),
                  phi=np.random.uniform(2 * np.pi),
                  lam=np.random.uniform(2 * np.pi))

        point = unitary_to_alcove_coordinate(
            qiskit.quantum_info.Operator(qc).data
        )
        computed_depth = cheapest_container(
            coverage_set, exactly(*point[:-1])
        ).cost

        if computed_depth >= target_gate_polytope.cost:
            break

    return qc
