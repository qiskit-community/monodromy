"""
monodromy/decompose/ordered.py


"""

def calculate_scipy_coverage_set(
        coverage_set: List[CircuitPolytope],
        operations: List[OperationPolytope],
        chatty=False
) -> List[CircuitPolytope]:
    """
    Precalculates a set of backsolution polytopes associated to `covering_set`
    and `operations`.

    Used as efficient input to `scipy_decomposition_hops` below.
    """
    coordinates = {
        "a": [0, 1, 2, 3],
        "b": [0, 4, 5, 6],
        "c": [0, 7, 8, 9],
    }

    inflated_operation_polytope = prereduce_operation_polytopes(
        operations=operations,
        target_coordinate="a",
        background_polytope=Polytope(convex_subpolytopes=[
            # equate first source and first target coordinates
            ConvexPolytope(inequalities=[
                [0,  1,  1, 0, 0, 0, 0, -1, -1, 0],
                [0, -1, -1, 0, 0, 0, 0,  1,  1, 0],
            ]),
            # equate first source and second target coordinates
            ConvexPolytope(inequalities=[
                [0,  1,  1, 0, 0, 0, 0, -1, 0, -1],
                [0, -1, -1, 0, 0, 0, 0,  1, 0,  1],
            ]),
            # equate first source and third target coordinates
            ConvexPolytope(inequalities=[
                [0,  1,  1, 0, 0, 0, 0, 0, -1, -1],
                [0, -1, -1, 0, 0, 0, 0, 0,  1,  1],
            ]),
            # equate second source and second target coordinates
            ConvexPolytope(inequalities=[
                [0,  1, 0,  1, 0, 0, 0, -1, 0, -1],
                [0, -1, 0, -1, 0, 0, 0,  1, 0,  1],
            ]),
            # equate second source and third target coordinates
            ConvexPolytope(inequalities=[
                [0,  1, 0,  1, 0, 0, 0, 0, -1, -1],
                [0, -1, 0, -1, 0, 0, 0, 0,  1,  1],
            ]),
            # equate third source and third target coordinates
            ConvexPolytope(inequalities=[
                [0, 0,  1,  1, 0, 0, 0, 0, -1, -1],
                [0, 0, -1, -1, 0, 0, 0, 0,  1,  1],
            ])]),
        chatty=chatty,
    )

    scipy_coverage_set = []

    if chatty:
        print("Working on scipy precalculation.")
    for operation_polytope in coverage_set:
        if 0 == len(operation_polytope.operations):
            continue

        if chatty:
            print(f"Working on {'.'.join(operation_polytope.operations)}...")

        ancestor_polytope = next(
            (polytope for polytope in coverage_set
             if polytope.operations == operation_polytope.operations[:-1]),
            exactly(0, 0, 0))

        backsolution_polytope = inflated_operation_polytope[
            operation_polytope.operations[-1]
        ]

        # also impose whatever constraints we were given besides
        backsolution_polytope = backsolution_polytope.intersect(
            cylinderize(
                ancestor_polytope,
                coordinates["a"],
                parent_dimension=7
            )
        )
        backsolution_polytope = backsolution_polytope.reduce()

        scipy_coverage_set.append(CircuitPolytope(
            convex_subpolytopes=backsolution_polytope.convex_subpolytopes,
            cost=operation_polytope.cost,
            operations=operation_polytope.operations,
        ))

    return scipy_coverage_set


def scipy_decomposition_hops(
        coverage_set: List[CircuitPolytope],
        scipy_coverage_set: List[CircuitPolytope],
        target_polytope: PolytopeData
):
    """
    Fixing a `coverage_set` and a `scipy_coverage_set`, finds a minimal
    decomposition for a canonical interaction in `target_polytope` into a
    sequence of operations linking the polytopes in the coverage sets, together
    with specific intermediate canonical points linked by them.

    Returns a list of tuples of shape (source vertex, operation, target vertex),
    so that each target vertex is accessible from its source vertex by
    application of the operation, each target vertex matches its next source
    vertex, the original source vertex corresponds to the identity, and the
    last target lies in `target_polytope`.

    NOTE: `scipy_coverage_set` is extracted from `coverage_set` using
          `calculate_scipy_coverage_set` above.
    """
    decomposition = []  # retval
    working_polytope = None

    # NOTE: if `target_polytope` were an actual point, could use .has_element
    best_cost = float("inf")
    for polytope in coverage_set:
        if polytope.cost < best_cost:
            for convex_subpolytope in \
                    polytope.intersect(target_polytope).convex_subpolytopes:
                solution = scipy_get_random_vertex(convex_subpolytope)

                if solution.success:
                    working_polytope = polytope
                    best_cost = polytope.cost
                    break

    if working_polytope is None:
        raise ValueError(f"{target_polytope} not contained in coverage set.")

    working_operations = working_polytope.operations

    # if this polytope corresponds to the empty operation, we're done.
    while 0 < len(working_operations):
        backsolution_polytope = None
        solution = None

        for polytope in scipy_coverage_set:
            if polytope.operations == working_operations:
                backsolution_polytope = polytope
                break
        if backsolution_polytope is None:
            raise NoBacksolution()

        # impose the target constraints, which sit on "b"
        # (really on "c", but "b" has already been projected off)
        intersected_polytope = PolytopeData(convex_subpolytopes=[])
        for cp in backsolution_polytope.convex_subpolytopes:
            intersected_polytope.convex_subpolytopes.append(
                ConvexPolytopeData(
                    inequalities=[
                        *cp.inequalities,
                        *[[ineq[0], 0, 0, 0, ineq[1], ineq[2], ineq[3]]
                          for ineq in
                          target_polytope.convex_subpolytopes[0].inequalities]
                    ],
                    equalities=cp.equalities,
                )
            )
        backsolution_polytope = intersected_polytope

        # walk over the backsolution polytopes, try to find one that's solvable
        shuffle(backsolution_polytope.convex_subpolytopes)
        for convex_subpolytope in backsolution_polytope.convex_subpolytopes:
            solution = scipy_get_random_vertex(convex_subpolytope)
            if solution.success:
                break

        if solution is None or not solution.success:
            raise NoBacksolution()

        # a/k/a decomposition.push
        decomposition.insert(
            0,
            (solution.x[:3], working_operations[-1], solution.x[-3:])
        )
        # NOTE: using `exactly` here causes an infinite loop.
        target_polytope = nearly(*solution.x[:3])
        working_operations = working_operations[:-1]

    return decomposition
