"""
Explicit interference inequalities.  Seems buggy, and possibly insufficient for
protecting backsolution polytopes from backtracking.
"""


# a typical pair of trigonometric constraints looks like:
#
# sin(pi/2 ((x - y) + (x' - y') + 2 i)) \      in pi/2 * (                pos [0,   2])
# sin(pi/2 ((x - y) - (x' - y') + 2 i)) \      in pi/2 * (neg [-1/2, 0] u pos [0, 3/2])
# sin(pi/2 ((x - y) + (x' - y') - 2 i)) \      in pi/2 * (neg [  -1, 0] u pos [0,   1])
# sin(pi/2 ((x - y) - (x' - y') - 2 i)) <= 0   in pi/2 * (neg [-3/2, 0] u pos [0, 1/2])
#
# AND
#
# sin(pi/2 ((x + y + 2z) + (x' + y' + 2z') + 2 i)) \     in pi/2 * (neg [ 2, 3] u pos [0, 2])
# sin(pi/2 ((x + y + 2z) - (x' + y' + 2z') + 2 i)) \     in pi/2 * (neg [-1, 0] u pos [0, 2])
# sin(pi/2 ((x + y + 2z) - (x' + y' + 2z') - 2 i)) \     in pi/2 * (neg [-2, 0] u pos [0, 1])
# sin(pi/2 ((x + y + 2z) + (x' + y' + 2z') - 2 i)) <= 0  in pi/2 * (neg [-1, 0] u pos [0, 2])
#
# the arguments are linear functionals on the polytope, so they're extremized at the vertices.
# the monomials are constrained to [0, 1/2]; the binomials to [0, 1/2]; the trinomials to [0, 1].
#
# inequalities come in conjunctive pairs.
# each pair member comes in 8-fold disjunctive families.
# each family member comes in triplets: an a component, a signed b component, and a signed c component.

destructive_range_tables = {
    # c, b
    ( 1,  1): {
        1:  (   0,   2),
        -1: None,
    },
    (-1,  1): {
        1:  (   0, 3/2),
        -1: (-1/2,   0),
    },
    ( 1, -1): {
        1:  (   0,   1),
        -1: (  -1,   0),
    },
    (-1, -1): {
        1:  (   0,   2),
        -1: (  -1,   0),
    },
}

constructive_range_tables = {
    # c, b
    ( 1,  1): {
        1:  ( 0, 2),
        -1: ( 2, 3),
    },
    (-1,  1): {
        1:  ( 0, 2),
        -1: (-1, 0),
    },
    (-1, -1): {
        1:  ( 0, 1),
        -1: (-2, 0),
    },
    ( 1, -1): {
        1:  ( 0, 2),
        -1: (-1, 0),
    },
}

constraint_polytopes = [
    Polytope(convex_subpolytopes=[
        ConvexPolytope(inequalities=fractionify([
            [0,  1,  1,  0, 0, 0, 0, -1, -1,  0],
            [0, -1, -1,  0, 0, 0, 0,  1,  1,  0],
        ]))
    ]),
    Polytope(convex_subpolytopes=[
        ConvexPolytope(inequalities=fractionify([
            [0,  1,  0,  1, 0, 0, 0, -1,  0, -1],
            [0, -1,  0, -1, 0, 0, 0,  1,  0,  1],
        ]))
    ]),
    Polytope(convex_subpolytopes=[
        ConvexPolytope(inequalities=fractionify([
            [0,  0,  1,  1, 0, 0, 0,  0, -1, -1],
            [0,  0, -1, -1, 0, 0, 0,  0,  1,  1],
        ]))
    ]),
]


from itertools import product

def generate_xxyy_to_xx_xxyy_extra_polytopes():
    signs = [1, -1]
    extra_polytope = empty_polytope
    for index in [0, 1, 2]:
        constraint_polytope = constraint_polytopes[index]
        constructive_polytope = empty_polytope
        destructive_polytope  = empty_polytope

        # constructive
        for negative_hot in [True, False]:
            for hot_pair in product(signs, signs):
                inequalities = []

                for pair in product(signs, signs):
                    c_sign, b_sign = pair
                    inequality = [0] * 10
                    inequality[1], inequality[2], inequality[3] = 1, 1, 1
                    inequality[1 + index] += 1
                    inequality[7], inequality[8], inequality[9] = 1 * c_sign, 1 * c_sign, 1 * c_sign
                    inequality[7 + index] += c_sign
                    inequality[4] = 2 * b_sign

                    pair_sign = 1 if ((pair == hot_pair) != negative_hot) else -1
                    lower_bound, upper_bound = constructive_range_tables[pair][-1 if pair == hot_pair else 1]
                    inequalities.append([-lower_bound] + inequality[1:])
                    inequalities.append([upper_bound] + [-x for x in inequality[1:]])

                constructive_polytope = constructive_polytope.union(Polytope(convex_subpolytopes=[
                    ConvexPolytope(inequalities=fractionify(inequalities))
                ]))

        # destructive
        positive_index, negative_index = [x for x in [0, 1, 2] if x != index]
        for negative_hot in [True, False]:
            for hot_pair in product(signs, signs):
                inequalities = []

                for pair in product(signs, signs):
                    c_sign, b_sign = pair
                    inequality = [0] * 10
                    inequality[1 + positive_index] = 1
                    inequality[1 + negative_index] = -1
                    inequality[7 + positive_index] = c_sign
                    inequality[7 + negative_index] = -c_sign
                    inequality[4] = 2 * b_sign

                    pair_sign = 1 if ((pair == hot_pair) != negative_hot) else -1
                    bound_range = destructive_range_tables[pair][-1 if pair == hot_pair else 1]
                    if bound_range is None:
                        continue
                    lower_bound, upper_bound = bound_range
                    inequalities.append([-lower_bound] + inequality[1:])
                    inequalities.append([upper_bound] + [-x for x in inequality[1:]])

                destructive_polytope = destructive_polytope.union(Polytope(convex_subpolytopes=[
                    ConvexPolytope(inequalities=fractionify(inequalities))
                ]))

        extra_polytope = extra_polytope.union(
            constraint_polytope  # .intersect(
#                 constructive_polytope.intersect(
#                     destructive_polytope
#                 )
#             )
        )

    return extra_polytope


def decomposition_hops(
        coverage_set: List[GatePolytope],
        operations: List[GatePolytope],
        target_polytope: Polytope
):
    """
    Fixing a `coverage_set` and a set of `operations`, finds a minimal
    decomposition for a canonical interaction in `target_polytope` into a
    sequence of operations drawn from `operations`, together with specific
    intermediate canonical points linked by them.

    Returns a list of tuples of shape (source vertex, operation, target vertex),
    so that each target vertex is accessible from its source vertex by
    application of the operation, each target vertex matches its next source
    vertex, the original source vertex corresponds to the identity, and the
    last target lies in `target_polytope`.
    """
    decomposition = []

    working_polytope = cheapest_container(coverage_set, target_polytope)
    cost = working_polytope.cost

    if working_polytope is None:
        raise ValueError(f"{target_polytope} not contained in coverage set.")

    # if this polytope corresponds to the empty operation, we're done.
    while 0 != len(working_polytope.operations):
        source_vertex, operation, target_vertex, working_polytope = decomposition_hop(
            coverage_set, operations, working_polytope, target_polytope
        )

        # a/k/a decomposition.push
        decomposition.insert(0, (source_vertex, operation, target_vertex))
        target_polytope = exactly(*source_vertex)

    return decomposition, cost


decomposition = decomposition_hops(coverage_set, operations, exactly(*target))
print(decomposition)
xx_circuit_from_decomposition(
    decomposition, operations
)
