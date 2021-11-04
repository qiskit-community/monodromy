from time import perf_counter

import monodromy

from monodromy.coordinates import monodromy_alcove, monodromy_alcove_c2, monodromy_to_positive_canonical_polytope, rho_reflect
from monodromy.elimination import cylinderize, project
from monodromy.polytopes import ConvexPolytope, Polytope
from monodromy.static import qlr_polytope

from itertools import count

biswas_relations = (qlr_polytope
    # enlarge to the pu_4 version of the QLR relations
    .union(rho_reflect(qlr_polytope, [0, 7, 8, 9]))
    # constrain in- and out-coordinates to the appropriate alcove
    .intersect(cylinderize(monodromy_alcove, [0, 1, 2, 3], 10))
    .intersect(cylinderize(monodromy_alcove_c2, [0, 7, 8, 9], 10))
)

# constrain interaction coordinates to be of XX-type
biswas_relations = biswas_relations.intersect(Polytope(convex_subpolytopes=[
    ConvexPolytope(
        inequalities=[[1, 0, 0, 0, -4, 0, 0, 0, 0, 0]],
        equalities=[
            [0, 0, 0, 0, 1, -1, 0, 0, 0, 0],  # x1 == x2
            [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],  # x2 == -x3
        ]
    )
]))

# switch to canonical coordinates
biswas_relations = monodromy_to_positive_canonical_polytope(
    biswas_relations, coordinates=[0, 1, 2, 3])
biswas_relations = monodromy_to_positive_canonical_polytope(
    biswas_relations, coordinates=[0, 4, 5, 6])
biswas_relations = monodromy_to_positive_canonical_polytope(
    biswas_relations, coordinates=[0, 7, 8, 9])

# reduce the biswas relations to have following coordinates:
# k a1 a2 a3 beta b1 b2 b3
biswas_relations = biswas_relations.reduce()
biswas_relations = project(biswas_relations, 6).reduce()
biswas_relations = project(biswas_relations, 5).reduce()
    
xx_polytope = monodromy.static.examples.identity_polytope
for n in count(1):
    print(f"Working on an XX interaction sequence of length {n}...")
    
    # inflate xx_polytope from [*a_coords, *interaction_coords] to [*a_coords, *b_coords, *interaction_coords, beta]
    xx_polytope = cylinderize(
        xx_polytope,
        coordinate_map=[0, 1, 2, 3] + list(range(7, 7 + (n - 1))),
        parent_dimension=1 + 3 + 3 + n,
    ).intersect(cylinderize(
        biswas_relations,
        coordinate_map=[0, 1, 2, 3, -1, 4, 5, 6],
        parent_dimension=1 + 3 + 3 + n,
    ))
    
    # project away the old a-coordinates
    start_time = perf_counter()
    print("Working on the reduction 1/3...", end="")
    xx_polytope = project(xx_polytope, 3).reduce()
    print(f" done.  Took {perf_counter() - start_time} seconds.")

    start_time = perf_counter()
    print("Working on the reduction 2/3...", end="")
    xx_polytope = project(xx_polytope, 2).reduce()
    print(f" done.  Took {perf_counter() - start_time} seconds.")

    start_time = perf_counter()
    print("Working on the reduction 3/3...", end="")
    xx_polytope = project(xx_polytope, 1).reduce()
    print(f" done.  Took {perf_counter() - start_time} seconds.")
    
    # now the old c-coordinates are sitting where the a-coordinates were!
    print("The first three coordinates are the canonical coordinates CAN(x1, x2, x3).")
    print("The remaining coordinates x4, ..., xk are the XX interaction strengths.")
    print(xx_polytope)
