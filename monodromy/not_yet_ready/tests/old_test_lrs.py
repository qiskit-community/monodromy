## TODO: actually convert this to the form of a unit test
## TODO: not all of these belong in this file, I'm just dumping them here for now.

inequalities = fractionify([
    # c + xc*x + y*yc + z*zc + ... >= 0
    # c  xc  yc  zc ...
    [ 1,  1,  0,  0,],
    [ 1,  0,  1,  0,],
    [ 1,  0,  0,  1,],
    [ 1, -1,  0,  0,],
    [ 1,  0, -1,  0,],
    [ 1,  0,  0, -1,],
#     [ 0,  1,  1,  1,],
])

vertices = fractionify([
    # 1   x   y   z   ...
    [ 1,  1,  1,  1],
    [ 1,  1,  1, -1],
    [ 1,  1, -1,  1],
    [ 1,  1, -1, -1],
    [ 1, -1,  1,  1],
    [ 1, -1,  1, -1],
    [ 1, -1, -1,  1],
    [ 1, -1, -1, -1],
])

print(inequalities_from_vertices(vertices))
print(vertices_from_inequalities(inequalities))
for w, x, y, z in vertices_from_inequalities(inequalities)["vertices"]:
    print(f"{w}, {x}, {y}, {z}")

a2_vertices = vertices_from_inequalities(a2_inequalities)["vertices"]

print("The vertices of A_{C_2} are located at:")
print_vertex_table(a2_vertices)
ac2_volume = inequalities_from_vertices(a2_vertices)["volume"]
print(f"Volume: {ac2_volume}")


## Example: Compute P^2_CZ

## Pi CZ = (1/4, 1/4, -1/4, -1/4).
## impose 3x pairs of inequalities to get 3x exact equalities.

CZ_inequalities = equality_inequalities(1/4, 1/4, -1/4)

P2CZ = intersect_and_project_to_c(CZ_inequalities, CZ_inequalities)
print("P^2_CZ calculated from first principles:")
print_inequality_table(P2CZ)

print("A_{C_2} intersected with x2 == -x3:")
print_inequality_table(reduce_inequalities(a2_inequalities + [
    [0,  0,  1,  1,],  # x2 == -x3
    [0,  0, -1, -1,],
]))

# correct answer for P^2_CZ: a4 = -a1, a3 = -a2, 1/2 >= a1 >= a2 >= 0.
# generally, remember to apply rho to b to get the "other" convex polytope.
# don't have to worry about this with CZ because rho(Pi(CZ)) = Pi(CZ) is its own rotation.

# linearity k l1 ... lk  # the k lines l1, ..., lk are equations rather than inequalities


## Example: Compute P^3_CZ
## Plugs in the P^2_CZ calculation above!

P3CZ = intersect_and_project_to_c(P2CZ, CZ_inequalities)
print_inequality_table(P3CZ)
coverage_volume = inequalities_from_vertices(vertices_from_inequalities(P3CZ)['vertices'])['volume'] / ac2_volume
print(f"Coverage volume: {100*float(coverage_volume):.2f}%")

# correct answer for P^2_CZ: a4 = -a1, a3 = -a2, 1/2 >= a1 >= a2 >= 0.
# generally, remember to apply rho to b to get the "other" convex polytope.
# don't have to worry about this with CZ because rho(Pi(CZ)) = Pi(CZ) is its own rotation.

# linearity k l1 ... lk  # the k lines l1, ..., lk are equations rather than inequalities


## calculate low-cost polytopes for a given set of operations
operations = [
    GatePolytope(
        operations=["sqrtCZ"],
        cost=Fraction(1, 2),
        inequalities=[equality_inequalities(Fraction(1,  8), Fraction(1,  8), Fraction(-1,  8))],
        volume=None,
    ),
    GatePolytope(
        operations=["thirdCZ"],
        cost=Fraction(1, 3),
        inequalities=[equality_inequalities(Fraction(1, 12), Fraction(1, 12), Fraction(-1, 12))],
        volume=None,
    ),
]

# TODO: make the alcove_c2 volume a unit test?  1/48?
