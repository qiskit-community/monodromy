from monodromy.coverage import *
from monodromy.examples import *
import monodromy.render

operations = [
    GatePolytope(
        convex_subpolytopes=thirdCX_polytope.convex_subpolytopes,
        cost=Fraction(1, 3),
        operations=["thirdCX"],
    ),
    GatePolytope(
        convex_subpolytopes=sqrtCX_polytope.convex_subpolytopes,
        cost=Fraction(1, 2),
        operations=["sqrtCX"],
    ),
]

# build the set of covering polytopes
print("==== Working to build a set of covering polytopes ====")
coverage_set = build_coverage_set(operations, chatty=True)
# print it out for user inspection
print("==== Done. Here's what we found: ====")
print_coverage_set(coverage_set)
# flex the rendering code
print("==== Render these in Mathematica: =====")
print(monodromy.render.polytopes_to_mathematica(coverage_set))
