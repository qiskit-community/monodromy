"""
Dumping ground for extraneous material related to cost estimation and Mathematica.
"""

from subprocess import Popen, PIPE
from time import perf_counter
from warnings import warn

def dump_polytopes(coverage_set):
    output = ""
    output += "polytopeData = {"
    for n, gate_polytope in enumerate(coverage_set):
        output += "{"
        output += f"{float(gate_polytope.cost)}, "
        output += f'"{".".join(gate_polytope.operations)}", '
        for i, polytope in enumerate(gate_polytope.convex_subpolytopes):
            output += "{"
            output += "{"
            for j, inequality in enumerate(polytope.inequalities):
                output += "{"
                output += f"{inequality[0]}, {inequality[1]}, {inequality[2]}, {inequality[3]}"
                if 1 + j != len(polytope.inequalities):
                    output += "}, "
                else:
                    output += "}"
            output += "}, {"
            for j, equality in enumerate(polytope.equalities):
                output += "{"
                output += f"{equality[0]}, {equality[1]}, {equality[2]}, {equality[3]}"
                if 1 + j != len(polytope.equalities):
                    output += "}, "
                else:
                    output += "}"
            output += "}"
            if 1 + i != len(gate_polytope.convex_subpolytopes):
                output += "}, "
            else:
                output += "}"
        if 1 + n != len(coverage_set):
            output += "}, "
        else:
            output += "}"
    output += "};"
    return output

# def expected_cost(coverage_set):
#     payload = """
# HaarMeasure[{x_,y_,z_}]:=(0.00031910608080810857`^-1)With[{coords={x,y,z}},Product[Product[Sin[coords[[j]]+coords[[k]]]Sin[coords[[j]]-coords[[k]]],{k,j+1,3}],{j,1,3}]];
# MonodromyToCanonical[{a1_?NumericQ,a2_?NumericQ,a3_?NumericQ}]:=With[{first={a1+a2,a1+a3,a2+a3}*Pi/2},Which[first[[3]]>=0,first,True,{Pi/2-first[[1]],first[[2]],-first[[3]]}]];
# IneqsFromIneqCoeffs[coeffs_]:=(And@@(#[[1]]>=0&/@(coeffs.{{1},{x},{y},{z}})))

# If[Or@@(If[#[[2]]==={},IneqsFromIneqCoeffs[#[[1]]],False]&/@#[[3;;All]]),1,0]&/@polytopeData;
# Table[(Print["Working on "<>polytopeData[[j,2]]];NIntegrate[polytopeData[[j,1]]*HaarMeasure[MonodromyToCanonical[{x,y,z}]]*%[[j]]*Product[1-%[[k]],{k,1,j-1}],{x,-1/2,1/2},{y,-1/2,1/2},{z,-1/2,1/2}]),{j,2,Length[%]}]
# Plus@@%
#     """
#     proc = Popen(["/Applications/Mathematica.app/Contents/MacOS/MathKernel"], stdin=PIPE, stdout=PIPE, stderr=PIPE)
#     stdout, stderr = proc.communicate((dump_polytopes(coverage_set) + payload).encode('utf-8'))
#     try:
#         for line in reversed(stdout.decode().split('\n')):
#             if line.startswith("Out"):
#                 return float(line.split('=')[-1])
#     except Exception:
#         warn("Hit an error, dumping the whole output.")
#         return stdout
#     warn("Didn't find a float, dumping the whole output.")
#     return stdout

from collections import defaultdict
from scipy.stats import unitary_group

def expected_cost(coverage_set, bound=10000):
    costs = defaultdict(lambda: 0)

    while sum(costs.values()) < bound:
        u = unitary_group.rvs(4)
        u /= np.linalg.det(u) ** 1/ 4
        target = unitary_to_alcove_coordinate(u)[:3]

        best_cost = float("inf")
        for polytope in coverage_set:
            if polytope.cost < best_cost:
                for convex_subpolytope in \
                        polytope.intersect(nearly(*target)).convex_subpolytopes:
                    solution = scipy_get_random_vertex(convex_subpolytope)

                    if solution.success:
                        working_polytope = polytope
                        best_cost = polytope.cost
                        break
        costs[best_cost] += 1

    return sum([k * v for k, v in costs.items()]) / sum(costs.values())

# cost_table = {}

# for numerator in range(1, 4+1):
# #     c = Fraction(1, 3)
#     d = Fraction(numerator, 4)
#     operations = [
#     #     CircuitPolytope(
#     #         convex_subpolytopes=exactly(Fraction(1/4) * c, Fraction(1/4) * c, Fraction(-1/4) * c).convex_subpolytopes,
#     #         cost=c + Fraction(1, 100),
#     #         operations=[f"{str(c)} XX"],
#     #     ),
#         CircuitPolytope(
#             convex_subpolytopes=exactly(Fraction(1/4) * d, Fraction(1/4) * d, Fraction(-1/4) * d).convex_subpolytopes,
#             cost=d + Fraction(1, 100),
#             operations=[f"{str(d)} XX"],
#         ),
# #         CircuitPolytope(
# #             convex_subpolytopes=exactly(Fraction(1/4), Fraction(1/4), Fraction(-1/4)).convex_subpolytopes,
# #             cost=Fraction(1, 1) + Fraction(1, 100),
# #             operations=["CX"],
# #         ),
#     ]

#     timer = perf_counter()
#     coverage_set = build_coverage_set(operations, chatty=True)
#     print(f"Coverage set has {len(coverage_set)} components, took {perf_counter() - timer}s to build.")
#     timer = perf_counter()
#     cost_table[d] = expected_cost(coverage_set)
#     print(f"Expected cost: {cost_table[d]}; took {perf_counter() - timer}s to integrate.")


# (0.00031910608080810857`^-1) With[{coords = {x, y, z}},
#   Product[Product[
#     Sin[coords[[j]] + coords[[k]]] Sin[coords[[j]] - coords[[k]]], {k,
#       j + 1, 3}], {j, 1, 3}]]

from math import sin
import scipy


def monodromy_haar(z, y, x):
    coords = alcove_to_canonical_coordinate(x, y, z)
    factor = 1
    #     factor = 0.00031910608080810857  # ~~magic number~~
    for j in [0, 1, 2]:
        for k in range(j + 1, 3):
            factor *= sin(coords[j] + coords[k])
            factor *= sin(coords[j] - coords[k])
    return factor


def cost_f(z, y, x):
    for gate_polytope in coverage_set:
        for convex_subpolytope in gate_polytope.convex_subpolytopes:
            if 0 < len(convex_subpolytope.equalities):
                continue
            if all([k + a * x + b * y + c * z >= -1e-10
                    for (k, a, b, c) in convex_subpolytope.inequalities]):
                return gate_polytope.cost
    return 0


def f(z, y, x):
    return cost_f(z, y, x) * monodromy_haar(z, y, x)

# scipy.integrate.tplquad(f, -1, 1, -1, 1, -1, 1, epsabs=1e-6, epsrel=1e-6)




from monodromy.scipy import *

## driver for the optimization loop
def get_operations(*strengths):
    return [
        CircuitPolytope(
            convex_subpolytopes=exactly(Fraction(1/4), Fraction(1/4), Fraction(-1/4)).convex_subpolytopes,
            cost=Fraction(1, 1) + oneq_cost_offset,
            operations=["CX"],
        ),
        *[CircuitPolytope(
            convex_subpolytopes=exactly(
                Fraction(1/4) * strength, Fraction(1/4) * strength, Fraction(-1/4) * strength
            ).convex_subpolytopes,
            cost=strength + oneq_cost_offset,
            operations=[f"{str(strength)} XX"],
        ) for strength in strengths]
    ]

# def extract_cost(operations, chatty=True):
#     timer = perf_counter()
#     coverage_set = build_coverage_set(operations, chatty=chatty)
#     if chatty:
#         print(f"Coverage set has {len(coverage_set)} components, took {perf_counter() - timer}s to build.")
#     timer = perf_counter()
#     retval = expected_cost(coverage_set, bound=10_000)
#     if chatty:
#         print(f"Expected cost: {float(retval)}; took {perf_counter() - timer}s to integrate.")
#     return retval






# initialize the bisection state
# keep halving the interaction amount 'til we see an increase
window_width = 1e-2
oneq_cost_offset = Fraction(1, 100)

cost_table = {1: 3.03}

strength = Fraction(1, 1)
while True:
    strength = strength / Fraction(2)
    operations = get_operations(strength)
    cost_table[strength] = extract_cost(operations)
    if cost_table[strength] > cost_table[strength*2]:
        break

left = strength
middle = strength*2
right = strength*4

# perform bisection
while abs(left - right) > window_width:
    leftmiddle = (left + middle) * Fraction(1, 2)
    middleright = (middle + right) * Fraction(1, 2)
    cost_table[leftmiddle] = extract_cost(get_operations(leftmiddle))
    cost_table[middleright] = extract_cost(get_operations(middleright))

    if cost_table[left] > cost_table[leftmiddle] and cost_table[leftmiddle] < \
            cost_table[middle]:
        left, middle, right = left, leftmiddle, middle
    elif cost_table[leftmiddle] > cost_table[middle] and cost_table[middle] < \
            cost_table[middleright]:
        left, middle, right = leftmiddle, middle, middleright
    elif cost_table[middle] > cost_table[middleright] and cost_table[
        middleright] < cost_table[right]:
        left, middle, right = middle, middleright, right
    else:
        raise RuntimeError("Oops.")





def show_cost_table(cost_table):
    import matplotlib.pyplot as plt
    xs = []
    ys = []
    for k, v in sorted(cost_table.items(), key = lambda x: x[0]):
        xs.append(float(k))
        ys.append(float(v))

    plt.figure()
    plt.plot(xs, ys, 'bo-')
    plt.suptitle(f"Expected cost vs XX interaction strength, 1Qs cost {float(oneq_cost_offset)}, incl. full XX")
    plt.xlabel("XX fraction")
    plt.ylabel("Expected cost")
    plt.show()

show_cost_table(cost_table)
