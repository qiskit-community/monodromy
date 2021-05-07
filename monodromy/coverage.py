"""
monodromy/coverage.py
"""

from copy import copy
from dataclasses import dataclass
from fractions import Fraction
import heapq
from typing import List, Optional

from .elimination import cylinderize, project
from .examples import everything_polytope, identity_polytope
from .polytopes import ConvexPolytope, Polytope, trim_polytope_set
from .qlr_table import alcove, alcove_c2, qlr_polytope


@dataclass
class GatePolytope(Polytope):
    """
    A polytope describing the alcove coverage of a particular circuit type.
    """
    cost: Fraction
    operations: List[str]

    def __gt__(self, other):
        return (self.cost > other.cost) or \
               (self.cost == other.cost and self.volume > other.volume)

    def __ge__(self, other):
        return (self.cost > other.cost) or \
               (self.cost == other.cost and self.volume >= other.volume)

    def __lt__(self, other):
        return (self.cost < other.cost) or \
               (self.cost == other.cost and self.volume < other.volume)

    def __le__(self, other):
        return (self.cost < other.cost) or \
               (self.cost == other.cost and self.volume <= other.volume)


def rho_reflect(polytope, coordinates=None):
    """
    Applies rho-reflection to the indicated `coordinates` of `polytope`.
    If `coordinates` is not supplied, uses the final three coordinates.
    """

    if coordinates is None:
        coordinates = [0, -3, -2, -1]

    # an inequality
    #     d + x a1 + y a2 + z a3 >= 0
    # induces on rho-application
    #     d + x (a3 + 1/2) + y (a4 + 1/2) + z (a1 - 1/2) >= 0, or
    #    (d + 1/2 x + 1/2 y - 1/2 z) + (z - y) a1 + (-y) a2 + (x - y) a3 >= 0.
    rho_subpolytopes = []
    for convex_subpolytope in polytope.convex_subpolytopes:
        rotated_equalities, rotated_inequalities = [], []
        for inequality in convex_subpolytope.inequalities:
            d = inequality[coordinates[0]]
            x = inequality[coordinates[1]]
            y = inequality[coordinates[2]]
            z = inequality[coordinates[3]]

            new_inequality = copy(inequality)
            new_inequality[coordinates[0]] = d + x / 2 + y / 2 - z / 2
            new_inequality[coordinates[1]] = z - y
            new_inequality[coordinates[2]] = 0 - y
            new_inequality[coordinates[3]] = x - y

            rotated_inequalities.append(new_inequality)

        for equality in convex_subpolytope.equalities:
            d = equality[coordinates[0]]
            x = equality[coordinates[1]]
            y = equality[coordinates[2]]
            z = equality[coordinates[3]]

            new_equality = copy(equality)
            new_equality[coordinates[0]] = d + x / 2 + y / 2 - z / 2
            new_equality[coordinates[1]] = z - y
            new_equality[coordinates[2]] = 0 - y
            new_equality[coordinates[3]] = x - y

            rotated_equalities.append(new_equality)

        rho_subpolytopes.append(ConvexPolytope(
            inequalities=rotated_inequalities,
            equalities=rotated_equalities,
        ))

    return Polytope(convex_subpolytopes=rho_subpolytopes)


def intersect_and_project(
        target: str,
        a_polytope: Polytope,
        b_polytope: Polytope,
        c_polytope: Polytope,
        extra_polytope: Optional[Polytope] = None,
) -> Polytope:
    """
    Produces the consequences for `target` for a family of a-, b-, and
    c-inequalities.  `target` can take on the values 'a', 'b', or 'c'.
    """

    coordinates = {
        "a": [0, 1, 2, 3],
        "b": [0, 4, 5, 6],
        "c": [0, 7, 8, 9],
    }
    assert target in coordinates.keys()

    if extra_polytope is None:
        extra_polytope = everything_polytope

    p = extra_polytope.intersect(qlr_polytope)
    p = p.union(rho_reflect(p, coordinates[target]))
    # impose the "large" alcove constraints
    for value in coordinates.values():
        p = p.intersect(cylinderize(alcove, value))

    # also impose whatever constraints we were given besides
    p = p.intersect(cylinderize(a_polytope, coordinates["a"]))
    p = p.intersect(cylinderize(b_polytope, coordinates["b"]))
    p = p.intersect(cylinderize(c_polytope, coordinates["c"]))

    # restrict to the A_{C_2} part of the target coordinate
    p = p.intersect(cylinderize(alcove_c2, coordinates[target]))

    # lastly, project away the non-target parts
    p = p.reduce()
    for index in range(9, 0, -1):
        if index in coordinates[target]:
            continue
        p = project(p, index)
        p = p.reduce()

    return p


def build_coverage_set(
        operations: List[GatePolytope],
        chatty: bool = False,
) -> List[GatePolytope]:
    """
    Given a set of `operations`, thought of as members of a native gate set,
    this emits a list of circuit shapes built as sequences of those operations
    which is:

    + Exhaustive: Every two-qubit unitary is covered by one of the circuit
                  designs in the list.
    + Irredundant: No circuit design is completely contained within other
                   designs in the list which are of equal or lower cost.

    If `chatty` is toggled, emits progress messages.
    """

    # a collection of polytopes explored so far, and their union
    total_polytope = GatePolytope(
        convex_subpolytopes=identity_polytope.convex_subpolytopes,
        operations=[],
        cost=Fraction(0),
    )
    necessary_polytopes = [total_polytope]

    # a priority queue of sequences to be explored
    to_be_explored = []
    for operation in operations:
        heapq.heappush(to_be_explored, operation)

    # a set of polytopes waiting to be reduced, all of equal cost
    to_be_reduced = []
    waiting_cost = 0

    # main loop: dequeue the next cheapest gate combination to explore
    while 0 < len(to_be_explored):
        next_polytope = heapq.heappop(to_be_explored)
        
        # if this cost is bigger than the old cost, flush
        if next_polytope.cost > waiting_cost:
            necessary_polytopes += trim_polytope_set(
                to_be_reduced, fixed_polytopes=[total_polytope]
            )
            for new_polytope in to_be_reduced:
                total_polytope = total_polytope.union(new_polytope).reduce()
            to_be_reduced = []
            waiting_cost = next_polytope.cost

        if chatty:
            print(f"Considering {'·'.join(next_polytope.operations)};\t",
                  end="")

        # find the ancestral polytope
        tail_polytope = next((p for p in necessary_polytopes
                              if p.operations == next_polytope.operations[:-1]),
                             None)
        # if there's no ancestor, skip
        if tail_polytope is None:
            if chatty:
                print("no ancestor, skipping.")
            continue
        
        # take the head's polytopes, adjoin the new gate (& its rotation),
        # calculate new polytopes, and add those polytopes to the working set
        new_polytope = intersect_and_project(
            target="c",
            a_polytope=tail_polytope,
            b_polytope=next_polytope,
            c_polytope=everything_polytope,
        )
        # specialize it from a Polytope to a GatePolytope
        new_polytope = GatePolytope(
            operations=next_polytope.operations,
            cost=next_polytope.cost,
            convex_subpolytopes=new_polytope.convex_subpolytopes
        )

        new_polytope = new_polytope.reduce()
        to_be_reduced.append(new_polytope)

        if chatty:
            print(f"Cost {next_polytope.cost} ", end="")
            volume = new_polytope.volume
            if volume.dimension == 3:
                volume = volume.volume / alcove_c2.volume.volume
                print(f"and volume {float(100 * volume):6.2f}%")
            else:
                print(f"and volume {0:6.2f}%")
        
        # if this polytope is NOT of maximum volume,
        if alcove_c2.volume > new_polytope.volume:
            # add this polytope + the continuations to the priority queue
            for operation in operations:
                heapq.heappush(to_be_explored, GatePolytope(
                    operations=next_polytope.operations + operation.operations,
                    cost=next_polytope.cost + operation.cost,
                    convex_subpolytopes=operation.convex_subpolytopes,
                ))
        else:
            # the cheapest option that gets us to 100% is good enough.
            break
                
    # one last flush
    necessary_polytopes += trim_polytope_set(
        to_be_reduced, fixed_polytopes=[total_polytope]
    )
    for new_polytope in to_be_reduced:
        total_polytope = total_polytope.union(new_polytope).reduce()

    return necessary_polytopes


def print_coverage_set(necessary_polytopes):
    print("Percent volume of A_C2\t | Cost\t | Sequence name")
    for gate in necessary_polytopes:
        vol = gate.volume
        if vol.dimension == 3:
            vol = vol.volume / alcove_c2.volume.volume
        else:
            vol = Fraction(0)
        print(f"{float(100 * vol):6.2f}% = {str(vol.numerator): >4}/{str(vol.denominator): <4} "
              f"\t | {float(gate.cost):4.2f}"
              f"\t | {'.'.join(gate.operations)}")
