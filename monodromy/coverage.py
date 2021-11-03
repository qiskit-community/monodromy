"""
monodromy/coverage.py

Routines for converting a family of native gates to a minimal set of minimum-
cost circuits whose union covers the entire monodromy polytope.
"""

from dataclasses import dataclass
from fractions import Fraction
import heapq
from typing import Dict, List, Optional

from .coordinates import monodromy_alcove, monodromy_alcove_c2, rho_reflect
from .io.base import CircuitPolytopeData
from .elimination import cylinderize, project
from .polytopes import Polytope, trim_polytope_set
from .static.examples import everything_polytope, identity_polytope
from .static.qlr_table import qlr_polytope


@dataclass
class CircuitPolytope(Polytope, CircuitPolytopeData):
    """
    A polytope describing the alcove coverage of a particular circuit type.
    """

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


def deduce_qlr_consequences(
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
        p = p.intersect(cylinderize(monodromy_alcove, value))

    # also impose whatever constraints we were given besides
    p = p.intersect(cylinderize(a_polytope, coordinates["a"]))
    p = p.intersect(cylinderize(b_polytope, coordinates["b"]))
    p = p.intersect(cylinderize(c_polytope, coordinates["c"]))

    # restrict to the A_{C_2} part of the target coordinate
    p = p.intersect(cylinderize(monodromy_alcove_c2, coordinates[target]))

    # lastly, project away the non-target parts
    p = p.reduce()
    for index in range(9, 0, -1):
        if index in coordinates[target]:
            continue
        p = project(p, index)
        p = p.reduce()

    return p


def prereduce_operation_polytopes(
        operations: List[CircuitPolytope],
        target_coordinate: str = "c",
        background_polytope: Optional[Polytope] = None,
        chatty: bool = False,
) -> Dict[str, CircuitPolytope]:
    """
    Specializes the "b"-coordinates of the monodromy polytope to a particular
    operation, then projects them away.
    """

    coordinates = {
        "a": [0, 1, 2, 3],
        "b": [0, 4, 5, 6],
        "c": [0, 7, 8, 9],
    }
    prereduced_operation_polytopes = {}

    for operation in operations:
        if chatty:
            print(f"Prereducing QLR relations for {'.'.join(operation.operations)}")
        p = background_polytope if background_polytope is not None \
            else everything_polytope
        p = p.intersect(qlr_polytope)
        p = p.union(rho_reflect(p, coordinates[target_coordinate]))
        for value in coordinates.values():
            p = p.intersect(cylinderize(monodromy_alcove, value))
        p = p.intersect(cylinderize(operation, coordinates["b"]))
        p = p.intersect(cylinderize(monodromy_alcove_c2, coordinates[target_coordinate]))

        # project away the operation part
        p = p.reduce()
        for index in [6, 5, 4]:
            p = project(p, index)
            p = p.reduce()
        prereduced_operation_polytopes[operation.operations[-1]] = p

    return prereduced_operation_polytopes


def build_coverage_set(
        operations: List[CircuitPolytope],
        chatty: bool = False,
) -> List[CircuitPolytope]:
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

    # start by generating precalculated operation polytopes
    prereduced_operation_polytopes = prereduce_operation_polytopes(
        operations
    )

    # a collection of polytopes explored so far, and their union
    total_polytope = CircuitPolytope(
        convex_subpolytopes=identity_polytope.convex_subpolytopes,
        operations=[],
        cost=0.,
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
            to_be_reduced = trim_polytope_set(
                to_be_reduced, fixed_polytopes=[total_polytope]
            )
            necessary_polytopes += to_be_reduced
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
        # TODO: This used to be part of a call to `deduce_qlr_consequences`, which
        #       we split up for efficiency.  See GH #13.
        new_polytope = prereduced_operation_polytopes[
            next_polytope.operations[-1]
        ]
        new_polytope = new_polytope.intersect(cylinderize(
            tail_polytope, [0, 1, 2, 3], parent_dimension=7
        ))
        new_polytope = new_polytope.reduce()
        for index in [3, 2, 1]:
            new_polytope = project(new_polytope, index).reduce()
        # specialize it from a Polytope to a CircuitPolytope
        new_polytope = CircuitPolytope(
            operations=next_polytope.operations,
            cost=next_polytope.cost,
            convex_subpolytopes=new_polytope.convex_subpolytopes
        )

        to_be_reduced.append(new_polytope)

        if chatty:
            print(f"Cost {next_polytope.cost} ", end="")
            volume = new_polytope.volume
            if volume.dimension == 3:
                volume = volume.volume / monodromy_alcove_c2.volume.volume
                print(f"and Euclidean volume {float(100 * volume):6.2f}%")
            else:
                print(f"and Euclidean volume {0:6.2f}%")
        
        # if this polytope is NOT of maximum volume,
        if monodromy_alcove_c2.volume > new_polytope.volume:
            # add this polytope + the continuations to the priority queue
            for operation in operations:
                heapq.heappush(to_be_explored, CircuitPolytope(
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
            vol = vol.volume / monodromy_alcove_c2.volume.volume
        else:
            vol = Fraction(0)
        print(f"{float(100 * vol):6.2f}% = "
              f"{str(vol.numerator): >4}/{str(vol.denominator): <4} "
              f"\t | {float(gate.cost):4.2f}"
              f"\t | {'.'.join(gate.operations)}")
