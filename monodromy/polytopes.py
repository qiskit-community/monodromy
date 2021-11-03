"""
monodromy/polytopes.py

Basic data structures for manipulating (non/convex) polytopes.
"""

from copy import copy
from dataclasses import dataclass
from fractions import Fraction
from typing import List, Optional

import monodromy.backend
from monodromy.exceptions import NoFeasibleSolutions
from monodromy.io.base import ConvexPolytopeData, PolytopeData, \
    generate_anonymous_cp_name
from monodromy.volume import alternating_sum
from monodromy.utilities import clear_memoization, epsilon, memoized_property


@dataclass(order=True)
class PolytopeVolume:
    """
    Represents the volume of a (possibly not top-dimensional) polytope.
    """
    dimension: int
    volume: Fraction

    def __add__(self, other):
        if self.dimension > other.dimension:
            return self
        elif self.dimension < other.dimension:
            return other
        else:
            return PolytopeVolume(
                volume=self.volume + other.volume,
                dimension=self.dimension,
            )

    def __sub__(self, other):
        if self.dimension > other.dimension:
            return self
        elif self.dimension == other.dimension:
            return PolytopeVolume(
                dimension=self.dimension,
                volume=self.volume - other.volume,
            )
        else:
            raise ValueError(f"Illegal to subtract high dim'l volume "
                             f"from low dim'l source.")


@dataclass
class ConvexPolytope(ConvexPolytopeData):
    """
    Houses a single convex polytope, together with methods for manipulation.

    NOTE: This object is meant to be read-only after instantiation.
    """

    @memoized_property
    def volume(self) -> PolytopeVolume:
        """
        (Top-dimensional) Euclidean volume of this convex body.
        """
        try:
            return monodromy.backend.backend.volume(self)
        except NoFeasibleSolutions:
            return PolytopeVolume(dimension=0, volume=Fraction(0))

    @memoized_property
    def vertices(self) -> List[List[Fraction]]:
        """
        Set of extremal vertices of this convex body.
        """
        try:
            return monodromy.backend.backend.vertices(self)
        except NoFeasibleSolutions:
            return []

    @memoized_property
    def triangulation(self) -> List[List[int]]:
        """
        Non-overlapping simplices which constitute this polytope, specified as
        tuples of indices into .vertices .
        """
        if 0 == len(self.vertices):
            return []
        return monodromy.backend.backend.triangulation(self)

    @classmethod
    def convex_hull(cls, vertices):
        """
        Produces the minimal ConvexPolytope containing the list of `vertices`.
        """
        return monodromy.backend.backend.convex_hull(vertices)

    def reduce(self):  # -> ConvexPolytope
        """
        Produces an equivalent convex body with irredundant inequalities.

        Raises NoFeasibleSolutions if the reduced polytope is empty.
        """
        return monodromy.backend.backend.reduce(self)

    def __str__(self) -> str:
        output = f"# {self.name}: \n"
        for inequality in self.inequalities:
            output += f"{str(inequality[0]): >5}"
            for index, item in enumerate(inequality[1:]):
                output += f" + {str(item): >5} x{1+index}"
            output += " >= 0\n"

        for equality in self.equalities:
            output += f"{str(equality[0]): >5}"
            for index, item in enumerate(equality[1:]):
                output += f" + {str(item): >5} x{1+index}"
            output += " == 0\n"

        return output

    def intersect(self, other):  # ConvexPolytope, ConvexPolytope -> ConvexPolytope
        """
        Returns A cap B.
        """
        return ConvexPolytope(
            inequalities=self.inequalities + other.inequalities,
            equalities=self.equalities + other.equalities,
            name=f"{self.name} ∩ {other.name}"
        )

    def has_element(self, point) -> bool:
        """
        Returns True when `point` belongs to `self`.
        """
        return (all([-epsilon <= inequality[0] +
                                 sum(x * y for x, y in
                                     zip(point, inequality[1:]))
                     for inequality in self.inequalities]) and
                all([abs(equality[0] + sum(x * y for x, y in
                                           zip(point, equality[1:])))
                     <= epsilon
                     for equality in self.equalities]))

    def contains(self, other) -> bool:
        """
        Returns True when this convex body is contained in the right-hand one.
        """
        # NOTE: Alternatively, you could check volumes, as below.  Also
        #       alternatively, you could use .reduce() and check that the facet
        #       definitions are the same (up to rescaling?).  I think this is
        #       the most efficient version, since it doesn't enumerate vertices?
        cap_vertices = other.intersect(self).vertices
        return all([v in cap_vertices for v in other.vertices])


@dataclass
class Polytope(PolytopeData):
    """
    A manipulable union of convex polytopes.

    NOTE: This object is meant to be read-only after instantiation.
    """

    @classmethod
    def inflate(cls, data):
        """
        Converts the `data` produced by `dataclasses.asdict` to a live object.
        """

        data = {
            **data,
            "convex_subpolytopes": [
                ConvexPolytope.inflate(x) if isinstance(x, dict) else x
                for x in data["convex_subpolytopes"]
            ],
        }

        return super().inflate(data)

    @memoized_property
    def volume(self) -> PolytopeVolume:
        """
        Computes the Euclidean volume of this polytope.
        """

        volumes = [cp.volume for cp in self.convex_subpolytopes]
        top_dimension = 0 if len(volumes) == 0 \
            else max([volume.dimension for volume in volumes])

        def unwrapped_volume(convex_polytope):
            if convex_polytope.volume.dimension == top_dimension:
                return convex_polytope.volume.volume
            else:
                return 0

        volume = alternating_sum(
            polytope=Polytope(convex_subpolytopes=[
                cp for cp in self.convex_subpolytopes
                    if cp.volume.dimension == top_dimension
            ]),
            volume_fn=unwrapped_volume,
        )

        return PolytopeVolume(dimension=top_dimension, volume=volume)

    @memoized_property
    def vertices(self):
        """
        Returns the vertices of the convex subpolytopes.
        """
        return [convex_subpolytope.vertices
                for convex_subpolytope in self.convex_subpolytopes]

    def reduce(self):
        """
        Removes redundant inequality sets from a Polytope.
        """

        independent_polytopes = []
        for convex_subpolytope in self.convex_subpolytopes:
            try:
                independent_polytopes.append(Polytope(
                    convex_subpolytopes=[convex_subpolytope.reduce()]
                ))
            except NoFeasibleSolutions:
                pass

        independent_polytopes = trim_polytope_set(independent_polytopes)

        clone = copy(self)
        clone.convex_subpolytopes = [
            independent_polytope.convex_subpolytopes[0]
            for independent_polytope in independent_polytopes
        ]

        return clone

    def union(self, other):
        """
        Returns A cup B.
        """
        clone = copy(self)
        clone.convex_subpolytopes = (self.convex_subpolytopes +
                                     other.convex_subpolytopes)
        clear_memoization(clone)
        return clone

    def intersect(self, other):
        """
        Returns A cap B.
        """
        # distribute the intersection over the union
        convex_subpolytopes = []
        for left_subpolytope in self.convex_subpolytopes:
            for right_subpolytope in other.convex_subpolytopes:
                convex_subpolytopes.append(left_subpolytope.intersect(
                    right_subpolytope
                ))

        clone = copy(self)
        clone.convex_subpolytopes = convex_subpolytopes
        clear_memoization(clone)
        return clone

    def __str__(self):
        output = "[\n"
        for index, item in enumerate(self.convex_subpolytopes):
            output += str(item)
            if 1 + index < len(self.convex_subpolytopes):
                output += ","
            output += "\n"
        output += "]"

        return output

    def has_element(self, point) -> bool:
        """
        Returns T when point belongs to this Polytope.
        """
        return any([cp.has_element(point) for cp in self.convex_subpolytopes])

    def contains(self, other) -> bool:
        """
        Returns True when the other polytope is contained in this one.
        """
        # for n self.convex_subpolytopes and m other.convex_subpolytopes,
        # computing these volumes takes worst-case 2^m + 2^(nm) calls to lrs.
        # however, a necessary-but-insufficient condition for containment is
        # a containment of vertex sets, which takes only m + nm calls to lrs.
        # we check that first and short-circuit if it fails.

        for other_subvertices in other.vertices:
            for other_vertex in other_subvertices:
                if not self.has_element(other_vertex):
                    return False

        # now do the expensive version that also handles sufficiency
        intersection = other.intersect(self)
        little_volume = other.volume
        cap_volume = intersection.volume
        return cap_volume == little_volume


def trim_polytope_set(
        trimmable_polytopes: List[Polytope],
        fixed_polytopes: Optional[List[Polytope]] = None
) -> List[Polytope]:
    """
    Reduce a family of `Polytope`s by removing those which are in the union of
    the rest.

    For flexibility, we break the input into two parts: a set of ConvexPolytopes
    which we're trying to trim, as well as a set of ConvexPolytopes which
    contribute to the notion of redundancy but which we don't attempt to reduce.

    Returns an irredundant subsequence from trimmable_polytopes.
    """

    # NOTE: This is an expensive call, because testing for (convex) polytope
    #           containment in a (nonconvex!!) polytope is tricky business.
    #           There is absolutely room for improvement in performance here.

    fixed_polytope = Polytope(convex_subpolytopes=[])
    if fixed_polytopes is not None:
        for polytope in fixed_polytopes:
            fixed_polytope = fixed_polytope.union(polytope)

    # sort them by volume, then traverse in ascending order
    trimmable_polytopes = sorted(
        trimmable_polytopes,
        key=lambda x: x.volume,
        reverse=True,
    )
    for index in range(len(trimmable_polytopes) - 1, -1, -1):
        # pick a polytope, test whether it's contained in the others
        this_polytope = trimmable_polytopes[index]

        other_polytope = fixed_polytope
        for subindex, polytope in enumerate(trimmable_polytopes):
            if subindex == index:
                continue
            other_polytope = other_polytope.union(polytope)

        if other_polytope.contains(this_polytope):
            del trimmable_polytopes[index]

    return trimmable_polytopes


def make_convex_polytope(
        inequalities: List[List[int]],
        equalities: Optional[List[List[int]]] = None,
        name: Optional[str] = None,
) -> Polytope:
    """
    Convenience method for forming a Polytope with one component.
    """
    equalities = equalities if equalities is not None else []
    name = name if name is not None else generate_anonymous_cp_name()

    return Polytope(convex_subpolytopes=[
        ConvexPolytope(inequalities=inequalities,
                       equalities=equalities,
                       name=name)
    ])
