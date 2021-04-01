"""
monodromy/polytopes.py

Basic data structures for manipulating (non/convex) polytopes.
"""

from copy import copy
from dataclasses import dataclass
from fractions import Fraction
from typing import List, Optional

import monodromy.backend


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
class ConvexPolytope:
    """
    Houses a single convex polytopes, specified by a family of `inequalities`,
    each entry of which corresponds to the inequality

        inequalities[j][0] + sum_i inequalities[j][i] * xi >= 0
    """

    inequalities: List[List[Fraction]]

    @property
    def volume(self) -> PolytopeVolume:
        """
        (Top-dimensional) Euclidean volume of this convex body.
        """
        try:
            return monodromy.backend.backend.volume(self)
        except monodromy.backend.backend_abc.NoFeasibleSolutions:
            return PolytopeVolume(dimension=0, volume=Fraction(0))

    @property
    def vertices(self) -> List[List[Fraction]]:
        """
        Set of extremal vertices of this convex body.
        """
        return monodromy.backend.backend.vertices(self)

    def reduce(self):  # -> ConvexPolytope
        """
        Produces an equivalent convex body with irredundant inequalities.
        """
        return monodromy.backend.backend.reduce(self)

    def __str__(self) -> str:
        output = ""
        for inequality in self.inequalities:
            output += f"{str(inequality[0]): >5}"
            for index, item in enumerate(inequality[1:]):
                output += f" + {str(item): >5} x{1+index}"
            output += " >= 0\n"

        return output

    def intersect(self, other):  # ConvexPolytope, ConvexPolytope -> ConvexPolytope
        """
        Returns A cap B.
        """
        return ConvexPolytope(
            inequalities=self.inequalities + other.inequalities
        )

    def contains(self, other) -> bool:
        """
        Returns True when this convex body is contained in the right-hand one.
        """
        # NOTE: alternatively, you could check volumes, as below
        cap_vertices = other.intersect(self).vertices
        return all([v in cap_vertices for v in other.vertices])


@dataclass
class Polytope:
    """
    A union of convex polytopes.
    """

    convex_subpolytopes: List[ConvexPolytope]

    @property
    def volume(self) -> PolytopeVolume:
        """
        Computes the Euclidean volume of this polytope.
        """
        # NOTE: The volume of the union is not the straight-up sum of the
        #       volumes of the parts, but rather the alternating sum of the
        #       intersections of the parts. This is potentially expensive, since
        #       it requires traversing the power set of the convex constituents.
        #       We trim this tree by stopping the traversal whenever we see an
        #       intersection with vanishing volume, since that means further
        #       intersections also vanish: 0 <= vol(A) <= vol(B) for A <= B.
        vanishing_indices = []
        volume = PolytopeVolume(dimension=0, volume=Fraction(0, 1))

        top_dimension = 0
        for convex_subpolytope in self.convex_subpolytopes:
            top_dimension = max(top_dimension,
                                convex_subpolytope.volume.dimension)

        for j in range(1, 2 ** len(self.convex_subpolytopes)):
            # if this subset is known to vanish, skip it
            if any([x == (j & x) for x in vanishing_indices]):
                continue
            # otherwise, calculate the bucket of inequalities
            intersection = ConvexPolytope(inequalities=[])
            sign = -1
            for index, convex_subpolytope in enumerate(self.convex_subpolytopes):
                if 0 != j & (1 << index):
                    intersection = intersection.intersect(convex_subpolytope)
                    sign *= -1

            # calculate its volume...
            subvolume = intersection.volume

            # ... update the vanishing indices ...
            if subvolume.dimension < top_dimension:
                vanishing_indices.append(j)

            # ... and account for it
            if sign > 0:
                volume = volume + subvolume
            else:
                volume = volume - subvolume

        return volume

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
            except monodromy.backend.NoFeasibleSolutions:
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
        return clone

    def __str__(self):
        output = "[\n"
        for index, item in enumerate(self.convex_subpolytopes):
            output += str(item)
            if 1 + index < len(self.convex_subpolytopes):
                output += "\n,"
            output += "\n"
        output += "]"

    def contains(self, other) -> bool:
        """
        Returns True when the other polytope is contained in this one.
        """

        little_volume = other.volume
        intersection = other.intersect(self)
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
    trimmable_polytopes = sorted(trimmable_polytopes,
                                 key=lambda x: x.volume,
                                 reverse=True)
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


def make_convex_polytope(inequalities: List[List[Fraction]]) -> Polytope:
    """Convenience method for forming a Polytope with one component."""
    return Polytope(convex_subpolytopes=[
        ConvexPolytope(inequalities=inequalities)
    ])
