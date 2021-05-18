"""
monodromy/polytopes.py

Basic data structures for manipulating (non/convex) polytopes.
"""

from copy import copy
from dataclasses import dataclass, field
from fractions import Fraction
from typing import List, Optional

import monodromy.backend
from monodromy.utilities import bit_iteration, clear_memoization, \
    memoized_property


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
    equalities: List[List[Fraction]] = field(default_factory=list)

    @memoized_property
    def volume(self) -> PolytopeVolume:
        """
        (Top-dimensional) Euclidean volume of this convex body.
        """
        try:
            return monodromy.backend.backend.volume(self)
        except monodromy.backend.backend_abc.NoFeasibleSolutions:
            return PolytopeVolume(dimension=0, volume=Fraction(0))

    @memoized_property
    def vertices(self) -> List[List[Fraction]]:
        """
        Set of extremal vertices of this convex body.
        """
        try:
            return monodromy.backend.backend.vertices(self)
        except monodromy.backend.backend_abc.NoFeasibleSolutions:
            return []

    @memoized_property
    def triangulation(self) -> List[List]:
        if 0 == len(self.vertices):
            return []
        return monodromy.backend.backend.triangulation(self.vertices)

    def reduce(self):  # -> ConvexPolytope
        """
        Produces an equivalent convex body with irredundant inequalities.

        Raises NoFeasibleSolutions if the reduced polytope is empty.
        """
        return monodromy.backend.backend.reduce(self)

    def __str__(self) -> str:
        output = ""
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
        )

    def has_element(self, point) -> bool:
        return (all([0 <= inequality[0] + sum(x * y for x, y in
                                              zip(point, inequality[1:]))
                     for inequality in self.inequalities]) and
                all([0 == equality[0] + sum(x * y for x, y in
                                            zip(point, equality[1:]))
                     for equality in self.equalities]))

    def contains(self, other) -> bool:
        """
        Returns True when this convex body is contained in the right-hand one.
        """
        # NOTE: alternatively, you could check volumes, as below
        cap_vertices = other.intersect(self).vertices
        return all([v in cap_vertices for v in other.vertices])


def alternating_sum(polytope, volume_fn):
    """
    Efficiently computes the inclusion-exclusion alternating sum for the volume
    of a `Polytope`, as computed by `volume_fn` on its convex intersections.

    `volume_fn` is required to be:

        + real-valued,
        + strictly monotonic: if A ≤ B, then vol(A) ≤ vol(B),
                              with equality only if A = B,
        + weakly additive: vol(A u B) ≤ vol(A) + vol(B).
    """
    total_volume = 0

    vanishing_skip_masks = []
    clever_skip_masks = []

    previous_volumes = {}
    for d in range(len(polytope.convex_subpolytopes)):
        volumes = {}
        for bitstring in bit_iteration(length=len(polytope.convex_subpolytopes),
                                       weight=1 + d):\
            # if this is guaranteed to be zero, skip it
            if any([mask & bitstring == mask for mask in vanishing_skip_masks]):
                continue

            # if this is inheritable from the previous stage, inherit it
            for mask, toggle in clever_skip_masks:
                if (mask & bitstring == mask) and (bitstring & toggle != 0):
                    volumes[bitstring] = previous_volumes[bitstring ^ toggle]
                    break

            # if we still haven't inherited it, calculate it from scratch
            if volumes.get(bitstring, None) is None:
                intersection = ConvexPolytope(inequalities=[])
                for index, convex_subpolytope in enumerate(polytope.convex_subpolytopes):
                    if 0 != (bitstring & (1 << index)):
                        intersection = intersection.intersect(convex_subpolytope)
                volumes[bitstring] = volume_fn(intersection)

            # if this (freshly calculated value) has vanishing volume, add it to the skip set
            if volumes[bitstring] == 0:
                vanishing_skip_masks.append(bitstring)

            # if this overlaps with a parent not already related to this, record the relation
            for parent_index in range(0, len(polytope.convex_subpolytopes)):
                toggle = 2 ** parent_index
                parent_string = bitstring ^ toggle

                # ensure this is a parent of our string
                if 0 == toggle & bitstring:
                    continue

                # ensure that our volume matches our parent's
                if (volumes[bitstring] != previous_volumes.get(parent_string, None)):
                    continue

                # ensure we're not already related by a mask
                if any([(parent_string & mask == mask) and (mask_toggle == toggle)
                        for mask, mask_toggle in clever_skip_masks]):
                    continue

                # record this fresh mask :)
                clever_skip_masks.append((parent_string, toggle))

        # sum the signed volumes at this weight
        # TODO: find an early stopping condition
        for bitstring in bit_iteration(length=len(polytope.convex_subpolytopes),
                                       weight=1 + d):
            if (volumes.get(bitstring, None) is None):
                continue
            if any([mask & bitstring == mask for mask in vanishing_skip_masks]):
                continue
            if 1 == d % 2:
                total_volume = total_volume - volumes[bitstring]
            else:
                total_volume = total_volume + volumes[bitstring]

        # rotate the volume records
        previous_volumes = volumes

    return total_volume


@dataclass
class Polytope:
    """
    A union of convex polytopes.
    """

    convex_subpolytopes: List[ConvexPolytope]

    @memoized_property
    def volume(self) -> PolytopeVolume:
        """
        Computes the Euclidean volume of this polytope.
        """

        top_dimension = 0
        for convex_subpolytope in self.convex_subpolytopes:
            top_dimension = max(top_dimension,
                                convex_subpolytope.volume.dimension)

        def unwrapped_volume(convex_polytope):
            if convex_polytope.volume.dimension == top_dimension:
                return convex_polytope.volume.volume
            else:
                return 0

        volume = alternating_sum(
            polytope=self,
            volume_fn=unwrapped_volume,
        )

        return PolytopeVolume(dimension=top_dimension, volume=volume)

    @memoized_property
    def vertices(self):
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
        inequalities: List[List[Fraction]],
        equalities: Optional[List[List[Fraction]]] = None,
) -> Polytope:
    """Convenience method for forming a Polytope with one component."""
    equalities = equalities if equalities is not None else []

    return Polytope(convex_subpolytopes=[
        ConvexPolytope(inequalities=inequalities,
                       equalities=equalities)
    ])
