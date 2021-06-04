"""
monodromy/polytopes.py

Basic data structures for manipulating (non/convex) polytopes.
"""

from copy import copy
from dataclasses import dataclass, field
from fractions import Fraction
from typing import List, Optional

import monodromy.backend
from monodromy.utilities import bitcount, bitscatter, bit_iteration, \
    clear_memoization, epsilon, memoized_property


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
class ConvexPolytopeData:
    """
    The raw data underlying a ConvexPolytope.  Describes a single convex
    polytope, specified by families of `inequalities` and `equalities`, each
    entry of which respectively corresponds to

        inequalities[j][0] + sum_i inequalities[j][i] * xi >= 0

    and

        equalities[j][0] + sum_i equalities[j][i] * xi == 0.
    """

    inequalities: List[List[int]]
    equalities: List[List[int]] = field(default_factory=list)

    @classmethod
    def inflate(cls, data):
        """
        Converts the `data` produced by `dataclasses.asdict` to a live object.
        """

        return cls(**data)


@dataclass
class PolytopeData:
    """
    The raw data of a union of convex polytopes.
    """

    convex_subpolytopes: List[ConvexPolytopeData]

    @classmethod
    def inflate(cls, data):
        """
        Converts the `data` produced by `dataclasses.asdict` to a live object.
        """

        data = {
            **data,
            # overrides
            "convex_subpolytopes": [
                ConvexPolytopeData.inflate(x) if isinstance(x, dict) else x
                for x in data["convex_subpolytopes"]
            ]
        }

        return cls(**data)


@dataclass
class ConvexPolytope(ConvexPolytopeData):
    """
    Houses a single convex polytope, together with methods for manipulation.
    """

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
    def triangulation(self) -> List[List[int]]:
        """
        Non-overlapping simplices which constitute this polytope, specified as
        tuples of indices into .vertices .
        """
        if 0 == len(self.vertices):
            return []
        return monodromy.backend.backend.triangulation(self)

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


def bitmask_iterator(mask, determined_bitmask, total_bitcount, negative_bitmasks):
    """
    Yields bitstrings of length `total_bitcount` which match `mask`
    on the raised bits in `determined_bitmask` and which _do not_ wholly
    match any of the masks in `negative_bitmasks`.
    """
    undetermined_bitmask = determined_bitmask ^ ((1 << total_bitcount) - 1)
    if 0 == len(negative_bitmasks):
        remaining_bitcount = total_bitcount - bitcount(determined_bitmask)
        for j in range(1 << remaining_bitcount):
            # paint possible remainders into the undetermined mask
            yield mask | bitscatter(j, undetermined_bitmask)
    else:
        negative_bitmask, rest = negative_bitmasks[0], negative_bitmasks[1:]
        # ensure that it's possible to find any non-matches
        if ((negative_bitmask & determined_bitmask == negative_bitmask) and
            (negative_bitmask & mask == negative_bitmask)):
            return

        # if we're wholly determined, just recurse
        if negative_bitmask & determined_bitmask == negative_bitmask:
            yield from bitmask_iterator(
                mask,
                determined_bitmask | negative_bitmask,
                total_bitcount,
                rest
            )
            return

        # otherwise, fill in the undetermined bits in negative_bitmask other
        # than the value 11...1 .
        undetermined_bitmask &= negative_bitmask
        for j in range((1 << bitcount(undetermined_bitmask))):
            augmented_mask = mask | bitscatter(j, undetermined_bitmask)
            if augmented_mask & negative_bitmask == negative_bitmask:
                continue
            yield from bitmask_iterator(
                augmented_mask,
                determined_bitmask | negative_bitmask,
                total_bitcount,
                rest
            )


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
    # This method is quite complex. The basic idea is to use inclusion-exclusion
    # to calculate the volume of `polytope` according to `volume_fn`, but our
    # Polytopes tend to come in highly degenerate families, which we exploit to
    # lower the naively exponential complexity of this procedure.  (Compare
    # `naive_alternating_sum` at the bottom of this file.) The two basic
    # mechanisms are:
    #
    # + If vol(A) = 0, then vol(AB) = 0 for any B, hence the entire lattice
    #   under A can be discarded.
    # + If vol(A) = vol(AB), then vol(AC) = vol(ABC) for any C.  These occur in
    #   canceling pairs, hence the entire lattice under A can be discarded.
    #
    # The complexity comes from applying multiple relations of the second sort:
    # one half of a canceling pair in a later application might have been
    # consumed as part of a canceling pair in an earlier application.  We're
    # encouraged to delay dealing with this: the sooner we can finish a large
    # exponential walk, the better off we are, so we collect these relations
    # until we have finished a full intersection depth with no new volumes.
    # See below for a comment describing the correction to the double-counting.

    total_volume = 0
    volume_fn_calls = 0

    vanishing_masks = []  # mask
    alternating_masks = []  # (mask, toggle)

    previous_volumes = {}  # mask -> volume

    # compute the "single-count" sum
    for d in range(len(polytope.convex_subpolytopes)):
        volumes = {}
        did_work = False
        for bitstring in bit_iteration(length=len(polytope.convex_subpolytopes),
                                       weight=1 + d):
            # if this is guaranteed to be zero, skip it
            if any([mask & bitstring == mask for mask in vanishing_masks]):
                continue

            # if this belongs to an alternating skip, skip it
            if any([mask & bitstring == mask
                    for mask, toggle in alternating_masks]):
                continue

            # if this is inheritable from the previous stage, inherit it
            for mask, toggle in alternating_masks:
                previous_volume = previous_volumes.get(bitstring ^ toggle, None)
                if ((mask & bitstring == mask) and (bitstring & toggle != 0)
                        and previous_volume is not None):
                    volumes[bitstring] = previous_volume
                    break

            # if that failed, calculate from scratch
            if volumes.get(bitstring, None) is None:
                intersection = ConvexPolytope(inequalities=[])
                for index, convex_subpolytope in enumerate(polytope.convex_subpolytopes):
                    if 0 != (bitstring & (1 << index)):
                        intersection = intersection.intersect(convex_subpolytope)
                volumes[bitstring] = volume_fn(intersection)
                volume_fn_calls += 1

            # if this has vanishing volume, add it to the skip set; all done.
            if volumes[bitstring] == 0:
                vanishing_masks.append(bitstring)
                volumes[bitstring] = None
                continue

            # try to pair this volume with parents
            for parent_index in range(len(polytope.convex_subpolytopes)):
                parent_toggle = 1 << parent_index
                parent_bitstring = bitstring ^ parent_toggle
                parent_volume = previous_volumes.get(parent_bitstring, None)
                # ensure that we have this parent
                if 0 == bitstring & parent_toggle:
                    continue
                # ensure that our volumes agree
                if volumes[bitstring] != parent_volume:
                    continue
                # if we're noticing a coincidence now, it's the first time it's
                # happened, since otherwise we would have been caught by the
                # skip clause at the start of the middle loop.
                alternating_masks.append((parent_bitstring, parent_toggle))
                if 1 == d % 2:
                    total_volume = total_volume - volumes[bitstring]
                else:
                    total_volume = total_volume + volumes[bitstring]
                break

        for bitstring in bit_iteration(length=len(polytope.convex_subpolytopes),
                                       weight=1+d):
            volume = volumes.get(bitstring, None)
            if volume is None:
                continue
            if any([mask & bitstring == mask
                    for mask, toggle in alternating_masks]):
                continue

            did_work = True
            if 1 == d % 2:
                total_volume = total_volume - volume
            else:
                total_volume = total_volume + volume

        if not did_work:
            break

        # rotate the records
        previous_volumes = volumes

    # Now we account for the multiply-canceled terms, using the existing sorting
    # of `alternating_masks`.  Double-counting occurs only arises when a pair
    # (mask, toggle) matches an untoggled bitstring and a preceding mask matches
    # its toggle-on form.  We can search for such bitstrings according to the
    # _earliest_ preceding mask that matches its toggled form, ensuring that we
    # only correct each previously-erroneously-included string once.

    for j, (mask, toggle) in enumerate(alternating_masks):
        for k, (kth_mask, _) in enumerate(alternating_masks[:j]):
            if 0 == toggle & kth_mask:
                continue
            for bitstring in bitmask_iterator(
                # jth mask matches bitstring, kth mask matches only after toggle
                mask | (kth_mask ^ toggle), mask | kth_mask,
                # expect masks of this size
                len(polytope.convex_subpolytopes),
                # masks in [0, k) don't match and don't include the toggle
                [earlier_mask & -(toggle + 1) for
                 earlier_mask, _ in alternating_masks[:k]] +
                # masks in (k, j) don't match, regardless of toggle
                [earlier_mask for earlier_mask, _ in alternating_masks[1+k:j]] +
                vanishing_masks
            ):
                intersection = ConvexPolytope(inequalities=[])
                for index, convex_subpolytope in enumerate(
                        polytope.convex_subpolytopes):
                    if 0 != (bitstring & (1 << index)):
                        intersection = intersection.intersect(
                            convex_subpolytope)

                volume_fn_calls += 1
                if 1 == bitcount(bitstring) % 2:
                    total_volume += volume_fn(intersection)
                else:
                    total_volume -= volume_fn(intersection)

    return total_volume


@dataclass
class Polytope(PolytopeData):
    """
    A manipulable union of convex polytopes.
    """

    convex_subpolytopes: List[ConvexPolytope]

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
        inequalities: List[List[int]],
        equalities: Optional[List[List[int]]] = None,
) -> Polytope:
    """Convenience method for forming a Polytope with one component."""
    equalities = equalities if equalities is not None else []

    return Polytope(convex_subpolytopes=[
        ConvexPolytope(inequalities=inequalities,
                       equalities=equalities)
    ])


# PEDAGOGICAL VALUE ONLY
def naive_alternating_sum(polytope):
    """
    Inefficiently computes the Euclidean volume of a `Polytope` using the
    inclusion-exclusion alternating.
    """
    total_volume = 0

    for d in range(len(polytope.convex_subpolytopes)):
        volumes = {}
        for bitstring in bit_iteration(length=len(polytope.convex_subpolytopes),
                                       weight=1 + d):
            if volumes.get(bitstring, None) is None:
                intersection = ConvexPolytope(inequalities=[])
                for index, convex_subpolytope in enumerate(polytope.convex_subpolytopes):
                    if 0 != (bitstring & (1 << index)):
                        intersection = intersection.intersect(convex_subpolytope)
                volumes[bitstring] = intersection.volume

            if 1 == d % 2:
                total_volume = total_volume - volumes[bitstring]
            else:
                total_volume = total_volume + volumes[bitstring]

    return total_volume
