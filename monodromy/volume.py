"""
monodromy/volume.py

Helper routines for efficiently calculating the volume of a `Polytope`,
presented as a union of `ConvexPolytope`s.
"""

from monodromy.utilities import bitcount, bit_iteration, bitscatter


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
                intersection = None
                for index, convex_subpolytope in enumerate(polytope.convex_subpolytopes):
                    if 0 != (bitstring & (1 << index)):
                        intersection = convex_subpolytope if intersection is None \
                            else intersection.intersect(convex_subpolytope)
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
    # of `alternating_masks`.  Double-counting occurs only when a pair
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
                intersection = None
                for index, convex_subpolytope in enumerate(
                        polytope.convex_subpolytopes):
                    if 0 != (bitstring & (1 << index)):
                        intersection = convex_subpolytope if intersection is None \
                                else intersection.intersect(convex_subpolytope)

                volume_fn_calls += 1
                if 1 == bitcount(bitstring) % 2:
                    total_volume += volume_fn(intersection)
                else:
                    total_volume -= volume_fn(intersection)

    return total_volume


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
                intersection = None
                for index, convex_subpolytope in enumerate(polytope.convex_subpolytopes):
                    if 0 != (bitstring & (1 << index)):
                        intersection = convex_subpolytope if intersection is None \
                            else intersection.intersect(convex_subpolytope)
                volumes[bitstring] = intersection.volume

            if 1 == d % 2:
                total_volume = total_volume - volumes[bitstring]
            else:
                total_volume = total_volume + volumes[bitstring]

    return total_volume
