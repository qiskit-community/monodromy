"""
monodromy/xx_decompose/xx_polytope.py

Defines a data structure which describes those two-qubit programs accessible to
a given sequence of XX-type interactions.
"""

from dataclasses import dataclass
import itertools

import numpy as np

from ..utilities import epsilon, memoized_property


A = np.array([
    [ 1, -1,  0],  # a ≥ b
    [ 0,  1, -1],  # b ≥ c
    [ 0,  0,  1],  # c ≥ 0
    [-1, -1,  0],  # pi/2 ≥ a + b
    [-1, -1, -1],  # strength
    [ 1, -1, -1],  # slant
    [ 0,  0, -1],  # frustrum
])
A1 = A.reshape(7, 1, 3)
A1inv = np.linalg.pinv(A1)
A2 = np.array([np.array([x, y], dtype=float)
               for (x, y) in itertools.combinations(A, 2)])
A2inv = np.linalg.pinv(A2)
A3 = np.array([np.array([x, y, z], dtype=float)
              for (x, y, z) in itertools.combinations(A, 3)])
A3inv = np.linalg.pinv(A3)
"""
These globals house matrices, computed once-and-for-all, which project onto the
Euclidean-nearest planes spanned by the planes/points/lines defined by the faces
of any XX polytope.

See `XXPolytope.nearest`.
"""


@dataclass
class XXPolytope:
    """
    Describes those two-qubit programs accessible to a given sequence of XX-type
    interactions.

    Strengths are normalized so that CX corresponds to pi / 4.
    """

    total_strength: float = 0.0
    max_strength: float = 0.0
    place_strength: float = 0.0

    @classmethod
    def from_strengths(cls, *strengths):
        """
        Constructs an XXPolytope from a sequence of strengths.
        """
        total_strength, max_strength, place_strength = 0, 0, 0
        for strength in strengths:
            total_strength += strength
            if strength >= max_strength:
                max_strength, place_strength = strength, max_strength
            elif strength >= place_strength:
                place_strength = strength

        return XXPolytope(
            total_strength=total_strength,
            max_strength=max_strength,
            place_strength=place_strength
        )

    def add_strength(self, new_strength: float = 0.0):
        """
        Returns a new XXPolytope with one new XX interaction appended.
        """
        return XXPolytope(
            total_strength=self.total_strength + new_strength,
            max_strength=max(self.max_strength, new_strength),
            place_strength=(
                new_strength if new_strength > self.place_strength else
                self.max_strength if new_strength > self.max_strength else
                self.place_strength
            )
        )

    @memoized_property
    def _offsets(self):
        """
        Returns b with A*x + b ≥ 0 iff x belongs to the XXPolytope.
        """
        return np.array([
            0, 0, 0, np.pi / 2,
            self.total_strength,
            self.total_strength - 2 * self.max_strength,
            self.total_strength - self.max_strength - self.place_strength
        ])

    def member(self, point):
        """Returns True when `point` is a member of `self`."""
        global A

        reflected_point = point.copy().reshape(-1, 3)
        rows = reflected_point[:, 0] >= np.pi / 4 + epsilon
        reflected_point[rows, 0] = np.pi / 2 - reflected_point[rows, 0]
        reflected_point = reflected_point.reshape(point.shape)

        return np.all(self._offsets +
                      np.einsum('ij,...j->...i', A, reflected_point)
                      >= -epsilon,
                      axis=-1)

    def nearest(self, point):
        """
        Finds the nearest point (in Euclidean or infidelity distance) to `self`.
        """

        # NOTE: A CAS says that there are no degenerate double intersections,
        #       and the only degenerate triple intersections are
        #       (1, -1, 0), (0, 0, 1), (-1, 1, 1) and
        #       (1, 1, 0), (0, 0, 1), (1, 1, 1).
        #       Skipping this pair won't save much work, so we don't bother.

        global A1, A1inv, A2, A2inv, A3, A3inv
        # These global variables contain projection matrices, computed once-and-
        # for-all, which produce the Euclidean-nearest projection.

        if isinstance(point, np.ndarray) and len(point.shape) == 1:
            y0 = point.copy()
        elif isinstance(point, list):
            y0 = np.array(point)
        else:
            raise TypeError(
                f"Can't handle type of point: {point} ({type(point)})")

        reflected_p = y0[0] > np.pi / 4 + epsilon
        if reflected_p:
            y0[0] = np.pi / 2 - y0[0]

        # short circuit in codimension 0
        if self.member(y0):
            if reflected_p:
                y0[0] = np.pi / 2 - y0[0]
            return y0

        # codimension 1
        b1 = self._offsets.reshape(7, 1)
        A1y0 = np.einsum('ijk,k->ij', A1, y0)
        nearest1 = np.einsum('ijk,ik->ij', A1inv, b1 - A1y0) + y0

        # codimension 2
        b2 = np.array([*itertools.combinations(self._offsets, 2)])
        A2y0 = np.einsum('ijk,k->ij', A2, y0)
        nearest2 = np.einsum('ijk,ik->ij', A2inv, b2 - A2y0) + y0

        # codimension 3
        b3 = np.array([*itertools.combinations(self._offsets, 3)])
        nearest3 = np.einsum('ijk,ik->ij', A3inv, b3)

        # pick the nearest
        nearest = -np.concatenate([nearest1, nearest2, nearest3])
        nearest = nearest[self.member(nearest)]
        smallest_index = np.argmin(np.linalg.norm(nearest - y0, axis=1))

        if reflected_p:
            nearest[smallest_index][0] = np.pi / 2 - nearest[smallest_index][0]
        return nearest[smallest_index]
