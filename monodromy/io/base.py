"""
monodromy/io/base.py

Bare dataclasses which house polytope information.
"""

from dataclasses import dataclass, field
from typing import List


anonymous_convex_polytope_counter = 0


def generate_anonymous_cp_name():
    global anonymous_convex_polytope_counter
    anonymous_convex_polytope_counter += 1
    return f"anonymous_convex_polytope_{anonymous_convex_polytope_counter}"


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
    name: str = field(default_factory=generate_anonymous_cp_name)

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
class CircuitPolytopeData(PolytopeData):
    """
    A polytope describing the alcove coverage of a particular circuit type.
    """
    cost: float
    operations: List[str]
