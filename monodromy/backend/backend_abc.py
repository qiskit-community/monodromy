"""
monodromy/backend/backend_abc.py

A generic backend specification for polytope computations.
"""

from abc import ABC, abstractmethod


class NoFeasibleSolutions(Exception):
    """Emitted when reducing a convex polytope with no solutions."""
    pass


class Backend(ABC):
    """
    Generic backend interface for polytope procedures.
    """

    @staticmethod
    @abstractmethod
    def volume(convex_polytope):  # ConvexPolytope -> PolytopeVolume
        """
        Calculates the Eucliean volume of the ConvexPolytope.

        Signals `NoFeasibleSolutions` if the ConvexPolytope has no solutions.
        """
        pass

    @staticmethod
    @abstractmethod
    def reduce(convex_polytope):  # ConvexPolytope -> ConvexPolytope
        """
        Calculates a minimum set of inequalities specifying the ConvexPolytope.

        Signals `NoFeasibleSolutions` if the ConvexPolytope has no solutions.
        """
        pass

    @staticmethod
    @abstractmethod
    def vertices(convex_polytope):  # ConvexPolytope -> List[List[Fraction]]
        """
        Calculates the vertices of the ConvexPolytope.

        Signals `NoFeasibleSolutions` if the ConvexPolytope has no solutions.
        """
        pass

    @staticmethod
    @abstractmethod
    def triangulation(vertices):  # List[List[Fraction]] -> List[List]
        """
        Given a set of vertices, calculate a triangulation of their convex hull.
        Returns a list of 4-tuples of indices into the vertex sequence.
        """
        pass
