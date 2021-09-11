"""
monodromy/backend/backend_abc.py

A generic backend specification for polytope computations.
"""

from abc import ABC, abstractmethod


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
    def triangulation(convex_polytope):  # ConvexPolytope -> List[List[int]]
        """
        Calculates a triangulation of the input ConvexPolytope.  Returns a list
        of simplices, each specified as a list of its vertices, in turn each
        specified as the index into .vertices at which it appears.
        """
        pass

    @staticmethod
    @abstractmethod
    def convex_hull(vertices):  # List[List[Fraction]] -> ConvexPolytope
        """
        Produces a minimal ConvexPolytope from a set of vertices.
        """
        pass
