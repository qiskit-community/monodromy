"""
monodromy/exceptions.py

Exception classes used throughout the project.
"""


class NoBacksolution(Exception):
    """
    Signaled when the circuit backsolver can't find a suitable preimage point.

    Conjectured to be probabilistically meaningless: should be fine to re-run
    the call after catching this error.
    """
    pass


class NoFeasibleSolutions(Exception):
    """Emitted when reducing a convex polytope with no solutions."""
    pass
