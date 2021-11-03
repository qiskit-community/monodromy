"""
monodromy/backend/lrs.py

Communication interface for `lrs`, a package for convex hull problems.

More information about `lrs`: http://cgm.cs.mcgill.ca/~avis/C/lrs.html
"""


from copy import copy
from fractions import Fraction
from functools import reduce
import math  # for gcd
from operator import itemgetter
from os import getenv
from subprocess import Popen, PIPE
from typing import List

from .backend_abc import Backend
from ..exceptions import NoFeasibleSolutions
from ..polytopes import ConvexPolytope, PolytopeVolume
from ..utilities import lcm


LRS_ENV = "LRS_PATH"
"""Environment variable used to override the path to the `lrs` executable."""


LRS_PATH = getenv(LRS_ENV, "lrs")
"""Default path to the user's `lrs` executable."""


def check_for_lrs():
    """
    Checks whether `lrs` is findable and executable.
    """
    try:
        proc = Popen([LRS_PATH], stdin=PIPE, stdout=PIPE, stderr=PIPE)
        proc.communicate(b"")
        return True
    except Exception:  # FileNotFoundError, for instance
        return False


class LRSBackend(Backend):
    def __init__(self):
        super().__init__()
        assert check_for_lrs(), "LRS not present."

    @staticmethod
    def volume(convex_polytope: ConvexPolytope) -> PolytopeVolume:
        if 0 == len(convex_polytope.vertices):
            raise NoFeasibleSolutions()

        vertices = [[Fraction(1, 1), *x] for x in convex_polytope.vertices]
        vertex_payload = encode_vertices(vertices)
        inequality_response = single_lrs_pass(vertex_payload)
        inequality_dictionary = decode_inequalities(inequality_response)
        return PolytopeVolume(
            volume=inequality_dictionary["volume"],
            dimension=inequality_dictionary["dimension"],
        )

    # NOTE: This method uses the `redund` option, which is a recent addition to
    #       `lrs` and may be buggy.  The (commented out) variant of this method
    #       below performs the same computation without `redund`, but is slower.
    # @staticmethod
    # def reduce(convex_polytope: ConvexPolytope) -> ConvexPolytope:
    #     clone = copy(convex_polytope)
    #
    #     inequalities = convex_polytope.inequalities
    #     equalities = convex_polytope.equalities
    #     inequality_payload = encode_inequalities(
    #         inequalities, equalities,
    #         options=["redund 0 0"]  # lrs ≥ 7.1
    #     )
    #     inequality_response = single_lrs_pass(inequality_payload)
    #     inequality_dictionary = decode_inequalities(inequality_response)
    #
    #     clone.inequalities = inequality_dictionary["inequalities"]
    #     clone.equalities = inequality_dictionary["equalities"]
    #
    #     return clone

    # NOTE: This method does not use the `redund` option, which is a recent
    #       addition to `lrs` and may be buggy.  The variant of this method
    #       above performs the same computation with `redund`, so is faster.
    @staticmethod
    def reduce(convex_polytope: ConvexPolytope) -> ConvexPolytope:
        clone = copy(convex_polytope)

        inequalities = convex_polytope.inequalities
        equalities = convex_polytope.equalities
        inequality_payload = encode_inequalities(
            inequalities, equalities,
        )
        vertex_response = single_lrs_pass(inequality_payload)
        vertices = decode_vertices(vertex_response)
        vertex_payload = encode_vertices(vertices)
        inequality_response = single_lrs_pass(vertex_payload)
        inequality_dictionary = decode_inequalities(inequality_response)

        clone.inequalities = inequality_dictionary["inequalities"]
        clone.equalities = inequality_dictionary["equalities"]

        return clone

    @staticmethod
    def vertices(convex_polytope: ConvexPolytope) -> List[List[Fraction]]:
        inequalities = convex_polytope.inequalities
        equalities = convex_polytope.equalities
        inequality_payload = encode_inequalities(inequalities, equalities)
        vertex_response = single_lrs_pass(inequality_payload)
        vertices = decode_vertices(vertex_response)
        if any([v[0] == 0 for v in vertices]):
            raise ValueError("Polytope is not bounded.")

        return [v[1:] for v in vertices]

    @staticmethod
    def triangulation(convex_polytope: ConvexPolytope) -> List[List]:
        if 0 == len(convex_polytope.vertices):
            raise NoFeasibleSolutions()

        vertex_payload = encode_vertices([(Fraction(1, 1), *v)
                                          for v in convex_polytope.vertices],
                                         options=["triangulation"])
        response = single_lrs_pass(vertex_payload)
        simplices = decode_simplices(response)
        return simplices["simplices"]

    @staticmethod
    def convex_hull(vertices: List[List[Fraction]]) -> ConvexPolytope:
        payload = encode_vertices([(1, *x) for x in vertices])
        response = single_lrs_pass(payload)
        inequalities, equalities = itemgetter("inequalities", "equalities")(
            decode_inequalities(response)
        )
        return ConvexPolytope(
            inequalities=inequalities,
            equalities=equalities,
        )


def single_lrs_pass(payload: bytes, chatty=False) -> bytes:
    """Generic wrapper for lrs."""
    if chatty:
        print("=== LRS CALL ===")
        print("Payload:")
        print(payload.decode())
    proc = Popen([LRS_PATH], stdin=PIPE, stdout=PIPE, stderr=PIPE)
    stdout, stderr = proc.communicate(payload)
    # TODO: You could do something with stderr, but beware: if lrs restarts with
    #       a different arithmetic type, it puts some chatter on that stream.
    if chatty:
        print("Response:")
        print(stdout.decode())
    return stdout


def encode_inequalities(inequalities, equalities=None, name="name",
                        options=None) -> bytes:
    """Format `inequalities` for consumption by lrs."""
    equalities = equalities if equalities is not None else []
    options = options if options is not None else []
    output = ""
    output += name + "\n"
    output += "H-representation\n"
    # if 0 < len(equalities):
    #     output += f"linearity {len(equalities)} " \
    #               f"{' '.join(range(1, 1 + len(equalities)))}\n"
    output += "begin\n"
    output += (f"{len(inequalities) + 2*len(equalities)}"
               f" {len((inequalities + equalities)[0])}"
               " rational\n")
    for row in inequalities + equalities + [[-x for x in eq] for eq in equalities]:
        row_gcd = abs(reduce(math.gcd, row))
        row_gcd = row_gcd if row_gcd != 0 else 1
        output += " ".join([str(x // row_gcd) for x in row]) + "\n"
    output += "end\n"
    for option in options:
        output += f"{option}\n"

    return output.encode()


def decode_inequalities(lrs_output: bytes):
    """Parse lrs output (an `H-representation`) into python data."""
    volume = None
    rows = []
    name = None
    equality_indices = []
    invocation_signature = None
    break_at_end = False
    for line in lrs_output.decode('utf-8').splitlines():
        # initialize
        if line.startswith('*lrs') and line != invocation_signature:
            name = None
            invocation_signature = line
        # stash the Volume output
        if line.startswith('*Volume='):
            if volume is None:
                volume = Fraction(line[8:])
            continue
        # ignore comments
        if line.startswith('*') or line == '':
            continue
        # first non-comment line is our name
        if name is None:
            name = line
            continue
        # ignore begin / end, assume they're in the right place
        if line.startswith('end'):
            if break_at_end:
                break
            else:
                continue
        if line.startswith('begin'):
            rows = []
            continue
        # skip the table size, if it's present
        if 'rational' in line:
            continue
        # skip echoed option
        if line.startswith('redund'):
            break_at_end = True
            continue
        # check that we're looking at the right kind of representation
        if line == 'H-representation':
            continue
        if line == 'V-representation':
            raise ValueError("Inequality table decoder got a vertex table as input")
        # if we produced a degenerate polytope, note the indices
        if line.startswith('linearity'):
            equality_indices = [int(x) for x in line[9:].split()[1:]]
            continue

        new_row = [Fraction(x) for x in line.split()]
        row_lcm = abs(lcm(*[x.denominator for x in new_row]))
        rows.append([int(x * row_lcm) for x in new_row])

    if 0 == len(rows):
        if break_at_end:
            raise NoFeasibleSolutions()
        else:
            print(lrs_output.decode('utf-8'))
            raise TypeError("Something bad happened in `lrs`.")

    return dict(
        inequalities=[row for index, row in enumerate(rows)
                      if 1 + index not in equality_indices],
        equalities=[row for index, row in enumerate(rows)
                    if 1 + index in equality_indices],
        volume=volume,
        dimension=len(rows[0]) - 1 - len(equality_indices),
    )


def decode_simplices(lrs_output: bytes):
    """Parse lrs output from a tetrahedral run into python data."""
    simplices = []
    for line in lrs_output.decode('utf-8').splitlines():
        # initialize
        if line.startswith('*lrs'):
            simplices = []
        if line.startswith('end'):
            break
        if line.startswith('F#'):
            tokens = line.split()
            position = tokens.index("vertices/rays")
            indices = []
            while True:
                position += 1
                try:
                    indices.append(int(tokens[position]) - 1)
                except ValueError:
                    break
            simplices.append(indices)

    return dict(
        simplices=simplices
    )


def encode_vertices(vertices, name="name", options=None) -> bytes:
    """Format `vertices` for consumption by lrs."""
    options = [] if options is None else options
    output = ""
    output += name + "\n"
    output += "V-representation\n"
    output += "begin\n"
    output += f"{len(vertices)} {len(vertices[0])} rational\n"
    for vertex in vertices:
        output += " ".join([str(x) for x in vertex]) + "\n"
    output += "end\n"
    output += "volume\n"
    for option in options:
        output += f"{option}\n"

    return output.encode()


def decode_vertices(lrs_output: bytes):
    """Parse lrs output (a `V-representation`) into python data."""
    invocation_signature = None
    name = None
    vertices = []
    for line in lrs_output.decode('utf-8').splitlines():
        if line.startswith('*lrs') and line != invocation_signature:
            name = None
            invocation_signature = line
        # ignore comments
        if line.startswith('*') or line == '':
            continue
        # first non-comment line is our name
        if name is None:
            name = line
            continue
        # ignore begin / end, assume they're in the right place
        if line.startswith('end'):
            continue
        if line.startswith('begin'):
            vertices = []
            continue
        if line == 'V-representation':
            continue
        if line == 'H-representation':
            raise ValueError("Vertex table decoder got an inequality table as input")
        if line.startswith('linearity'):
            continue
        if line.startswith("No feasible solution"):
            raise NoFeasibleSolutions()
        vertices.append([Fraction(x) for x in line.split()])

    if 0 == len(vertices):
        print(lrs_output.decode('utf-8'))
        raise TypeError("Something bad happened in `lrs`.")

    return vertices
