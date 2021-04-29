"""
monodromy/backend/lrs.py

Communication interface for `lrs`, a package for convex hull problems.

More information about `lrs`: http://cgm.cs.mcgill.ca/~avis/C/lrs.html
"""


from fractions import Fraction
import math  # gcd
from os import getenv
from subprocess import Popen, PIPE
from typing import List, Tuple

from .backend_abc import Backend, NoFeasibleSolutions
from ..polytopes import ConvexPolytope, PolytopeVolume


"""Environment variable used to override the path to the `lrs` executable."""
LRS_ENV = "LRS_PATH"


"""Default path to the user's `lrs` executable."""
LRS_PATH = getenv(LRS_ENV, "lrs")


class LRSBackend(Backend):
    def __init__(self):
        super().__init__()
        assert check_for_lrs(), "LRS not present."

    @staticmethod
    def volume(convex_polytope: ConvexPolytope) -> PolytopeVolume:
        inequalities = convex_polytope.inequalities
        equalities = convex_polytope.equalities
        inequality_payload = encode_inequalities(inequalities, equalities)
        vertex_response = single_lrs_pass(inequality_payload)
        vertices = decode_vertices(vertex_response)
        vertex_payload = encode_vertices(vertices)
        inequality_response = single_lrs_pass(vertex_payload)
        inequality_dictionary = decode_inequalities(inequality_response)
        return PolytopeVolume(
            volume=inequality_dictionary["volume"],
            dimension=inequality_dictionary["dimension"],
        )

    @staticmethod
    def reduce(convex_polytope: ConvexPolytope) -> ConvexPolytope:
        inequalities = convex_polytope.inequalities
        equalities = convex_polytope.equalities
        inequality_payload = encode_inequalities(inequalities, equalities)
        vertex_response = single_lrs_pass(inequality_payload)
        vertices = decode_vertices(vertex_response)
        vertex_payload = encode_vertices(vertices)
        inequality_response = single_lrs_pass(vertex_payload)
        inequality_dictionary = decode_inequalities(inequality_response)
        return ConvexPolytope(
            inequalities=inequality_dictionary["inequalities"],
            equalities=inequality_dictionary["equalities"],
        )

    @staticmethod
    def vertices(convex_polytope: ConvexPolytope) -> List[List[Fraction]]:
        inequalities = convex_polytope.inequalities
        equalities = convex_polytope.equalities
        inequality_payload = encode_inequalities(inequalities, equalities)
        vertex_response = single_lrs_pass(inequality_payload)
        vertices = decode_vertices(vertex_response)
        return [v[1:] for v in vertices]


def check_for_lrs():
    try:
        proc = Popen([LRS_PATH], stdin=PIPE, stdout=PIPE, stderr=PIPE)
        proc.communicate(b"")
        return True
    except Exception:  # FileNotFoundError, for instance
        return False


def single_lrs_pass(payload: bytes) -> bytes:
    """Generic wrapper for lrs."""
    proc = Popen([LRS_PATH], stdin=PIPE, stdout=PIPE, stderr=PIPE)
    stdout, stderr = proc.communicate(payload)
    # TODO: You could do something with stderr, but beware: if lrs restarts with
    #       a different arithmetic type, it puts some chatter on that stream.
    return stdout


def encode_inequalities(inequalities, equalities=None, name="name") -> bytes:
    """Format `inequalities` for consumption by lrs."""
    equalities = equalities if equalities is not None else []
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
        gcd = 1
        for x in row:
            gcd = math.gcd(gcd, x.denominator)
        gcd = abs(gcd)
        output += " ".join([str(x * gcd) for x in row]) + "\n"
    output += "end\n"

    return output.encode()


def decode_inequalities(lrs_output: bytes):
    """Parse lrs output (an `H-representation`) into python data."""
    volume = None
    rows = []
    name = None
    equality_indices = []
    invocation_signature = None
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
            continue
        if line.startswith('begin'):
            rows = []
            continue
        # skip the table size, if it's present
        if 'rational' in line:
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
        rows.append([Fraction(x) for x in line.split()])

    if 0 == len(rows):
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


def encode_vertices(vertices, name="name") -> bytes:
    """Format `vertices` for consumption by lrs."""
    output = ""
    output += name + "\n"
    output += "V-representation\n"
    output += "begin\n"
    output += f"{len(vertices)} {len(vertices[0])} rational\n"
    for vertex in vertices:
        output += " ".join([str(x) for x in vertex]) + "\n"
    output += "end\n"
    output += "volume\n"

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
