"""
monodromy/render.py

Utilities for rendering polytopes.
"""

from typing import List

from monodromy.coverage import CircuitPolytope


def polytopes_to_mathematica(necessary_polytopes: List[CircuitPolytope]):
    output = ""
    output += "polytopeData = {"
    for n, gate_polytope in enumerate(necessary_polytopes):
        output += "{"
        output += f"{float(gate_polytope.cost)}, "
        for i, polytope in enumerate(gate_polytope.convex_subpolytopes):
            output += "{"
            vertices = polytope.vertices
            for j, vertex in enumerate(vertices):
                output += "{"
                output += f"{vertex[0]}, {vertex[1]}, {vertex[2]}"
                if 1 + j != len(vertices):
                    output += "}, "
                else:
                    output += "}"
            if 1 + i != len(gate_polytope.convex_subpolytopes):
                output += "}, "
            else:
                output += "}"
        if 1 + n != len(necessary_polytopes):
            output += "}, "
        else:
            output += "}"
    output += "};"

    output += """
corners = {{0, 0, 0}, {1/4, 1/4, 1/4}, {1/4, 1/4, -(1/4)}, {3/8, 3/
    8, -(1/8)}, {3/8, -(1/8), -(1/8)}, {1/2, 0, 0}};
names = {{0, 0, 0} -> "I", {1/4, 1/4, -1/4} -> "CZ", {1/2, 0, 0} -> 
    "ISWAP", {1/4, 1/4, 1/4} -> "SWAP", {3/8, 3/8, -1/8} -> Sqrt[
    SWAP], {3/8, -1/8, -1/8} -> Sqrt[SWAP]'};

(* tune this loop bound to skip degenerate solids *)
skipTo := 6
(* tune these scalars to get a picture w/o Z-fighting *)
OffsetCoords[coord_, n_] := ((1 + 0.5^n) (coord - {0.25, 0.1, -0.1}))

Module[{directives = {}, n, depth, vertices, maxdepth},
  maxdepth = Max[First /@ polytopeData];  
  For[n = skipTo, n <= Length[polytopeData], n++,
   (* inject new color settings *)
   depth = polytopeData[[n, 1]];
   vertices = Rest[polytopeData[[n]]];
   directives = Join[directives, {
      Hue[(depth - skipTo)/maxdepth],
      Opacity[1.0 - (depth - skipTo)/maxdepth]
      }];
   (* add new polyhedra *)
   directives = Join[directives,
     With[{mesh = ConvexHullMesh[OffsetCoords[#, n] & /@ #]}, 
        GraphicsComplex[
         MeshCoordinates[mesh], {EdgeForm[], MeshCells[mesh, 2]}]] & /@
       vertices];
   ];
  directives];
Show[Graphics3D[Lighting -> "Neutral", Boxed -> False], 
 Graphics3D@(Join @@ 
    Table[{Sphere[OffsetCoords[corner, skipTo], 0.02], 
      Text[corner /. names, 
       OffsetCoords[corner, 5 skipTo] + 
        0.05*If[Norm[corner] == 0, 0, corner/Norm[corner]]]}, {corner,
       corners}]),
 Graphics3D[%, Boxed -> False, ViewPoint -> {0, 1, 1}, 
  Lighting -> {{"Ambient", White}}]]
"""

    return output
