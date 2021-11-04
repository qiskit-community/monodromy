"""
test/test_polytopes.py

Tests for monodromy/polytopes.py .
"""

import ddt
import unittest

from monodromy.polytopes import *

epsilon = 0.001


@ddt.ddt
class TestMonodromyConvexPolytopes(unittest.TestCase):
    """Check various convex polytope routines."""

    cube = ConvexPolytope(
        inequalities=[
            [1, -1,  0,  0],
            [1,  1,  0,  0],
            [1,  0, -1,  0],
            [1,  0,  1,  0],
            [1,  0,  0, -1],
            [1,  0,  0,  1],
        ],
    )

    def test_volume(self):
        self.assertEqual(PolytopeVolume(3, Fraction(8)),
                         self.cube.volume)

    def test_vertices(self):
        self.assertEqual(
            set([tuple(v) for v in self.cube.vertices]),
            {(-1, -1, -1), (-1, -1,  1), (-1,  1, -1), (-1,  1,  1),
             ( 1, -1, -1), ( 1, -1,  1), ( 1,  1, -1), ( 1,  1,  1)})

    def test_triangulation(self):
        tetrahedralized_cube = []
        for tetrahedron_indices in self.cube.triangulation:
            tetrahedron_vertices = [self.cube.vertices[i] for i in tetrahedron_indices]
            tetrahedron = ConvexPolytope.convex_hull(tetrahedron_vertices)
            tetrahedralized_cube.append(tetrahedron)
            self.assertTrue(self.cube.contains(tetrahedron))

        total_volume = sum((x.volume for x in tetrahedralized_cube),
                           PolytopeVolume(3, Fraction(0)))
        self.assertEqual(total_volume, PolytopeVolume(3, Fraction(8)))

    def test_reduce(self):
        overspecified_cube = ConvexPolytope(
            inequalities=self.cube.inequalities + [
                [5, 1, 0, 0], [5, 0, 1, 0], [5, 0, 0, 1],
            ]
        )
        reduced_cube = overspecified_cube.reduce()
        self.assertTrue(reduced_cube.contains(self.cube))
        self.assertTrue(self.cube.contains(reduced_cube))
        self.assertTrue(all([x in self.cube.inequalities
                             for x in reduced_cube.inequalities]))

    def test_intersect(self):
        shifted_cube = ConvexPolytope(inequalities=[
            [0,  1,  0,  0],
            [2, -1,  0,  0],
            [0,  0,  1,  0],
            [2,  0, -1,  0],
            [0,  0,  0,  1],
            [2,  0,  0, -1],
        ])
        intersected_cubes = self.cube.intersect(shifted_cube).reduce()
        clipped_cube = ConvexPolytope(inequalities=[
            [0,  1,  0,  0],
            [1, -1,  0,  0],
            [0,  0,  1,  0],
            [1,  0, -1,  0],
            [0,  0,  0,  1],
            [1,  0,  0, -1],
        ])
        self.assertTrue(clipped_cube.contains(intersected_cubes))
        self.assertTrue(intersected_cubes.contains(clipped_cube))

    def test_empty_intersection(self):
        unsatisfiable_polytope = self.cube.intersect(ConvexPolytope(
            inequalities=[[-9, 1, 0, 0]])
        )
        with self.assertRaises(NoFeasibleSolutions):
            unsatisfiable_polytope.reduce()

    def test_has_element(self):
        self.assertTrue(self.cube.has_element([0, 0, 0]))
        self.assertFalse(self.cube.has_element([10, -10, 20]))


@ddt.ddt
class TestMonodromyPolytopes(unittest.TestCase):
    """Check various non-convex polytope routines."""

    overlapping_cubes = make_convex_polytope([
            [1, -1,  0,  0],  # cube [-1, 1]^(x 3)
            [1,  1,  0,  0],
            [1,  0, -1,  0],
            [1,  0,  1,  0],
            [1,  0,  0, -1],
            [1,  0,  0,  1],
        ]).union(make_convex_polytope([
            [0,  1,  0,  0],  # cube [0, 2]^(x 3)
            [2, -1,  0,  0],
            [0,  0,  1,  0],
            [2,  0, -1,  0],
            [0,  0,  0,  1],
            [2,  0,  0, -1],
        ]))

    def test_volume(self):
        self.assertEqual(PolytopeVolume(3, Fraction(15)),
                         self.overlapping_cubes.volume)

    def test_reduce_eliminates_convex_components(self):
        redundant_polytope = self.overlapping_cubes.union(make_convex_polytope([
            [0,  1,  0,  0],  # cube [0, 1]^(x 3)
            [1, -1,  0,  0],
            [0,  0,  1,  0],
            [1,  0, -1,  0],
            [0,  0,  0,  1],
            [1,  0,  0, -1],
        ]))
        result = redundant_polytope.reduce()
        self.assertTrue(self.overlapping_cubes.contains(result))
        self.assertTrue(result.contains(self.overlapping_cubes))
        self.assertTrue(len(result.convex_subpolytopes) <
                        len(redundant_polytope.convex_subpolytopes))

    def test_union_PIE(self):
        self.assertEqual(
            self.overlapping_cubes.volume, (
                self.overlapping_cubes.convex_subpolytopes[0].volume +
                self.overlapping_cubes.convex_subpolytopes[1].volume -
                self.overlapping_cubes.convex_subpolytopes[0].intersect(
                    self.overlapping_cubes.convex_subpolytopes[1]
                ).volume
            )
        )
        self.assertTrue(self.overlapping_cubes.contains(
            make_convex_polytope(
                self.overlapping_cubes.convex_subpolytopes[0].inequalities
            )
        ))
        self.assertTrue(self.overlapping_cubes.contains(
            make_convex_polytope(
                self.overlapping_cubes.convex_subpolytopes[1].inequalities
            )
        ))

    def test_self_intersection(self):
        self.assertEqual(
            self.overlapping_cubes.volume,
            self.overlapping_cubes.intersect(
                self.overlapping_cubes
            ).volume
        )
