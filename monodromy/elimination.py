"""
monodromy/elimination.py

Implements coordinate-wise inclusion and projections of inequality families.
"""

from typing import List

from .polytopes import Polytope, ConvexPolytope


def cylinderize(
        polytope: Polytope,
        coordinate_map: List[int],
        parent_dimension: int = 10):
    """
    Consumes a `polytope` and a list of n coordinates on Q^m, and emits a
    polytope cylinderized along the complement of those Q^m coordinates.
    """

    cylinderized_polytope = Polytope(convex_subpolytopes=[])
    for convex_subpolytope in polytope.convex_subpolytopes:
        cylinderized_subpolytope = ConvexPolytope(
            inequalities=[],
            name=convex_subpolytope.name
        )
        for inequality in convex_subpolytope.inequalities:
            new_row = [0] * parent_dimension
            for source_value, target_index in zip(inequality, coordinate_map):
                new_row[target_index] += source_value
            cylinderized_subpolytope.inequalities.append(new_row)
        for equality in convex_subpolytope.equalities:
            new_row = [0] * parent_dimension
            for source_value, target_index in zip(equality, coordinate_map):
                new_row[target_index] += source_value
            cylinderized_subpolytope.equalities.append(new_row)
        cylinderized_polytope.convex_subpolytopes.append(
            cylinderized_subpolytope
        )

    return cylinderized_polytope


def project(polytope, index):
    """
    Returns the projection of `polytope` away from coordinate `index`.

    Implements the (naive) Fourier-Motzkin elimination algorithm; see

        https://en.wikipedia.org/wiki/Fourier%E2%80%93Motzkin_elimination

    for more details.

    NOTE: Some pairs of inequalities of the result may belong to equalities.
          To collect these equalities, call reduce.
    NOTE: lrs supposedly supports this, but they note that others have reported
          bugs and so suggest that users don't engage with it.
    """

    projected_polytope = Polytope(convex_subpolytopes=[])

    for convex_subpolytope in polytope.convex_subpolytopes:
        # F-M collects inequalities into three buckets:
        # those with the `index` summand zero, positive, and negative.
        zero_equalities = []
        zero_inequalities = []
        positive_inequalities = []
        negative_inequalities = []
        for inequality in convex_subpolytope.inequalities:
            if 0 == inequality[index]:
                zero_inequalities.append(inequality)
            elif 0 < inequality[index]:
                positive_inequalities.append(inequality)
            elif 0 > inequality[index]:
                negative_inequalities.append(inequality)
            else:
                raise TypeError(f"Switch failure on {inequality[index]}")
        for equality in convex_subpolytope.equalities:
            if 0 == equality[index]:
                zero_equalities.append(equality)
            elif 0 < equality[index]:
                positive_inequalities.append(equality)
                negative_inequalities.append([-x for x in equality])
            elif 0 > equality[index]:
                negative_inequalities.append(equality)
                positive_inequalities.append([-x for x in equality])
            else:
                raise TypeError(f"Switch failure on {equality[index]}")

        joined_inequalities = []
        for positive_inequality in positive_inequalities:
            for negative_inequality in negative_inequalities:
                # `positive_inequality` can be written as `1 * xj >= pos_rest`,
                # `negative_inequality` can be written as `neg_rest >= 1 * xj`.
                # Each pair contributes an inequality `neg_rest >= pos_rest`,
                # or `neg_rest - pos_rest >= 0` .

                pos_scalar = positive_inequality[index]
                neg_scalar = negative_inequality[index]
                joined_inequality = [p * -neg_scalar + n * pos_scalar
                                     for (p, n) in zip(positive_inequality, negative_inequality)]
                joined_inequality = joined_inequality[:index] + joined_inequality[1+index:]

                joined_inequalities.append(joined_inequality)

        # For the remainder, we just ignore the unwanted coordinate.
        zero_inequalities = [z[:index] + z[1+index:] for z in zero_inequalities]
        zero_equalities = [z[:index] + z[1+index:] for z in zero_equalities]

        projected_polytope.convex_subpolytopes.append(
            ConvexPolytope(inequalities=zero_inequalities + joined_inequalities,
                           equalities=zero_equalities,
                           name=convex_subpolytope.name)
        )

    return projected_polytope
