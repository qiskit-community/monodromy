"""
monodromy/static/qlr_table.py

Precomputed descriptions of the monodromy polytope for SU(4) (and PU(4)).
"""

from ..polytopes import make_convex_polytope


#              r  k   a    b    c   d
qlr_table = [[1, 3, [0], [0], [0], 0],
             [1, 3, [0], [1], [1], 0],
             [1, 3, [0], [2], [2], 0],
             [1, 3, [0], [3], [3], 0],
             [1, 3, [1], [1], [2], 0],
             [1, 3, [1], [2], [3], 0],
             [1, 3, [1], [3], [0], 1],
             [1, 3, [2], [2], [0], 1],
             [1, 3, [2], [3], [1], 1],
             [1, 3, [3], [3], [2], 1],
             # r k    a       b       c     d
             [2, 2, [0, 0], [0, 0], [0, 0], 0],
             [2, 2, [0, 0], [1, 0], [1, 0], 0],
             [2, 2, [0, 0], [1, 1], [1, 1], 0],
             [2, 2, [0, 0], [2, 0], [2, 0], 0],
             [2, 2, [0, 0], [2, 1], [2, 1], 0],
             [2, 2, [0, 0], [2, 2], [2, 2], 0],
             [2, 2, [1, 0], [1, 0], [1, 1], 0],
             [2, 2, [1, 0], [1, 0], [2, 0], 0],
             [2, 2, [1, 0], [1, 1], [2, 1], 0],
             [2, 2, [1, 0], [2, 0], [2, 1], 0],
             [2, 2, [1, 0], [2, 1], [2, 2], 0],
             [2, 2, [1, 0], [2, 1], [0, 0], 1],
             [2, 2, [1, 0], [2, 2], [1, 0], 1],
             [2, 2, [1, 1], [1, 1], [2, 2], 0],
             [2, 2, [1, 1], [2, 0], [0, 0], 1],
             [2, 2, [1, 1], [2, 1], [1, 0], 1],
             [2, 2, [1, 1], [2, 2], [2, 0], 1],
             [2, 2, [2, 0], [2, 0], [2, 2], 0],
             [2, 2, [2, 0], [2, 1], [1, 0], 1],
             [2, 2, [2, 0], [2, 2], [1, 1], 1],
             [2, 2, [2, 1], [2, 1], [2, 0], 1],
             [2, 2, [2, 1], [2, 1], [1, 1], 1],
             [2, 2, [2, 1], [2, 2], [2, 1], 1],
             [2, 2, [2, 2], [2, 2], [0, 0], 2],
             # r k      a          b          c      d
             [3, 1, [0, 0, 0], [0, 0, 0], [0, 0, 0], 0],
             [3, 1, [0, 0, 0], [1, 0, 0], [1, 0, 0], 0],
             [3, 1, [0, 0, 0], [1, 1, 0], [1, 1, 0], 0],
             [3, 1, [0, 0, 0], [1, 1, 1], [1, 1, 1], 0],
             [3, 1, [1, 0, 0], [1, 0, 0], [1, 1, 0], 0],
             [3, 1, [1, 0, 0], [1, 1, 0], [1, 1, 1], 0],
             [3, 1, [1, 0, 0], [1, 1, 1], [0, 0, 0], 1],
             [3, 1, [1, 1, 0], [1, 1, 0], [0, 0, 0], 1],
             [3, 1, [1, 1, 0], [1, 1, 1], [1, 0, 0], 1],
             [3, 1, [1, 1, 1], [1, 1, 1], [1, 1, 0], 1]]
"""
Precomputed table of quantum Littlewood-Richardson coefficients for the small
quantum cohomology ring of k-planes in C^4, 0 < k < 4.  Each entry is of the
form [r, k, [*a], [*b], [*c], d], corresponding to the relation

    N_{ab}^{c, d} = 1   or   <sigma_a, sigma_b, sigma_{*c}> = q^d.

NOTE: We include only entries with a <= b in the traversal ordering used by
      `monodromy.io.lrcalc.displacements`.

NOTE: This table can be regenerated using
      `monodromy.io.lrcalc.regenerate_qlr_table` .
"""


def ineq_from_qlr(r, k, a, b, c, d):
    """
    Generates a monodromy polytope inequality from the position of a nonzero
    quantum Littlewood-Richardson coefficient for su_4.

    See (*) in Theorem 23 of /1904.10541 .

    NOTE: `r` is ignored, since `4 = r + k` makes it redundant with `k`.
    """

    # $$d - \sum_{i=1}^r \alpha_{k+i-a_i}
    #     - \sum_{i=1}^r \beta_{k+i-b_i}
    #     + \sum_{i=1}^r \delta_{k+i-c_i} \ge 0$$

    new_row = [d,
               0, 0, 0, 0,  # alpha's
               0, 0, 0, 0,  # beta's
               0, 0, 0, 0, ]  # gamma's
    for i, ai in enumerate(a):
        index = k + (i + 1) - ai  # subscript in the Biswas inequality
        offset = 0  # last entry before alpha
        new_row[index + offset] -= 1  # poke the value in
    for i, bi in enumerate(b):
        index = k + (i + 1) - bi  # subscript in the Biswas inequality
        offset = 4  # last entry before beta
        new_row[index + offset] -= 1  # poke the value in
    for i, ci in enumerate(c):
        index = k + (i + 1) - ci  # subscript in the Biswas inequality
        offset = 8  # last entry before gamma
        new_row[index + offset] += 1  # poke the value in

    # now remember that a4 = -a1-a2-a3 and so on
    new_row = [new_row[0],
               *[x - new_row[4] for x in new_row[1:4]],
               *[x - new_row[8] for x in new_row[5:8]],
               *[x - new_row[12] for x in new_row[9:12]]
               ]

    return new_row


def generate_qlr_inequalities():
    """
    Regenerates the set of monodromy polytope inequalities from the stored table
    `qlr_table` of quantum Littlewood-Richardson coefficients.
    """
    qlr_inequalities = []
    for r, k, a, b, c, d in qlr_table:
        qlr_inequalities.append(ineq_from_qlr(r, k, a, b, c, d))
        if a != b:
            qlr_inequalities.append(ineq_from_qlr(r, k, b, a, c, d))

    return qlr_inequalities


qlr_polytope = make_convex_polytope(
    generate_qlr_inequalities(),
    name="QLR relations"
)
"""
This houses the monodromy polytope, the main static input of the whole calc'n.
This polytope does _not_ also contain the alcove constraints.
"""
