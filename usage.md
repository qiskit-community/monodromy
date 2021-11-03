# `monodromy`: User's Guide

## Motivation

Here are two common questions among people interested in gate set design / quantum compilation / ... :

1. Which two-qubit gates can be simulated by a particular sequence of two-qubit entanglers (a "circuit shape"))?
2. Given a gate set, what is the cheapest circuit which enacts a given two-qubit operation?

If single-qubit gates are considered to be "cheap", then it suffices to be able to answer these questions for "canonical gates": $$\mathit{CAN}(x, y, z) = \exp\left(-i \frac{\pi}{2} (x \cdot XX + y \cdot YY + z \cdot ZZ)\right).$$

The main result from [Peterson–Crooks–Smith](https://quantum-journal.org/papers/q-2020-03-26-247/) is:

> **Theorem:** The set of triples $(a, b, c)$ for which there is a solution to
> 
> $$\mathrm{local} \cdot \mathit{CAN}(a) \cdot \mathrm{local} \cdot \mathit{CAN}(b) \cdot \mathrm{local} = \mathit{CAN}(c)$$
> 
> form a (3, 3, 3)-dimensional polytope with known presentation.

The two questions initially posed above can be reduced to a restricted form of the equation appearing in the Theorem: for which $c$ does the equation

$$\mathrm{local} \cdot \mathit{CAN}(a \in \text{a short circuit}) \cdot \mathrm{local} \cdot \mathit{CAN}(b \in \text{new operations}) \cdot \mathrm{local} = \mathit{CAN}(c)$$

have a solution?
A direct consequence of the Theorem is that if $a$ and $b$ are constrained by linear inequalities (i.e., if they belong to polytopes of their own), then $c$ will be too.

The goal of `monodromy` is to automate the process of calculating the constraints on $c$.
To support this computation, we'll want:

+ **Storage:** `ConvexPolytope` / `Polytope` objects to house the data.
+ **Manipulation:** Intersection, union, containment, equality, projection, ... .
+ **Simplification:** To tame the overall time complexity of a long computation, we will want to be able to simplify the data emitted by the manipulation routines so that further manipulations are also efficient.


## `ConvexPolytope`s and `Polytope`s

`ConvexPolytopes` are described as intersections of constraint sets.
Here is an example `ConvexPolytope` object (meant to capture the gate "CZ", to be discussed in detail later)):

```python
CZ_convex_subpolytope = ConvexPolytope(
    inequalities=[],
    equalities=[
        [-1, 4, 0,  0],
        [-1, 0, 4,  0],
        [-1, 0, 0, -4],
    ],
    name='CZ'
)
```

Each line in the list of `equalities` corresponds to a linear equality constraint, where the first value is a constant term, the remaining values are the constant coefficients appearing in a linear combination, and the whole affine-linear combination is set equal to zero.
For instance, the entry `[-1, 4, 0, 0]` corresponds to the equation $$-1 + 4 \cdot x_1 + 0 \cdot x_2 + 0 \cdot x_3 = 0.$$
Similarly, each line in the list of `inequalities` corresponds to an affine-linear combination which is constrained to be nonnegative.
This particular example thus selects the singleton set $\{1/4, 1/4, -1/4\}$.

`ConvexPolytope`s support almost all of the operations we require, with one exception: the _union_ of two convex polytopes need no longer be convex.^[Think of two line segments, each individually a convex polytope, joining at a corner to form an "L"-shape, no longer a convex set.]
The polytope emitted in the Theorem need not be convex, so we introduce a general `Polytope` class which describes a union of the list of `ConvexPolytope`s which it retains.
The following example elevates the above `ConvexPolytope` to a `Polytope` object:

```python
CZ_polytope = Polytope(convex_subpolytopes=[CZ_convex_subpolytope])
```

`ConvexPolytope` and `Polytope` objects enjoy the following properties:

+ `convex_subpolytopes` (`Polytope` only): The list of `ConvexPolytope`s which make up the `Polytope`.
+ `equalities` (`ConvexPolytope` only): The list of exact linear constaints.  These are required to have integer coefficients.
+ `inequalities` (`ConvexPolytope` only): The list of weak linear constrains.  These are required to have integer coefficients.
+ `triangulation`: (`ConvexPolytope` only): A presentation of a `ConvexPolytope` as a non-overlapping union of "simplices" (i.e., general-dimension tetrahedra).  Returns a list of simplices, each specified as a list of indices into the `vertices` property describing the corners of the simplex.
+ `vertices` (`ConvexPolytope` only): The extremal points of the convex polytope, i.e., a minimal set of points for which all other points in the body appear as convex-linear combinations.
+ `volume`: The measure of the polytope in the appropriate dimension

The following snippet shows the (derived) attributes for `CZ_convex_subpolytope`:

```
# CZ: 
   -1 +     4 x1 +     0 x2 +     0 x3 == 0
   -1 +     0 x1 +     4 x2 +     0 x3 == 0
   -1 +     0 x1 +     0 x2 +    -4 x3 == 0

Attributes:
    equalities: [[-1, 4, 0, 0], [-1, 0, 4, 0], [-1, 0, 0, -4]]
    inequalities: []
    name: CZ
    triangulation: [[0]]
    vertices: [[Fraction(1, 4), Fraction(1, 4), Fraction(-1, 4)]]
    volume: PolytopeVolume(dimension=0, volume=Fraction(1, 1))
```

+ `contains`: `P.contains(Q)` checks whether `Q` is a subset of `P`.
+ `convex_hull` (class method): Constructs a `ConvexPolytope` from a set of extremal vertices.
+ `has_element`: `P.has_element(Q)` checks whether `Q` belongs to `P`.
+ `inflate`: Deserializes data previously stored using `dataclasses.asdict`.
+ `intersect`: `P.intersect(Q)` returns a (`Convex`)`Polytope` expressing $P \cap Q$.
+ `reduce`: Returns `ConvexPolytope`s specified by a minimal set of in/equalities and `Polytope`s specified with irredundant summands.
+ `union`: `union(Q)` returns a `Polytope` expressing $P \cup Q$.


## Computational `Backend`s

The computational engine which powers these basic operations is governed by the interface stored at `monodromy.backend.backend_abc.Backend`.
Any Backend implementation must implement the following methods:

+ `volume`: Calculates the volume of a `ConvexPolytope`.

This is by far the most method to implement and is used ubiquitously.
For example, testing `P.contains(Q)` for `P` a `Polytope` and `Q` a `ConvexPolytope` is accomplished by testing $\operatorname{vol}(Q) = \operatorname{vol}(P \cap Q)$.^[A useful example for why this is necessary is to check $\Delta^2 \not\subseteq \partial \Delta^2$.]

+ `reduce`: Produces a minimal set of inequalities expressing a `ConvexPolytope`.

This is the second most important method to implement.
This does not affect correctness, but it is also used ubiquitously to tame calculation complexity.

The remaining methods are used sparingly, and much of the package would work even if they were unimplemented:

+ `convex_hull`: Produces a minimal `ConvexPolytope` from a set of vertices.
+ `triangulation`: Calculates a triangulation of the input `ConvexPolytope`.
+ `vertices`: Calculates the vertices of the `ConvexPolytope`.

### `lrs`

This entire family of methods are actually hard to find in computational geometry packages.
The best software option we've found to date is [`lrs`](http://cgm.cs.mcgill.ca/~avis/C/lrslib/USERGUIDE.html), which handles all of these exact calculations for _convex_ polytopes.
`monodromy` interacts with `lrs` through the [`lrs` backend](https://github.ibm.com/IBM-Q-Software/monodromy/blob/main/monodromy/backend/lrs.py).

### Non-convex computation

The [method implementations for `ConvexPolytope`](https://github.ibm.com/IBM-Q-Software/monodromy/blob/main/monodromy/polytopes.py#L53) are almost all thin wrappers over the `Backend`.
For nonconvex unions appearing in general `Polytope`s, `monodromy` supplies [more elaborate wrappers](https://github.ibm.com/IBM-Q-Software/monodromy/blob/main/monodromy/polytopes.py#L152).

For a simple example, the intersection of two `Polytope`s, written as the union of convex summands, is computed according to the formula $$\left( \bigcup_i P_i \right) \cap \left( \bigcup_j Q_j \right) = \bigcup_{i, j} (P_i \cap Q_j),$$ which reduces the computation to the convex case.

For a more complicated example, the volume of a non-convex body can be computed from the volumes of its convex pieces using [inclusion-exclusion](https://en.wikipedia.org/wiki/Inclusion%E2%80%93exclusion_principle).
However, if the nonconvex body decomposes into $k$ convex pieces, a naive application of inclusion-exclusion will require $(2^k - 1)$ volume computations of convex pieces (or their various intersections).
`monodromy` computes volume using an [optimized variant of inclusion-exclusion](https://github.ibm.com/IBM-Q-Software/monodromy/blob/main/monodromy/volume.py#L55), which uses vanishing volume and containment to skip parts of the computational lattice.
In practice, this greatly accelerates computation; see Appendix B of the paper for details.

> **NOTE:** This is not guaranteed to accelerate computation.
> There are configurations of convex polytopes which still require $2^k - 1$ convex volume computations when applying inclusion-exclusion.
> There exist volume algorithms, particularly in the "sweep" family, whose complexity follows $O(\binom{k}{d})$, where $d$ is the ambient dimension of the space.
> We have not implemented such an algorithm, though maintainers may want to look into this in the future.

### Complex computations

Granting these primitives, one can then write the relevant high-level manipulation routines for `Polytope`s.

#### [Cylinderization](https://github.ibm.com/IBM-Q-Software/monodromy/blob/main/monodromy/elimination.py#L12)

Cylinderization extrudes a low-dimensional polytope into high-dimensional space.
For example, a family of constraints on the inputs to CAN, a 3-dimensional object, need to be extruded in order to be re-used as constraints on $a$–coordinates in the 9-dimensional monodromy polytope.

#### [Projection](https://github.ibm.com/IBM-Q-Software/monodromy/blob/main/monodromy/elimination.py#L44)

Projection calculates the shadow of a high-dimensional polytope onto a low-dimensional plane.
There is a standard algorithm for this, [Fourier–Motzkin elimination](https://en.wikipedia.org/wiki/Fourier%E2%80%93Motzkin_elimination), and we implement its most basic variant.

> **NOTE:** This operation is fairly expensive.
> The algorithm eliminates one coordinate at a time, in the worst case squaring the number of inequalities it emits as output, so that eliminating $k$ coordinates carries doubly-exponential complexity.
> Users are advised to call `.reduce()` after each elimination step.

> **NOTE:** `lrs` supposedly implements Fourier–Motzkin elimination, but it seems to be broken.
> Maintainers are advised not to hook into it.


## Quantum information

The above subroutines suffice to make the computations initially advertised.
`monodromy` bundles the basic step common to all such computations as `deduce_qlr_consequences`, which takes as input some constraint sets on $a$–, $b$–, and $c$–coordinates, then deduces the total effect on any one of those coordinates.

For example, the following snippet computes the possible $c$–coordinates when $CAN(a)$ is fixed to have the type of $CZ$ and $CAN(b)$ is fixed to have the type of $ISWAP$:

```python
from monodromy.coverage import deduce_qlr_consequences
from monodromy.static.examples import everything_polytope, exactly

ISWAP_polytope = exactly(Fraction(1, 2), Fraction(0), Fraction(0), name="ISWAP")

CZ_ISWAP_polytope = deduce_qlr_consequences(
    target="c",
    a_polytope=CZ_polytope,
    b_polytope=ISWAP_polytope,
    c_polytope=everything_polytope
)
```

The contents of `CZ_ISWAP_polytope` turns out to be:

```
[
    0 +     0 x1 +     1 x2 +     1 x3 >= 0
    0 +     1 x1 +    -1 x2 +     0 x3 >= 0
    0 +     0 x1 +     1 x2 +    -1 x3 >= 0
   -1 +     2 x1 +     2 x2 +     0 x3 >= 0
    1 +    -2 x1 +    -2 x2 +     0 x3 >= 0

]
```

### Standard coordinate systems

This is somewhat illegible, essentially because the coordinate system in which the Theorem is most easily expressed is not the one most commonly used by quantum information theorists.

+ "Monodromy coordinates" are essentially the logarithms of the eigenvalues of the canonical operator.
+ "Canonical coordinates" are essentially the $x$, $y$, $z$ appearing in the definition of $CAN$.

Some further "normalization" conditions are used to put these values into standard ranges.

> **Example:**
> In monodromy coordinates, the canonical gate equivalent to CZ lies at $(1/4, 1/4, -1/4)$.
> In canonical coordinates, it lies at $(1/2, 0, 0)$.

The subpackage `monodromy.coordinates` has routines which convert between these two settings.
Applying such a routine to the derived polytope above yields:

```python
from monodromy.coordinates import monodromy_to_positive_canonical_polytope

monodromy_to_positive_canonical_polytope(CZ_ISWAP_polytope)
```

```
[
# anonymous_convex_polytope_2: 
    0 +     0 x1 +     0 x2 +     2 x3 >= 0
    0 +     0 x1 +     2 x2 +    -2 x3 >= 0
    0 +     2 x1 +    -2 x2 +     0 x3 >= 0
   -2 +     4 x1 +     0 x2 +     0 x3 >= 0
    2 +    -4 x1 +     0 x2 +     0 x3 >= 0
]
```

which has vertices at $$(1/2, 1/2, 1/2), \quad (1/2, 0, 0), \quad (1/2, 1/2, 0),$$ corresponding respectively to $SWAP$, $CZ$, and $ISWAP$.

### Circuits of higher depth

As an exercise in stringing several invocations of `deduce_qlr_consequences` in a row, we can also check that three $CZ$ gates are sufficient to simulate any two-qubit operation:

```python
from monodromy.coordinates import monodromy_alcove_c2

CZCZ_polytope = deduce_qlr_consequences(
    target="c",
    a_polytope=CZ_polytope,
    b_polytope=CZ_polytope,
    c_polytope=everything_polytope
)

CZCZCZ_polytope = deduce_qlr_consequences(
    target="c",
    a_polytope=CZCZ_polytope,
    b_polytope=CZ_polytope,
    c_polytope=everything_polytope
)

CZCZCZ_polytope.contains(monodromy_alcove_c2)
```

yields `True`.

As an elaboration of this idea, we can consume any sequence of Qiskit gates and calculate the associated circuit polytope.  (This is a modification of a stock script found in `scripts/single_circuit.py`.)

```python
import qiskit
from qiskit.circuit.library import RZXGate
import numpy as np

from monodromy.coordinates import positive_canonical_alcove_c2, unitary_to_monodromy_coordinate
from monodromy.static.examples import identity_polytope


gates = [RZXGate(np.pi/12), RZXGate(np.pi/8), RZXGate(np.pi/10), RZXGate(np.pi/6)]

circuit_polytope = identity_polytope  # start with the identity gate

# for each gate in the sequence...
for gate in gates:
    # ... calculate its associated b–coordinate constraint ...
    b_polytope = exactly(
        *(Fraction(x).limit_denominator(10_000)
          for x in unitary_to_monodromy_coordinate(gate.to_matrix())[:-1])
    )
    # ... and apply the constraints produced thus far to the a–coordinates.
    circuit_polytope = deduce_qlr_consequences(
        target="c",
        a_polytope=circuit_polytope,
        b_polytope=b_polytope,
        c_polytope=everything_polytope
    )
```

After the loop finishes, this gives:

```
[
    2 +    -2 x1 +    -2 x2 +     0 x3 >= 0
    0 +     0 x1 +     0 x2 +     2 x3 >= 0
    0 +     0 x1 +     2 x2 +    -2 x3 >= 0
  -42 +    80 x1 +   -80 x2 +   -80 x3 >= 0
  274 +  -240 x1 +  -240 x2 +  -240 x3 >= 0
,
    0 +     0 x1 +     0 x2 +     2 x3 >= 0
    0 +     0 x1 +     2 x2 +    -2 x3 >= 0
    0 +     2 x1 +    -2 x2 +     0 x3 >= 0
   34 +   240 x1 +  -240 x2 +  -240 x3 >= 0
   38 +   -80 x1 +   -80 x2 +   -80 x3 >= 0
]
```

One can confirm that this matches the "Main Global Theorem" from the paper — but we're also free to plug in any gate we like into `gates`, provided it has a `.to_matrix` method.


### Haar volume

The `.volume` method on `Convex`/`Polytope` objects computes its _Euclidean_ volume.
For example, continuing the example from the previous subsection and writing

```python
p = monodromy_to_positive_canonical_polytope(circuit_polytope)
q = positive_canonical_alcove_c2
```

gives

```
p.volume  # PolytopeVolume(dimension=3, volume=Fraction(6857, 1152000))
q.volume  # PolytopeVolume(dimension=3, volume=Fraction(1, 24))
float(p.volume.volume / q.volume.volume)  # 0.14285416666666666
```

In quantum informational contexts, however, it is more common to care about such alternative volume measures as the _Haar volume_, which reflects the statistical distribution of $CAN$ gates appearing as the nonlocal parts of an unbiased random distribution of $SU(4)$ operators.
The calculation of the Haar volume can be expressed as a certain trigonometric integral, and for a polytope-shaped domain the problem can be further reduced to integrating over a tetrahedron.
`monodromy` has routines for performing such calculations exactly and efficiently, as in the following example:

```python
from monodromy.haar import haar_volume

haar_volume(p)  # 0.027227084187167933
haar_volume(q)  # 1.0
```

One can see from this example that this is a rather different volume measure: despite $P$ occupying ~12% of the Euclidean volume of the total space $Q$ of all $CAN$ operations, it is hit only ~2% of the time by operators drawn randomly from $SU(4)$.

The engine which computes these Haar volumes is located in two places:

* [haar.py](https://github.ibm.com/IBM-Q-Software/monodromy/blob/main/monodromy/haar.py#L125):
  The volume calculation re-uses the same inclusion-exclusion tricks as above, uses `.triangulation` to reduce to the tetrahedral case, and then sets up the relevant trigonometric integrals.
* [polynomials.py](https://github.ibm.com/IBM-Q-Software/monodromy/blob/main/monodromy/polynomials.py):
  These are symbolic algebra classes which embody (trigonometric) polynomials and include the basic functionality needed to integrate and evaluate them.

> **NOTE:**
> It would be preferable to replace `polynomials.py` with `sympy`, but I have been unable to figure out how to make it fast.


### Coverage set calculation

The examples immediately preceding this illustrated how to use the package to solve the first of our two original questions: how to calculate the "circuit polytope" associated to a given sequence of two-qubit interactions.
`monodromy` also provides a wrapper for solving the second question: how to calculate the best circuit embodiment for a target two-qubit operation using a sequence of gates drawn from a given gateset.
The following example is adpated from `scripts/demo.py`:

```python
from monodromy.coverage import *
from monodromy.static.examples import *

scale_factor = (64 * 90) / (10000 * 100),
offset = 909 / (10000 * 100) + 1 / 1000,

# build the native gate set
operations = [
    CircuitPolytope(
        operations=[f"rzx(pi/2 * {strength})"],
        cost=strength * scale_factor + offset,
        convex_subpolytopes=exactly(
            strength / 4, strength / 4, -strength / 4,
        ).convex_subpolytopes
    )
    for strength in [Fraction(1), Fraction(1, 2), Fraction(1, 3)]
]

# calculate the set of covering polytopes
coverage_set = build_coverage_set(operations, chatty=True)

# print it out for user inspection
print_coverage_set(coverage_set)
```

`build_coverage_set` explores the possible sequences of native gates, arranged in order of increasing cost, and calculates the circuit polytope associated to each such sequence.
It also remembers the regions swept out by previously examined circuit polytopes, so that if a new sequence is wholly contained within the already-explored region, it can be discarded.
Similarly, if a new sequence is a descendant of one which was previously discarded, it too can be skipped.
Between these two rules, the space of sequences to explore can become quite lean.
When the entire space has been swept out, the loop terminates and this minimal set of circuit types involved in the sweep are emitted (together with their costs and associated circuit polytopes).

After quite a lot of work, the example above yields:

```
Percent volume of A_C2	 | Cost	 | Sequence name
  0.00% =    0/1    	 | 0.00	 | 
  0.00% =    0/1    	 | 0.00	 | rzx(pi/2 * 1/3)
  0.00% =    0/1    	 | 0.00	 | rzx(pi/2 * 1/2)
  0.00% =    0/1    	 | 0.01	 | rzx(pi/2 * 1/3).rzx(pi/2 * 1/3)
  0.00% =    0/1    	 | 0.01	 | rzx(pi/2 * 1)
  0.00% =    0/1    	 | 0.01	 | rzx(pi/2 * 1/2).rzx(pi/2 * 1/3)
  0.00% =    0/1    	 | 0.01	 | rzx(pi/2 * 1/2).rzx(pi/2 * 1/2)
 16.67% =    1/6    	 | 0.01	 | rzx(pi/2 * 1/3).rzx(pi/2 * 1/3).rzx(pi/2 * 1/3)
  0.00% =    0/1    	 | 0.01	 | rzx(pi/2 * 1/3).rzx(pi/2 * 1)
 25.00% =    1/4    	 | 0.01	 | rzx(pi/2 * 1/2).rzx(pi/2 * 1/3).rzx(pi/2 * 1/3)
  0.00% =    0/1    	 | 0.01	 | rzx(pi/2 * 1).rzx(pi/2 * 1/2)
 37.04% =   10/27   	 | 0.01	 | rzx(pi/2 * 1/2).rzx(pi/2 * 1/2).rzx(pi/2 * 1/3)
 50.00% =    1/2    	 | 0.01	 | rzx(pi/2 * 1/2).rzx(pi/2 * 1/2).rzx(pi/2 * 1/2)
 14.81% =    4/27   	 | 0.02	 | rzx(pi/2 * 1/3).rzx(pi/2 * 1/3).rzx(pi/2 * 1)
  0.00% =    0/1    	 | 0.02	 | rzx(pi/2 * 1).rzx(pi/2 * 1)
 28.70% =   31/108  	 | 0.02	 | rzx(pi/2 * 1).rzx(pi/2 * 1/2).rzx(pi/2 * 1/3)
 62.35% =  101/162  	 | 0.02	 | rzx(pi/2 * 1/2).rzx(pi/2 * 1/2).rzx(pi/2 * 1/3).rzx(pi/2 * 1/3)
 50.00% =    1/2    	 | 0.02	 | rzx(pi/2 * 1).rzx(pi/2 * 1/2).rzx(pi/2 * 1/2)
 73.77% =  239/324  	 | 0.02	 | rzx(pi/2 * 1/2).rzx(pi/2 * 1/2).rzx(pi/2 * 1/3).rzx(pi/2 * 1/2)
 83.33% =    5/6    	 | 0.02	 | rzx(pi/2 * 1/2).rzx(pi/2 * 1/2).rzx(pi/2 * 1/2).rzx(pi/2 * 1/2)
 70.37% =   19/27   	 | 0.02	 | rzx(pi/2 * 1).rzx(pi/2 * 1).rzx(pi/2 * 1/3)
 71.06% =  307/432  	 | 0.02	 | rzx(pi/2 * 1).rzx(pi/2 * 1/2).rzx(pi/2 * 1/3).rzx(pi/2 * 1/3)
 87.50% =    7/8    	 | 0.02	 | rzx(pi/2 * 1).rzx(pi/2 * 1/2).rzx(pi/2 * 1)
 85.19% =   23/27   	 | 0.02	 | rzx(pi/2 * 1/2).rzx(pi/2 * 1/2).rzx(pi/2 * 1/3).rzx(pi/2 * 1)
 90.35% = 1171/1296 	 | 0.02	 | rzx(pi/2 * 1/2).rzx(pi/2 * 1/2).rzx(pi/2 * 1/3).rzx(pi/2 * 1/3).rzx(pi/2 * 1/2)
 93.75% =   15/16   	 | 0.02	 | rzx(pi/2 * 1).rzx(pi/2 * 1/2).rzx(pi/2 * 1/2).rzx(pi/2 * 1/2)
 95.06% =   77/81   	 | 0.02	 | rzx(pi/2 * 1/2).rzx(pi/2 * 1/2).rzx(pi/2 * 1/2).rzx(pi/2 * 1/2).rzx(pi/2 * 1/3)
 96.30% =   26/27   	 | 0.02	 | rzx(pi/2 * 1).rzx(pi/2 * 1).rzx(pi/2 * 1/3).rzx(pi/2 * 1/3)
100.00% =    1/1    	 | 0.02	 | rzx(pi/2 * 1).rzx(pi/2 * 1).rzx(pi/2 * 1)
```

These percentages reported are of Euclidean volumes, and as before it is often more useful to compute Haar volumes instead.
The following wrapper function computes the expected cost, in the Haar-random sense, of running a circuit:

```python
# calculate Haar-measure statistics
from monodromy.haar import cost_statistics

cost_statistics(
    coverage_set,
    offset=909 / (10000 * 100) + 1 / 1000,
    scale_factor=(64 * 90) / (10000 * 100),
    chatty=True,
)
```

```
{'average_cost': 0.015448974523296053,
 'average_overshot': 0.0010819745232960518,
 'sigma_cost': 0.00224229672978459,
 'sigma_overshot': 0.0008532811346104187}
```

`average_cost` is the promised Haar-expected cost, and `sigma_cost` is the standard deviation of the random variable describing Haar-random cost.
The other values reported compare against a model where $CAN(x, y, z)$ costs^[Up to reflection in $x$.] $m \cdot (x + y + z) + b$ for some parameters $m$, the `scale_factor`, and $b$, the `offset`.
Specifically, `overshot` is defined as the circuit cost less this auxiliary cost model; `average_overshot` reports its Haar-random expectation; and `sigma_overshot` reports its Haar-random standard deviation.



## Symbolic manipulation

We also used this software package to perform the _symbolic_ manipulations appearing in the $XX$ synthesis paper, which describe the circuit polytope associated to a sequence of $XX$–type interactions of arbitrary length and value.
The main idea is to leave the interaction strength as a variable and _not_ project it away, so that its effect on the shape of the circuit polytope is retained and made visible.

* `scripts/xx_sequence.py`:
  Computes the circuit polytope for a sequence of XX interactions of length $n$, one at a time, for $n$ in $(1, 2, 3, \ldots)$.
* [`check_main_xx_theorem`](https://github.ibm.com/IBM-Q-Software/monodromy/blob/main/monodromy/static/interference.py#L317):
  Checks that appending a new XX interaction to an XX circuit produces a polytope of the expected form (as in a proof by induction on depth).
* [`regenerate_xx_solution_polytopes`](https://github.ibm.com/IBM-Q-Software/monodromy/blob/main/monodromy/static/interference.py#L273):
  Checks that our restricted choice of local gates and their associated "interference inequalities" add no extra constraints from the perspective of circuit polytopes.
  Also regenerates the polytope $P$ referenced in the proof and in the subsequent synthesis method.
  
These scripts are very time-expensive:

```
Working on an interaction sequence of length 5...
Working on the reduction 1/3... done.  Took 829.379017751 seconds.
Working on the reduction 2/3... done.  Took 444.05925238299994 seconds.
Working on the reduction 3/3... done.  Took 35.14525969400006 seconds.
```

```
Checking main global theorem... took 9.341573042116668 minutes.
Checking main local theorem...  took 38.09305827083333 minutes.
```

but this is still faster than proving these theorems by hand!


## `/scripts/`

Scripts we've seen already:

* `single_circuit.py`: Generates the circuit polytope associated to any sequence of entangling Qiskit-style gates.  _(This is covered in this notebook.)_
* `demo.py`: Demonstrates the coverage set calculator.
* `more_xx_interactions.py`: Uses the `Polytope` calculator to extract arbitrarily deep circuit polytopes for $XX$–type circuits.
* `proof.py`: Uses the `Polytope` calculator to verify the main local and global theorems from the published paper.

Some other scripts:

* `gateset.py`: Scans the optimization landscape to produce an infidelity-minimizing gateset against a given error model and for an _exact_ synthesis target.  This uses the Haar volume calculation feature.
* `approx_gateset.py`: Scans the optimization landscape to produce an infidelity-minimizing gateset against a given error model and for an _exact_ synthesis target.  This approximates the Haar integral by Monte Carlo sampling.
* `nuop.py`: Generates a `.dat` with timing and depth statistics for monodromy-style circuit generation (into a single entangling gate) vs Prakash-style numerical search.
* `qft.py`: Generates statistics for QFT circuits of increasing size, synthesized with Vatan–Williams and with our method.
