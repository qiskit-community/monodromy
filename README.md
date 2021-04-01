# `monodromy`

Computations in the monodromy polytope for quantum gate sets

## Overview

[Fixed-Depth Two-Qubit Circuits and the Monodromy Polytope](https://arxiv.org/abs/1904.10541) gave a prescription for a technique to determine whether a given two-qubit unitary can be written as a circuit with a prescribed sequence of two-qubit interactions and arbitrary single-qubit unitaries interleaved.
This python package is a computationally effective implementation of that technique.

## Installation

1. This package comes with a `requirements.txt` file.  Begin by installing those requirements, using `pip -r requirements.txt`.
2. Install [`lrs`](http://cgm.cs.mcgill.ca/~avis/C/lrs.html).  Typically, this means downloading the source, building it, and placing the generated executable somewhere in the search path for your python process.
3. *Optionally,* install [`lrcalc`](https://sites.math.rutgers.edu/~asbuch/lrcalc/).  Typically, this means downloading the source, building it, then building the Cython bindings, and installing the resulting package.  *This is not necessary: `lrcalc` is needed only for the curious user who wants to regenerate the contents of `qlr_table.py`.*

## Configuration

If you're unable to put `lrs` in your search path, you can specify its location via the environment variable `LRS_PATH`.

## Usage

Give it a whirl with the following:

```python
from monodromy.coverage import *
import monodromy.examples
import monodromy.render

operations = [
    GatePolytope(
        convex_subpolytopes=monodromy.examples.thirdCX_polytope.convex_subpolytopes,
        cost=Fraction(1, 3),
        operations=["thirdCX"],
    ),
    GatePolytope(
        convex_subpolytopes=monodromy.examples.sqrtCX_polytope.convex_subpolytopes,
        cost=Fraction(1, 2),
        operations=["sqrtCX"],
    ),
]

coverage_set = build_coverage_set(operations, chatty=True)
print_coverage_set(coverage_set)
print("====\nRender me in Mathematica:\n=====")
print(monodromy.render.polytopes_to_mathematica(coverage_set))
```

## Notes

We've designed the package around polytopes with rational coordinates.
In practice this suits the use cases of quantum computer scientists fine, but it is easy to imagine use cases outside of this.
`lrs` makes strong assumptions about the kind of arithmetic used, so if one were to want to process irrational polytopes, one would be obligated to move away from `lrs`.

`lrs` is licensed under GPLv2, which makes it insuitable for inclusion in some projects.
We may someday want to remove (or make optional) this dependency from this project.
To this end, we have set up a `backend` which describes a contained and minimal set of calls we make of any computational library.  
