## CONVENTIONS FROM PREVIOUS WORK

The code in `legacy/src/qecnoise/` is research code organized around the
mathematical steps in the accompanying thesis. New code that continues this
work should follow these conventions unless there is a reason to change them.

### Structure and style

- Keep related mathematical operations as small, top-level functions. The
  modules are organized in stages with prominent section dividers, and the
  public functions are imported directly by downstream modules.
- Use descriptive `snake_case` names for functions and variables. Short
  uppercase names are reserved for mathematical matrices and arrays such as
  `D`, `G`, `E`, `S`, and `B`; indices and dimensions use names such as `n`,
  `k`, `i`, and `q`.
- Put a module docstring first. It states the mathematical purpose, expected
  inputs and outputs, relevant thesis sections, and how to run the module as a
  script. Add function docstrings that define the mathematical meaning,
  parameters, return values, and any important limitations. Use nearby comments
  to explain algorithm stages and non-obvious implementation choices.
- Use f-strings for labels and status output. Guard demonstration code with
	`if __name__ == "__main__":`; keep reusable calculations in functions and
	demonstrations in that guarded block.

### Mathematical data and computation

- Represent an $n$-qubit Pauli operator as a pair `[x, z]` of length-$n$
  binary vectors, with `X = (1, 0)`, `Y = (1, 1)`, and `Z = (0, 1)`. A list of
  operators therefore has shape `(count, 2, n)`. Document array shapes and
  ordering where they are part of a function's interface.
- Use NumPy arrays and vectorized operations for numerical linear algebra.
  State explicitly when arithmetic is modulo two, use `astype(bool)` for
  support masks, and use `dtype=float` where real-valued projections or
  optimizers require it. Preserve stable, documented ordering for generated
  error families and their labels.
- Separate exact claims from numerical search. Numerical rank, projections,
  and geometric searches use explicit tolerances; when a verdict must be
  rigorous, use integer/rational arithmetic and verify the certificate rather
  than treating a floating-point result as proof.
- For large matrices or families, provide block/chunk processing where
  practical, and explain its memory motivation. Avoid materializing large
  intermediate matrices when a streaming calculation suffices.
- Validate public input shapes and raise informative `ValueError`s for invalid
  inputs. Use `RuntimeError` when an algorithm's expected invariant or
  completion condition fails. Distinguish a capped/inconclusive search from a
  mathematical negative result.

### Module responsibilities

- `Dmatrixsinglequbit.py` owns Pauli/error-family construction, symplectic
  commutators, measured-set construction, and operator labels.
- `gordanLP.py` builds on those primitives for shadows, certificates, and
  feasibility/search routines. It uses SciPy's `linprog` for the linear
  programme, with NumPy for matrix operations and `Fraction`/integer
  calculations for exact certification.
- `girth.py` composes the preceding APIs for exhaustive subset and girth
  searches. Keep shared mathematics in its owning module rather than copying
  the implementation into a downstream analysis.

### Dependencies and running

- The legacy implementation uses Python's standard library, NumPy, and SciPy.
  Import shared functions from the `qecnoise` package rather than duplicating
  them.
- Each analysis module can be run directly for a small, documented example;
  output is plain, human-readable console text with labels and summary counts.
  Computationally expensive searches should make their cap and the meaning of
  a capped result clear.

