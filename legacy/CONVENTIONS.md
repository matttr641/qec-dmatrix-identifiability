# Legacy code: conventions and interfaces

- **Source:** `qecnoise@b70f1148d6dbf609e19fa003dcfa6ce081b3c375`
- **Frozen.** Never edit anything in `legacy/src/`. New code goes in `src/qecdm/`.
- **What it does:** builds the design matrix D' for a stabilizer code (Algorithm 1, thesis Ch. 4), then decides whether errors from a declared family can hide from it (Ch. 6).
- **Dependencies:** NumPy, SciPy, standard library.
- **Imports:** `from qecnoise.<module> import ...`, with `legacy/src` on the Python path.

## Data conventions

- **Pauli operator on n qubits:** a pair `[x, z]` of length-n 0/1 vectors.
    - X = (1, 0), Y = (1, 1), Z = (0, 1).
    - Example: Z0Z1 on 7 qubits is `[[0,0,0,0,0,0,0], [1,1,0,0,0,0,0]]`.
- **List of operators:** an array of shape `(count, 2, n)`.
- **Commutation:** `commutators(A, B)` gives a `(len(A), len(B))` 0/1 matrix, where 1 means anticommute. Arithmetic is mod 2.
- **Single-qubit errors:** ordered X0, Y0, Z0, X1, Y1, Z1, ... (3n in total).
- **D':**
    - Rows are the measured stabilizers (the "measured set").
    - Columns are syndrome classes of single-qubit errors, one representative each. Undetectable errors are dropped.
    - An entry is 1 when that row anticommutes with that column's error.
- **Family:** the candidate multi-qubit errors being tested, in the same `(count, 2, n)` format.
- **Absorbed:** a family member with zero shadow. It hides on its own.
- **Resolved:** every family member that is not absorbed.
- **Exact vs float:**
    - Rank, SVD and the LP (`scipy.optimize.linprog`) are floating point, with `tol=1e-9`.
    - `certify_witness` gives an exact verdict, using integers and `Fraction`.

## Interfaces

### `Dmatrixsinglequbit.py`: inputs and D'

**Building operators**
- `paulis(n, letters, qubits)` returns one operator `[x, z]`. For example, `paulis(7, "XZ", (0, 3))` is X0 Z3.
- `flips(n, *qubits)` returns X on each listed qubit.
- `pair_family(n)` returns `(errors, labels)`: all C(n,2) XX errors, labelled `(a, b)`.
- `two_qubit_family(n)` returns `(errors, labels)`: all 9·C(n,2) two-qubit errors, labelled like `"X0Y3"`.
- `bit_flip_chain(n=7)` returns the n-1 generators Z_i Z_(i+1).

**Main function:** `design_matrix(n, k, stabilizers, max_order=None, extra_edges=())` returns `(D, merged, measured)`.

- **Inputs**
    - `stabilizers`: k generators, shape `(k, 2, n)`. A wrong shape raises `ValueError`.
    - Default behaviour: adds products of connected generator subsets, smallest subsets first, until rank(D') = |C*|. Raises `RuntimeError` if it never gets there.
    - `max_order`: walk to exactly this subset size instead. D' may then be rank-deficient. On the 7-qubit chain, `max_order=6` gives the 21-row complete weight-2 set.
    - `extra_edges`: extra generator pairs `(i, j)` to join in the correlation graph (the §10.2 repair).
- **Outputs**
    - `D`: 0/1 integers, shape `(rows, |C*|)`.
    - `merged`: one tuple per column, naming the single-qubit errors it stands for, e.g. `("X0", "Y0")`.
    - `measured`: the measured stabilizers, shape `(rows, 2, n)`. Row i of `measured` is row i of `D`.

**Helpers**
- `commutators(A, B)`: see Data conventions above.
- `stabilizer_label(op)` turns an operator into a string such as `"Z0Z1"`.
- `edge_rows(operators)` returns `{support tuple: row index}`.

### `gordanLP.py`: can anything in the family hide? (§6.1 to 6.3)

**Main functions**
- `gordan(D, measured, family, tol=1e-9)` returns `(certified, absorbed, resolved, witness)`.
    - `certified`: True when the LP is feasible, meaning no channel built from resolved family members is invisible.
    - `absorbed`, `resolved`: lists of indices into `family`.
    - `witness`: the certifying vector λ, or `None` when not certified.
    - If every member is absorbed, it returns `(False, absorbed, [], None)`.
- `certify_witness(D, measured, family, witness, block=2000)` returns `(proved, worst, checked)`.
    - Checks a witness exactly, with no floating point in the verdict.
    - `worst` is a `Fraction`.
- `shadows(D, measured, family, block=None, dtype=float)` returns β^P = (I - Π_D) s^P, one column per family member, shape `(rows, len(family))`.
- `straddles(measured, family)` returns s^P, one column per family member.

**For large codes** (blocked, so they never hold the whole matrix)
- `image_basis`, `shadow_norms`, `verify_witness`, `summed_shadow_certificate`.
- `witness_search`: when it fails, that means no witness was found. It is not a verdict that none exists.

### `girth.py`: smallest set of errors that can hide together (§6.4)

**Main function:** `physical_girth(D, measured, family, max_size=6)` returns `(absorbed, spark, m_star, members)`.

- `m_star = np.inf`: certified by the LP; nothing hides.
- `m_star` an integer: the size of the smallest positive circuit. `members` gives its family indices.
- `m_star = None`: the search hit `max_size` first, so m* > `max_size`.
- `spark`: the smallest dependent subset size. `None` when certified.

**Helper:** `spark_and_girth(B, max_size=6, tol=1e-9)` returns `(spark, size, members)` on a shadow matrix B.
- Exhaustive, so the cost grows exponentially with `max_size`.
- The mixed pairs-plus-triples family on the 21-row set is slow; keep it out of the default tests.

## Running the demos

From the repo root:

```bash
PYTHONPATH=legacy/src python -m qecnoise.Dmatrixsinglequbit   # D' for the 7-qubit chain
PYTHONPATH=legacy/src python -m qecnoise.gordanLP             # LP verdict, 11- and 21-row sets
PYTHONPATH=legacy/src python -m qecnoise.girth                # m* for both sets (slow at the end)
```

## Style for new code

- **Functions:** small and top-level.
- **Docstrings:** each module opens with one giving its purpose, inputs, outputs and thesis section.
- **Names:** uppercase for matrices (`D`, `G`, `E`, `S`, `B`); `snake_case` for everything else.
- **Errors:**
    - Bad input raises `ValueError`.
    - A broken internal invariant raises `RuntimeError`.
    - Always distinguish "search capped" from "answer is no".
- **Rigour:** rigorous claims use integers or `Fraction`. Floats are only used with an explicit tolerance.
