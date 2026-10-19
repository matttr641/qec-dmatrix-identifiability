"""Builds the design matrix D' by Algorithm 1. Every other script imports it.

In   n qubits, and k stabilizer generators, each written as a pair of length-n
     symplectic vectors [x, z], meaning qubit q carries X**x[q] Z**z[q]. So
     Z0Z1 on seven qubits is [[0,0,0,0,0,0,0], [1,1,0,0,0,0,0]].
Out  D', the single-qubit errors merged into each of its columns, and the
     measured set M' the algorithm uses.

Run this file directly for the seven-qubit bit-flip code matrix of §4.3.1.
"""

from itertools import combinations, product

import numpy as np


# --------------------------------------------------------------------------- #
# Building the inputs
# --------------------------------------------------------------------------- #


# Symplectic [x, z] for a chosen letter on each chosen qubit.
def paulis(n, letters, qubits):
    """The error carrying the given Pauli letters on the given qubits.

    ``paulis(7, "XZ", (0, 3))`` is X_0 Z_3; every other qubit is identity.
    """
    x, z = [0] * n, [0] * n
    for letter, q in zip(letters, qubits):
        x[q], z[q] = {"X": (1, 0), "Y": (1, 1), "Z": (0, 1)}[letter.upper()]
    return [x, z]


# The all-X case of paulis: a flip on each named qubit.
def flips(n, *qubits):
    """The X-type error on the given qubits, in symplectic form."""
    return paulis(n, "X" * len(qubits), qubits)


# Every X-type pair error X_a X_b, with its (a, b) label.
def pair_family(n):
    """The C(n,2) two-qubit X errors, and the qubit pairs naming them."""
    labels = list(combinations(range(n), 2))
    return [flips(n, a, b) for a, b in labels], labels


# Every two-qubit error: nine Pauli combinations on each pair.
def two_qubit_family(n):
    """All 9 * C(n,2) two-qubit errors, and their labels like "X0Y3"."""
    family, labels = [], []
    for a, b in combinations(range(n), 2):
        for first, second in product("XYZ", repeat=2):
            family.append(paulis(n, first + second, (a, b)))
            labels.append(f"{first}{a}{second}{b}")
    return family, labels


# The chain's generators g_i = Z_i Z_{i+1}, ready to pass in.
def bit_flip_chain(n=7):
    """The generators g_i = Z_i Z_{i+1} of the n-qubit chain, in symplectic form."""
    generators = []
    for i in range(n - 1):
        z = [0] * n
        z[i] = z[i + 1] = 1
        generators.append([[0] * n, z])
    return generators


# --------------------------------------------------------------------------- #
# Algorithm 1
# --------------------------------------------------------------------------- #


# Every pairwise symplectic product at once, as one matrix.
def commutators(A, B):
    """<a, b> for every pair: 1 where they anticommute, 0 where they commute."""
    return (A[:, 0] @ B[:, 1].T + A[:, 1] @ B[:, 0].T) % 2


# Depth-first walk inside the subset: does it hang together?
def connected(adjacent, subset):
    """Is the subgraph induced on `subset` connected?"""
    seen, stack = {subset[0]}, [subset[0]]
    while stack:
        u = stack.pop()
        for w in subset:
            if w not in seen and adjacent[u][w]:
                seen.add(w)
                stack.append(w)
    return len(seen) == len(subset)


# Algorithm 1: syndrome classes, graph, walk by order, then reduce.
def design_matrix(n, k, stabilizers, max_order=None, extra_edges=()):
    """Return D', the errors merged into each column, and the measured set M'.

    See the module docstring for the input format. `max_order` walks the graph
    to exactly that subgraph order instead of stopping when the rank reaches
    |C*| (§4.4.3, the extended measured set); D' may then be rank-deficient.

    `extra_edges` adds pairs (i, j) to the correlation graph beyond the support
    overlaps, so the walk may form products the code's own geometry would not
    offer. That is the repair of §10.2: a covariance the declared model cannot
    explain names a generator pair, and joining them enlarges the measured set.
    """
    G = np.asarray(stabilizers)
    if G.shape != (k, 2, n):
        raise ValueError(
            f"expected {k} generators as [x, z] pairs of length {n}, i.e. shape "
            f"{(k, 2, n)}, but got {G.shape}"
        )

    # The 3n single-qubit errors, ordered X0, Y0, Z0, X1, ...
    E = np.zeros((3 * n, 2, n), dtype=int)
    for q in range(n):
        E[3 * q, 0, q] = 1  # X
        E[3 * q + 1, :, q] = 1  # Y
        E[3 * q + 2, 1, q] = 1  # Z
    labels = [f"{p}{q}" for q in range(n) for p in "XYZ"]

    # 1. |C*|: group the errors by syndrome, dropping the undetectable ones.
    classes = {}
    for j, syndrome in enumerate(commutators(G, E).T):
        if syndrome.any():
            classes.setdefault(syndrome.tobytes(), []).append(j)
    if not classes:
        raise ValueError("no single-qubit error is detectable: |C*| = 0")
    keep = [group[0] for group in classes.values()]  # one column per class

    # 2. Correlation graph: an edge where two generators act non-trivially on a
    #    common qubit, the channel regions being singletons here (Def. 4.4).
    support = (G[:, 0] | G[:, 1]).astype(bool)
    adjacent = support @ support.T
    for i, j in extra_edges:
        adjacent[i][j] = adjacent[j][i] = True

    # 3. Products of connected subgraphs, order by order, until rank >= |C*|.
    rows, order, rank = [], 0, 0
    while order < k and (order < max_order if max_order is not None
                         else rank < len(classes)):
        order += 1
        for subset in combinations(range(k), order):
            if connected(adjacent, subset):
                M = G[list(subset)].sum(axis=0) % 2
                if M.any():  # a dependent subset can multiply to the identity
                    rows.append(M)
        D = commutators(np.array(rows), E)
        rank = np.linalg.matrix_rank(D[:, keep])
    if max_order is None and rank < len(classes):
        raise RuntimeError(f"rank {rank} < |C*| = {len(classes)} at every order")

    # 4. Merge identical columns, drop the zero ones.
    merged = [tuple(labels[j] for j in group) for group in classes.values()]
    return D[:, keep], merged, np.array(rows)


# --------------------------------------------------------------------------- #
# Reading the output
# --------------------------------------------------------------------------- #


# Name an operator by the Paulis it carries: [[0..],[1,1,0..]] -> "Z0Z1".
def stabilizer_label(operator):
    """The Pauli letters of an operator, qubit by qubit, skipping identities."""
    letters = {(1, 0): "X", (1, 1): "Y", (0, 1): "Z"}
    x, z = np.asarray(operator)
    return "".join(f"{letters[(x[q], z[q])]}{q}" for q in np.flatnonzero(x | z))


# Look up which row carries the stabilizer on a given support.
def edge_rows(operators):
    """Map each operator's support to its row index: {(a, b): i, ...}."""
    return {
        tuple(np.flatnonzero(o[0] | o[1])): i
        for i, o in enumerate(np.asarray(operators))
    }


if __name__ == "__main__":
    D, merged, measured = design_matrix(7, 6, bit_flip_chain(7))
    print(f"|C*| = {D.shape[1]}, D' is {D.shape[0]} x {D.shape[1]}, "
          f"measured set M' has {len(measured)} stabilizers\n")

    # Each row of D' is one measured stabilizer, so name it in the margin:
    # that prints all three outputs at once.
    print(" " * 6 + " ".join("=".join(c).rjust(5) for c in merged))
    for operator, row in zip(measured, D):
        print(stabilizer_label(operator).rjust(5) + " "
              + " ".join(str(v).rjust(5) for v in row))
