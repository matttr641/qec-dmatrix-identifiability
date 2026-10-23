from itertools import combinations, product

import numpy as np

from qecdm.stabilizer import label_to_stab


def commutators(A, B):
    """np.ndarray (a, 2, n), (b, 2, n) -> np.ndarray (a, b): 1 where they anticommute.
    A and B are a list of Pauli operators.

    Entry (i, j) is the symplectic product of A[i] and B[j]: the number of
    qubits where their Paulis anticommute, mod 2.
    """
    A, B = np.asarray(A), np.asarray(B)
    return (A[:, 0] @ B[:, 1].T + A[:, 1] @ B[:, 0].T) % 2


def pauli_errors(n, weight):
    """int, int -> (np.ndarray (m, 2, n), list of str): every Pauli error of that weight.

    Ordered by qubits, then letters: weight 1 gives X0, Y0, Z0, X1, ...;
    weight 2 gives X0X1, X0Y1, ..., Z0Z1, X0X2, ...
    """
    labels = ["".join(f"{p}{q}" for p, q in zip(letters, qubits))
              for qubits in combinations(range(n), weight)
              for letters in product("XYZ", repeat=weight)]
    return np.array([label_to_stab(label, n) for label in labels]), labels


def correlation_graph(stabilizers):
    """np.ndarray (k, 2, n) -> np.ndarray (k, k) of bools: which stabilizers share a qubit.

    Entry (i, j) is True when stabilizers i and j both act non-trivially on at
    least one common qubit, whatever the Paulis there (Z0 and X0 are joined).
    """
    G = np.asarray(stabilizers)
    support = (G[:, 0] | G[:, 1]).astype(bool)  # True where a stabilizer isn't identity
    return support @ support.T                  # True if any qubit is in both supports


def connected(graph, subset):
    """np.ndarray (k, k), tuple -> bool: do the subset's nodes form one connected piece?

    Starts at the first node and walks along edges to nodes in the subset; the
    subset is connected if the walk reaches all of them and outputs True.
    """
    reached = {subset[0]}
    to_visit = [subset[0]]
    while to_visit:
        node = to_visit.pop()
        for other in subset:
            if other not in reached and graph[node, other]:
                reached.add(other)
                to_visit.append(other)
    return len(reached) == len(subset)


def stab_products(stabilizers, order, graph=None):
    """np.ndarray (k, 2, n), int -> (np.ndarray (m, 2, n), list): products, subsets.

    Multiplies together every connected set of `order` stabilizers, and returns
    the products with the stabilizer indices that made each one, e.g. (0, 3).
    Products that come out as the identity are dropped. `graph` defaults to
    the correlation graph; pass your own to add or remove edges.
    """
    G = np.asarray(stabilizers)
    if graph is None:
        graph = correlation_graph(G)

    products, subsets = [], []
    for subset in combinations(range(len(G)), order):
        if not connected(graph, subset):
            continue
        # Multiplying Paulis is adding their [x, z] bits mod 2 (signs dropped).
        product = G[list(subset)].sum(axis=0) % 2
        if product.any():
            products.append(product)
            subsets.append(subset)
    return np.array(products, dtype=int).reshape(-1, *G.shape[1:]), subsets


def design_matrix(stabilizers, weights=(1,), max_order=None):
    """np.ndarray (k, 2, n) -> (D', merged, measured, subsets).

    Builds D' by Algorithm 1. Rows are products of connected sets of the
    generators, walked order by order; columns are the syndrome classes of the
    Pauli errors of each weight in `weights` (default single-qubit only).

    Stops when rank(D') reaches |C*|, the number of classes, or after exactly
    `max_order` orders if given (D' may then be rank-deficient).

    Returns D' (rows, |C*|), the errors merged into each column, the measured
    stabilizers (rows, 2, n), and which generators made each row, e.g. (0, 3).
    Pass an independent generating set (see independent_stabilizers): repeated
    products are dropped, but redundant generators still change the walk.
    """
    G = np.asarray(stabilizers)
    k, n = len(G), G.shape[2]

    # 1. The errors: every Pauli of each weight, single-qubit errors first.
    built = [pauli_errors(n, weight) for weight in weights]
    errors = np.concatenate([E for E, _ in built])
    labels = [label for _, L in built for label in L]

    # 2. |C*|: group the errors by syndrome on the generators, dropping the
    #    undetectable ones. Same syndrome means the same entry on every
    #    product, so each class is one column, represented by its first error.
    syndromes = commutators(G, errors).T  # one row per error
    classes = {}
    for j, syndrome in enumerate(syndromes):
        if syndrome.any():
            classes.setdefault(syndrome.tobytes(), []).append(j)
    if not classes:
        raise ValueError("no error is detectable: |C*| = 0")
    keep = [group[0] for group in classes.values()]
    merged = [tuple(labels[j] for j in group) for group in classes.values()]
    D_generators = syndromes[keep].T  # (k, |C*|): each generator against each class

    # 3. Walk the orders, adding each new product as a row, until the rank
    #    reaches |C*| (or for exactly max_order orders).
    graph = correlation_graph(G)
    measured, subsets, D, seen = [], [], [], set()
    order, rank = 0, 0
    while order < k and (rank < len(keep) if max_order is None else order < max_order):
        order += 1
        for stabilizer, subset in zip(*stab_products(G, order, graph)):
            if stabilizer.tobytes() in seen:  # already made by another subset
                continue
            seen.add(stabilizer.tobytes())
            measured.append(stabilizer)
            subsets.append(subset)
            # A product's row is the sum of its generators' rows, mod 2.
            D.append(D_generators[list(subset)].sum(axis=0) % 2)
        # The real rank: D' is used as a real matrix.
        rank = np.linalg.matrix_rank(np.array(D))
    if max_order is None and rank < len(keep):
        raise RuntimeError(f"rank {rank} < |C*| = {len(keep)} at every order")

    return np.array(D), merged, np.array(measured), subsets
