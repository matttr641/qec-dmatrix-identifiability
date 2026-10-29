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


def connected_subsets(graph):
    """np.ndarray (k, k) of bools -> generator of lists: the connected subsets, by size.

    Yields every connected subset of size 1, then size 2, and so on, each as a
    sorted tuple of node indices, each list in the order itertools.combinations
    would give. Size r is grown from size r - 1: every connected subset is a
    smaller connected one plus a neighbour. Only connected subsets are visited.
    """
    k = len(graph)
    neighbours = [set(np.flatnonzero(graph[i])) - {i} for i in range(k)]
    level = [(i,) for i in range(k)]
    while True:
        yield level
        grown = set()
        for subset in level:
            for v in set().union(*(neighbours[i] for i in subset)) - set(subset):
                grown.add(tuple(sorted(subset + (v,))))  # sorted, so each subset once
        level = sorted(grown)


def stab_products_by_order(stabilizers, graph=None):
    """np.ndarray (k, 2, n) -> generator of (np.ndarray (m, 2, n), list): products, subsets.

    For order 1, 2, ...: every connected set of that many stabilizers multiplied
    together, with the stabilizer indices that made each one. Products that come
    out as the identity are dropped. `graph` defaults to the correlation graph;
    pass your own to add or remove edges.
    """
    G = np.asarray(stabilizers)
    if graph is None:
        graph = correlation_graph(G)
    bits = (G % 2).astype(np.uint8)  # one byte per bit, to keep big levels small

    for level in connected_subsets(graph):
        if not level:
            yield np.zeros((0, *G.shape[1:]), dtype=int), []
            continue
        index = np.array(level)  # (m, order): row j lists subset j's stabilizers
        # Multiplying Paulis is adding their [x, z] bits mod 2: XOR them in, one member at a time.
        products = np.zeros((len(level), *G.shape[1:]), dtype=np.uint8)
        for member in index.T:
            products ^= bits[member]
        keep = products.reshape(len(level), -1).any(axis=1)  # drop the identity
        yield products[keep].astype(int), [s for s, kept in zip(level, keep) if kept]


def stab_products(stabilizers, order, graph=None):
    """np.ndarray (k, 2, n), int -> (np.ndarray (m, 2, n), list): products, subsets.

    The products of every connected set of exactly `order` stabilizers; see
    stab_products_by_order.
    """
    for r, result in enumerate(stab_products_by_order(stabilizers, graph), start=1):
        if r == order:
            return result


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
    levels = stab_products_by_order(G)  # one order at a time, each built from the last
    measured, subsets, D, seen = [], [], [], set()
    order, rank = 0, 0
    while order < k and (rank < len(keep) if max_order is None else order < max_order):
        order += 1
        for stabilizer, subset in zip(*next(levels)):
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
