"""The Gordan programme reduced by a symmetry of the code.

A permutation of the qubits that maps the measured set and the family to
themselves permutes the rows of D' and the errors, and only reorders the LP's
constraints. Averaging over the group shows a witness exists if and only if one
exists that is constant on each orbit of rows, so the LP needs one unknown per
orbit of rows and one inequality per orbit of errors.
"""

from fractions import Fraction
from math import lcm

import numpy as np
from scipy import sparse

from qecdm.stabilizer.dmatrix import pauli_errors
from qecdm.stabilizer.gordanLP import rref
from qecdm.stabilizer.gordanLP_large import pivot_rows, sparse_lp, sparse_straddles, split_mod_p, verify_witness_blocks


def induced_permutation(paulis, q):
    """np.ndarray (m, 2, n), qubit permutation (n,) -> np.ndarray (m,): where each Pauli goes.

    Moving qubit j to qubit q[j] turns Pauli i into another Pauli; entry i is
    that Pauli's index in the list. Raises ValueError if some image is not in
    the list: q is then not a symmetry of it.
    """
    paulis = np.asarray(paulis)
    n = paulis.shape[2]
    if not np.array_equal(np.sort(q), np.arange(n)):
        raise ValueError("q is not a permutation of the qubits")
    index = {p.tobytes(): i for i, p in enumerate(paulis)}
    if len(index) < len(paulis):
        raise ValueError("the list repeats a Pauli, so where it goes is ambiguous")

    moved = np.empty_like(paulis) #same shape
    moved[:, :, q] = paulis  # qubit j's [x, z] bits go to qubit q[j]
    #q is the new order permuatation and is set to the previous 0, ..., n-1 order
    try:
        return np.array([index[p.tobytes()] for p in moved], dtype=int)
    except KeyError:
        raise ValueError("q does not map the list to itself") from None


def bb_translations(code):
    """qldpc BBCode -> list of l*m qubit permutations: the translations x^s y^t.

    qldpc numbers the qubits block * l*m + m*a + b: block 0 is L and block 1 is
    R, and (a, b) is the monomial x^a y^b, the exponents taken in the order of
    code.orders. Multiplying by x^s y^t moves (a, b) to (a + s, b + t) in both
    blocks. The group is abelian, so this maps every check to a check: run
    induced_permutation on the checks to confirm it.
    """
    l, m = code.orders
    if code.num_qudits != 2 * l * m:
        raise ValueError("the code does not have 2 * l * m qubits: not a BB layout")
    block, a, b = np.unravel_index(np.arange(2 * l * m), (2, l, m))  # each qubit's (block, a, b)
    return [np.ravel_multi_index((block, (a + s) % l, (b + t) % m), (2, l, m))
            for s in range(l) for t in range(m)]


def orbits(perms):
    """list of index permutations (a whole group) -> np.ndarray: each element's orbit label.

    The label of i is the smallest index the group can move i to, so two
    elements share a label exactly when they are in the same orbit. That needs
    every element of the group, not just generators: the labels must not change
    under any of the permutations, and a ValueError is raised if they do.
    """
    labels = np.min(np.stack(perms), axis=0)
    if not all(np.array_equal(labels[p], labels) for p in perms):
        raise ValueError("the permutations are not a whole group: pass every element")
    return labels


def orbit_matrix(labels):
    """orbit labels (k,) -> scipy.sparse.csr_matrix (k, orbits): B.

    B[r, O] = 1 when element r is in orbit O, so lambda = B mu gives every
    element of orbit O the value mu_O. Orbits are columns in increasing order
    of label.
    """
    _, column = np.unique(labels, return_inverse=True)  # each element's orbit, numbered 0, 1, ...
    k = len(labels)
    return sparse.csr_matrix((np.ones(k, dtype=int), (np.arange(k), column)),
                             shape=(k, column.max() + 1))


def exact_reduced_witness(D_red, S_red, mu, max_doublings=40):
    """The solver's approximate mu -> an integer mu, proved exactly on the reduced programme.

    The reduced equalities D_red^T mu = 0 are few (6 independent on BB72), so
    they are put in reduced row echelon form exactly: each pivot entry of mu is
    then fixed by the free entries. The free entries are round(t * mu), the
    pivot entries are solved for, and the denominators cleared. t starts at 1
    and doubles until every reduced inequality holds in Python ints. (Not
    verify_witness: that sums lambda where s^P = 1, and B^T s^P is a count, not 0/1.)
    """
    A, pivots = rref(np.asarray(D_red).T)  # row i reads mu[pivots[i]] + sum_free A[i][j] mu[j] = 0
    free = [j for j in range(len(mu)) if j not in set(pivots)]
    E = np.asarray(D_red, dtype=object).T                                   # D_red^T, exactly
    T = np.asarray(sparse.csr_matrix(S_red).T.toarray(), dtype=object)     # row P: B^T s^P

    t = 1
    for _ in range(max_doublings):
        values = np.empty(len(mu), dtype=object)
        for j in free:
            values[j] = Fraction(round(t * mu[j]))
        for row, p in zip(A, pivots):
            values[p] = -sum(row[j] * values[j] for j in free)

        delta = lcm(*(v.denominator for v in values))  # a positive multiple is still a witness
        witness = np.array([int(v * delta) for v in values], dtype=object)
        if all(v == 0 for v in E @ witness) and all(v > 0 for v in T @ witness):
            return witness
        t *= 2
    raise RuntimeError("could not round the LP solution to an exact witness")


def gordan_symmetric(D, measured, perms, family=None):
    """D', measured set, qubit permutations -> (certified, absorbed, resolved, witness, physical, unphysical).

    The same decision as gordanLP_large.gordan_large, with the LP reduced by a
    symmetry. `perms` must be a whole group of qubit permutations mapping the
    measured set and the family to themselves (for a BB code, bb_translations).
    Each orbit of errors is classified by one representative, the LP has one
    unknown per orbit of rows (lambda = B mu), and lambda is accepted only if
    it is a witness for the full problem. The straddles are only ever built for
    the representatives, or a block of errors at a time, never all at once.
    """
    measured = np.asarray(measured)
    if family is None:
        family, _ = pauli_errors(measured.shape[2], 2)
    family = np.asarray(family)  # so family[indices] picks errors

    # The orbits: B for the rows, a label for each error.
    B = orbit_matrix(orbits([induced_permutation(measured, q) for q in perms]))
    error_labels = orbits([induced_permutation(family, q) for q in perms])
    reps = np.unique(error_labels)  # a label is its orbit's smallest member, so a member

    # Classify the representatives only, then give each orbit its representative's verdict.
    S_reps = sparse.csc_matrix(sparse_straddles(measured, family[reps]))  # one column per orbit
    split = split_mod_p(D, S_reps, pivot_rows(D))  # positions within reps
    absorbed, resolved, physical, unphysical = (
        np.flatnonzero(np.isin(error_labels, reps[positions])).tolist() for positions in split
    )

    # The reduced programme: lambda = B mu, one inequality per resolved orbit.
    resolved_reps = split[1]
    D_red, S_red = B.T @ D, B.T @ S_reps[:, resolved_reps]
    certified, mu = sparse_lp(D_red, S_red, list(range(len(resolved_reps))))
    if not certified:
        return False, absorbed, resolved, None, physical, unphysical

    mu = exact_reduced_witness(D_red, S_red, mu)
    witness = mu[B.indices]  # lambda = B mu: B has one entry per row, in its orbit's column
    if not verify_witness_blocks(D, measured, family, resolved, witness):
        raise RuntimeError("lambda = B mu fails on the full problem: perms is not a symmetry")
    return True, absorbed, resolved, witness, physical, unphysical
