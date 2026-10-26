"""The Gordan programme for large codes: the same decision as gordanLP.py,
computed without ever forming the kernel matrix K.

The idea is to fit each straddle on c independent rows of D' (the pivot rows)
and check the fit on the rest. Arithmetic is exact modulo a prime p, which
keeps it vectorised in int64.
"""

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from qecdm.gordanLP import rref

# A prime below 2**26: a sum of c products of residues stays below 2**63 for
# c up to about 2**11 classes, so int64 never overflows.
PRIME = 67_108_859


def pivot_rows(D, p=PRIME):
    """np.ndarray (k, c) -> list of c row indices: rows of D' that are independent.

    These are the pivot columns of D^T, found by eliminating D^T modulo p. A set
    of rows independent mod p is independent over the rationals too, since a
    determinant that is nonzero mod p is nonzero. So D' restricted to these rows
    is an invertible c x c matrix.
    """
    A = np.asarray(D, dtype=np.int64).T % p  # D^T: one row per class, one column per row of D'
    pivots = []
    for column in range(A.shape[1]):
        r = len(pivots)                                  # the next row to hold a pivot
        candidates = np.flatnonzero(A[r:, column])
        if len(candidates) == 0:
            continue                                     # no pivot here: a free column
        i = r + candidates[0]
        A[[r, i]] = A[[i, r]]                            # swap it up
        A[r] = A[r] * pow(int(A[r, column]), -1, p) % p  # scale the pivot to 1 (inverse mod p)
        others = np.flatnonzero(A[:, column])
        others = others[others != r]
        A[others] = (A[others] - np.outer(A[others, column], A[r])) % p  # clear the column
        pivots.append(column)
        if len(pivots) == len(A):
            break
    if len(pivots) < D.shape[1]:
        raise ValueError("fewer pivots than classes: D' is rank deficient, or p divides a minor")
    return pivots


def sparse_straddles(measured, family):
    """np.ndarray (k, 2, n), (m, 2, n) -> scipy.sparse.csr_matrix (k, m): s^P for each P.

    The same matrix as gordanLP.straddles, stored sparsely: only the 1s are kept.
    Entry (i, j) is <M_i, P_j> = x_M . z_P + z_M . x_P mod 2, as two sparse
    matrix products.
    """
    measured, family = np.asarray(measured), np.asarray(family)
    X_M, Z_M = sparse.csr_matrix(measured[:, 0]), sparse.csr_matrix(measured[:, 1])
    X_E, Z_E = sparse.csr_matrix(family[:, 0]), sparse.csr_matrix(family[:, 1])

    S = (X_M @ Z_E.T + Z_M @ X_E.T).tocsr()
    S.data %= 2          # the counts of anticommuting qubits, reduced mod 2
    S.eliminate_zeros()  # an even count is a 0: drop it from the stored entries
    return S


def split_rows(M, pivots):
    """matrix (k, ...), list -> (M_P, M_F): the pivot rows, and every other row.

    Works on dense arrays and sparse matrices alike. The other rows keep their
    order, so row j of M_F is the j-th non-pivot row of M.
    """
    free = np.setdiff1d(np.arange(M.shape[0]), pivots)
    return M[pivots], M[free]


def inverse_mod(M, p=PRIME):
    """np.ndarray (c, c) -> np.ndarray (c, c): M^-1 modulo p.

    Gauss-Jordan on [M | I] mod p, the same row operations as pivot_rows: when
    the left half becomes I, the right half is the inverse.
    """
    c = len(M)
    A = np.hstack([np.asarray(M, dtype=np.int64) % p, np.eye(c, dtype=np.int64)])
    for column in range(c):
        candidates = np.flatnonzero(A[column:, column])
        if len(candidates) == 0:
            raise ValueError("M is singular mod p")
        i = column + candidates[0]
        A[[column, i]] = A[[i, column]]
        A[column] = A[column] * pow(int(A[column, column]), -1, p) % p
        others = np.flatnonzero(A[:, column])
        others = others[others != column]
        A[others] = (A[others] - np.outer(A[others, column], A[column])) % p
    return A[:, c:]


def _exact_fit(D_P, s_P, D, s):
    """The exact fallback for one error: g with D g = s, or None if there is none.

    Solves D_P g = s_P with Fractions, then checks the fit on every row of D.
    """
    c = len(D_P)
    A, _ = rref(np.hstack([D_P, s_P.reshape(-1, 1)]), columns=c)
    g = [row[c] for row in A[:c]]
    fits = all(sum(D[i, j] * g[j] for j in range(c)) == s[i] for i in range(len(D)))
    return g if fits else None


def split_mod_p(D, S, pivots, p=PRIME, block=2000):
    """D', straddles, pivot rows -> (absorbed, resolved, physical, unphysical, M, C).

    For every error: the fit g = D_P^-1 s_P and the residual c = s_F - D_F g,
    both mod p. c is the error's kernel coordinates, so c != 0 proves it
    resolved. c = 0 makes it a candidate for absorbed, confirmed by lifting g to
    integers and checking D g = s exactly. A confirmed error is physical when
    every entry of g is >= 0. M = D_P^-1 mod p and C (kappa x m, mod p) are
    returned for the later steps. The family is processed `block` columns at a
    time, so the dense straddles are never all held at once.
    """
    D = np.asarray(D, dtype=np.int64)
    D_P, D_F = split_rows(D, pivots)
    M = inverse_mod(D_P, p)

    absorbed, resolved, physical, unphysical, C_blocks = [], [], [], [], []
    for start in range(0, S.shape[1], block):
        S_block = S[:, start:start + block]
        S_block = S_block.toarray() if sparse.issparse(S_block) else np.asarray(S_block)
        S_P, S_F = split_rows(S_block.astype(np.int64), pivots)

        G = M @ S_P % p                 # the fits, mod p          (c x block)
        C = (S_F - D_F @ G) % p         # the residuals, mod p     (kappa x block)
        C_blocks.append(C)

        for j in range(S_block.shape[1]):
            error = start + j
            if C[:, j].any():
                resolved.append(error)  # c != 0 mod p, so c != 0: proved resolved
                continue
            # Candidate absorbed: lift g to integers, residues above p/2 being negative.
            g = np.where(G[:, j] > p // 2, G[:, j] - p, G[:, j])
            if not np.array_equal(D @ g, S_block[:, j]):
                g = _exact_fit(D_P, S_P[:, j], D, S_block[:, j])  # not a small integer fit
                if g is None:
                    resolved.append(error)  # c was 0 mod p only by coincidence
                    continue
            absorbed.append(error)
            (physical if all(x >= 0 for x in g) else unphysical).append(error)

    return absorbed, resolved, physical, unphysical, M, np.hstack(C_blocks)


def sparse_lp(D, S, resolved):
    """D', straddles, resolved indices -> (feasible, lambda): Theorem 6.5 as a sparse LP.

    Finds lambda in R^k with <lambda, s^P> >= 1 for every resolved P and
    D^T lambda = 0, the programme stated on straddles. The equality keeps
    lambda in ker(D^T); without it lambda could pass using its im(D) part.
    Every matrix stays sparse. lambda is the solver's floats, only approximately
    in ker(D^T): step 5 makes it exact. Infeasible is branch (ii), (False, None).
    """
    if not resolved:
        return False, None  # nothing resolved, so nothing to certify
    S_R = sparse.csc_matrix(S)[:, resolved]  # csc: fast column selection
    k = S_R.shape[0]

    # Interior point decides these problems quickly; simplex can stall on the
    # free variables and equalities, so it is only the fallback.
    for method in ("highs-ipm", "highs-ds"):
        program = linprog(
            c=np.zeros(k),                            # no objective: feasibility only
            A_ub=-S_R.T,                              # -S_R^T lambda <= -1 is S_R^T lambda >= 1
            b_ub=-np.ones(len(resolved)),
            A_eq=sparse.csr_matrix(np.asarray(D).T),  # D^T lambda = 0
            b_eq=np.zeros(np.asarray(D).shape[1]),
            bounds=(None, None),                      # lambda may be negative
            method=method,
        )
        if program.status in (0, 2):
            break
    if program.status == 2:
        return False, None  # infeasible: branch (ii), hiding is possible
    if program.status != 0:
        raise RuntimeError(f"the LP solver failed: {program.message}")
    return True, program.x
