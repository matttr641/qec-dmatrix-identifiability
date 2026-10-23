from fractions import Fraction
from math import gcd, lcm

import numpy as np
from scipy.optimize import linprog

from qecdm.dmatrix import commutators, pauli_errors


def straddles(measured, family):
    """np.ndarray (rows, 2, n), (m, 2, n) -> np.ndarray (rows, m): s^P for each P.

    Column j is the straddle of family[j]: 1 where a measured stabilizer
    anticommutes with that error.
    """
    return commutators(measured, family)


def rref(M, columns=None):
    """np.ndarray (r, c) -> (list of Fraction rows, pivots): reduced row echelon form.

    Exact, over the rationals. Pivots are only sought in the first `columns`
    columns (default all); later columns are carried along, as in solving
    [A | b]. `pivots[i]` is the column of row i's pivot.
    """
    A = [[Fraction(int(v)) for v in row] for row in np.asarray(M)]
    columns = len(A[0]) if columns is None else columns
    pivots = []
    for column in range(columns):
        r = len(pivots)  # the next row to hold a pivot
        pivot = next((i for i in range(r, len(A)) if A[i][column] != 0), None)
        if pivot is None:
            continue
        A[r], A[pivot] = A[pivot], A[r]
        A[r] = [value / A[r][column] for value in A[r]]  # scale the pivot to 1
        for i in range(len(A)):                          # clear the column elsewhere
            if i != r and A[i][column] != 0:
                factor = A[i][column]
                A[i] = [a - factor * b for a, b in zip(A[i], A[r])]
        pivots.append(column)
        if len(pivots) == len(A):
            break
    return A, pivots


def left_kernel(D):
    """np.ndarray (rows, cols) -> np.ndarray (rows, kappa) of Python ints: a basis of ker(D^T).

    Exact: Gaussian elimination over the rationals, then each basis vector is
    scaled to the smallest integers. kappa = rows - rank(D), the real rank.
    """
    A, pivots = rref(np.asarray(D).T)
    rows = len(A[0])  # the number of rows of D, i.e. the length of each kernel vector

    # One basis vector per free (non-pivot) column: set it to 1, solve for the pivots.
    basis = []
    for free in (c for c in range(rows) if c not in pivots):
        v = [Fraction(0)] * rows
        v[free] = Fraction(1)
        for i, column in enumerate(pivots):
            v[column] = -A[i][free]
        v = [int(x * lcm(*(x.denominator for x in v))) for x in v]  # clear denominators
        divisor = gcd(*v)
        basis.append([x // divisor for x in v])                     # smallest integers
    return np.array(basis, dtype=object).T.reshape(rows, len(basis))


def split_shadows(K, S):
    """np.ndarray (rows, kappa), (rows, m) -> (absorbed, resolved, C).

    C = K^T S holds each straddle's coordinates against ker(D^T), exactly in
    integers. A zero column means the straddle lies in im(D), so its shadow is
    zero and the error is absorbed; any other column is resolved. The shadows
    themselves are never computed.
    """
    C = K.T @ np.asarray(S, dtype=object)
    is_resolved = (C != 0).any(axis=0)
    absorbed = [int(i) for i in np.flatnonzero(~is_resolved)]
    resolved = [int(i) for i in np.flatnonzero(is_resolved)]
    return absorbed, resolved, C


def split_absorbed(D, S, absorbed):
    """np.ndarray (rows, cols), (rows, m), list -> (physical, unphysical, fits).

    Each absorbed straddle is exactly s = D g for single-qubit rates g, one per
    column of D. The error, alone, gives a physical report if every entry of g
    is >= 0, and an unphysical one if any is negative. `fits` holds each g as a
    column of Fractions. Needs D at full column rank (where Algorithm 1 stops),
    so that g is unique.
    """
    D = np.asarray(D)
    cols = D.shape[1]
    # Reduce [D | S_absorbed]: D's part becomes the identity, S's part is then g.
    A, pivots = rref(np.hstack([D, np.asarray(S)[:, absorbed]]), columns=cols)
    if len(pivots) < cols:
        raise ValueError("D is rank deficient, so the single-qubit fit is not unique")
    fits = np.array([row[cols:] for row in A[:cols]], dtype=object).reshape(cols, len(absorbed))

    physical = [p for p, g in zip(absorbed, fits.T) if all(x >= 0 for x in g)]
    unphysical = [p for p in absorbed if p not in physical]
    return physical, unphysical, fits


def gordan_lp(C, resolved, K):
    """np.ndarray (kappa, m), list, np.ndarray (k, kappa) -> (certified, witness).

    Solves the programme of Remark 6.6 in kernel coordinates: find y with
    <c^P, y> >= 1 for every resolved P. The solver only proposes y; the verdict
    is then checked exactly. y is scaled and rounded to integers, every product
    is recomputed in integers, and the witness lambda = K y lies in ker(D^T)
    exactly. Infeasible is branch (ii), returned as (False, None).
    """
    if not resolved:
        return False, None  # nothing resolved, so nothing to certify
    A = np.asarray(C[:, resolved])

    program = linprog(
        c=np.zeros(A.shape[0]),
        A_ub=-A.astype(float).T,     # -A^T y <= -1 is A^T y >= 1
        b_ub=-np.ones(A.shape[1]),
        bounds=(None, None),
    )
    if program.status == 2:
        return False, None           # infeasible: branch (ii), hiding is possible
    if program.status != 0:
        raise RuntimeError(f"the LP solver failed: {program.message}")

    # Scale by t >= the largest |c^P|_1 and round: each product moves by at most
    # |c^P|_1 / 2 < t, while the scaled products are at least t, so all stay > 0.
    t = max(sum(abs(v) for v in column) for column in A.T)
    y = [round(t * value) for value in program.x]
    products = A.T @ np.array(y, dtype=object)
    if not all(product > 0 for product in products):
        raise RuntimeError("the rounded witness failed its exact check")
    return True, K @ np.array(y, dtype=object)


def gordan(D, measured, family=None):
    """D', measured set, family -> (certified, absorbed, resolved, witness).

    Theorem 6.5 decided exactly. `family` defaults to every two-qubit Pauli
    error. certified is True for branch (i), with witness lambda in ker(D^T)
    and <lambda, s^P> > 0 for every resolved P, all in integers.

    Top level function, runs the whole chain.
    """
    if family is None:
        family, _ = pauli_errors(np.asarray(measured).shape[2], 2)
    S = straddles(measured, family)
    K = left_kernel(D)
    absorbed, resolved, C = split_shadows(K, S)
    certified, witness = gordan_lp(C, resolved, K)
    return certified, absorbed, resolved, witness
