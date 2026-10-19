"""Thesis §6.1-6.3. One linear programme determines whether any channel with resolved errors from a
declared family can ever be invisible. FEASIBLE programme is the good news, and hands back a witness proving nothing can hide.

In   D', the measured set, and the family of errors being declared.
Out  which errors are absorbed and which resolved, the verdict, and a witness
     when there is one.

This file also holds the blocked versions of these routines, which the larger
codes need. Run this file directly for the two measured sets of the chain.
"""

from fractions import Fraction
from math import gcd

import numpy as np
from scipy.optimize import linprog

from qecnoise.Dmatrixsinglequbit import (
    bit_flip_chain,
    commutators,
    design_matrix,
    pair_family,
)


# s^P for each candidate error: which stabilizers see it.
def straddles(measured, family):
    """s^P for every P in the family, one column each (Def. 5.2)."""
    return commutators(np.asarray(measured), np.asarray(family))


# Orthonormal columns spanning im(D), the small side of the projector.
def image_basis(D, tol=1e-9):
    """Orthonormal basis for im(D), as columns.

    Taken from the image rather than the kernel because rank(D) is bounded by
    the number of syndrome classes, while ker(D^T) grows with the number of
    measured stabilizers: on the bivariate bicycle code those are 216 and
    12564, and the kernel basis alone would not fit in memory.
    """
    columns, values, _ = np.linalg.svd(np.asarray(D, dtype=float),
                                       full_matrices=False)
    return columns[:, values > tol]


# Strip off what a single-qubit fit could reproduce, leaving beta.
def shadows(D, measured, family, block=None, dtype=float):
    """beta^P = (I - Pi_D) s^P, one column each (Def. 6.1).

    `block` assembles the columns in chunks rather than all at once, and
    `dtype` chooses the precision. Both matter only at scale: for the bicycle
    code the result is 12780 x 23004, which is 1.4 GB in double and 0.69 GB in
    single. The defaults give the whole matrix in double, which is what every
    small case wants.
    """
    Q = image_basis(D).astype(dtype)
    measured, family = np.asarray(measured), np.asarray(family)
    step = max(1, len(family) if block is None else block)
    out = np.empty((len(measured), len(family)), dtype=dtype)
    for start in range(0, len(family), step):
        S = commutators(measured, family[start:start + step]).astype(dtype)
        out[:, start:start + step] = S - Q @ (Q.T @ S)
    return out


# ||beta|| for every error, in blocks, never holding the matrix.
def shadow_norms(D, measured, family, block=1500):
    """Euclidean norm of every shadow, in blocks, without holding the matrix.

    beta = s - Q Q^T s with Q an orthonormal basis of im(D), so a block costs
    one projection of that block alone.
    """
    Q = image_basis(D)
    measured = np.asarray(measured)
    family = np.asarray(family)
    # One block at a time: project it, keep the column norms, drop the block.
    norms = []
    for start in range(0, len(family), block):
        S = commutators(measured, family[start:start + block]).astype(float)
        norms.append(np.linalg.norm(S - Q @ (Q.T @ S), axis=0))
    return np.concatenate(norms)


# Check a candidate witness against every resolved shadow.
def verify_witness(D, measured, family, witness, block=1500, tol=1e-9):
    """Check a stored witness: is <lambda, beta^P> > 0 for every resolved P?

    Returns (smallest product, number of non-positive products, resolved count).
    """
    # Project the witness into ker(D^T), which makes the straddle shortcut exact.
    Q = image_basis(D)
    lam = np.asarray(witness, dtype=float)
    lam = lam - Q @ (Q.T @ lam)
    measured = np.asarray(measured)
    family = np.asarray(family)

    # Block by block: form the shadows to see which members are resolved, then
    # take the products against the straddles, which are equal and cheaper.
    worst, negatives, resolved = np.inf, 0, 0
    for start in range(0, len(family), block):
        S = commutators(measured, family[start:start + block]).astype(float)
        keep = np.linalg.norm(S - Q @ (Q.T @ S), axis=0) > tol
        if not keep.any():
            continue
        products = (lam @ S)[keep]
        worst = min(worst, float(products.min()))
        negatives += int((products <= 0).sum())
        resolved += int(keep.sum())
    return worst, negatives, resolved


# The cheap guess: lambda = sum of shadows, then test it.
def summed_shadow_certificate(D, measured, family, block=1500, normalise=False):
    """Try lambda = sum of shadows as a Gordan witness (Thm. 6.5, branch (i)).

    Returns (witness, products). The witness lies in ker(D^T) by construction.
    If every product is strictly positive the witness
    certifies branch (i) and m* = infinity. If any is not, the test is
    inconclusive and the full programme of Remark 6.6 would be needed.

    The test is applied to the resolved members only.

    `normalise` scales each shadow to unit length before summing, so that the
    sum is not dominated by the largest shadows. It is a different candidate
    witness, not a better-conditioned computation of the same one.
    """
    Q = image_basis(D)
    measured = np.asarray(measured)
    family = np.asarray(family)

    # Pass one: accumulate the sum of the shadows into the candidate witness.
    witness = np.zeros(len(measured))
    for start in range(0, len(family), block):
        S = commutators(measured, family[start:start + block]).astype(float)
        block_shadows = S - Q @ (Q.T @ S)
        if normalise:
            sizes = np.linalg.norm(block_shadows, axis=0)
            block_shadows = block_shadows / np.where(sizes > 0, sizes, 1)
        witness += block_shadows.sum(axis=1)

    # Pass two: the products. The witness is a sum of shadows, hence already in
    # ker(D^T), so the straddles give the same answer without a projection.
    products = []
    for start in range(0, len(family), block):
        S = commutators(measured, family[start:start + block]).astype(float)
        products.append(witness @ S)
    return witness, np.concatenate(products)



# Exact solve, so that nothing after it has to round.
def _exact_solve(A, b):
    """Solve A x = b exactly, for integer A and b. Raises if A is singular.

    The elimination is Bareiss, which stays in the integers:

        below[j] = (below[j] * pivot - factor * row[j]) // previous

    Ordinary elimination would divide by the pivot here and land in fractions.
    Bareiss multiplies through by it instead, then divides by the pivot from
    the step before. That division is always exact, because every entry is a
    minor of A, and it is what stops the numbers doubling in size each step.
    Fractions appear only in the back-substitution at the end.
    """
    # The augmented matrix [A | b], Python integers throughout.
    n = len(A)
    M = [[int(A[i][j]) for j in range(n)] + [int(b[i])] for i in range(n)]

    previous = 1                                   # last pivot; 1 before the first
    for k in range(n - 1):
        # A zero on the diagonal needs a row swapped up. None available means
        # A is singular, which for D^T D would mean D is rank deficient.
        if M[k][k] == 0:
            swap = next((r for r in range(k + 1, n) if M[r][k]), None)
            if swap is None:
                raise ValueError("singular")
            M[k], M[swap] = M[swap], M[k]

        pivot, row = M[k][k], M[k]
        for i in range(k + 1, n):
            below, factor = M[i], M[i][k]
            for j in range(k + 1, n + 1):
                below[j] = (below[j] * pivot - factor * row[j]) // previous
            below[k] = 0
        previous = pivot
    if M[n - 1][n - 1] == 0:
        raise ValueError("singular")

    # Upper triangular now, so back-substitute. This is the only division that
    # cannot be exact in the integers, hence the only place fractions appear.
    x = [Fraction(0)] * n
    for i in range(n - 1, -1, -1):
        total = Fraction(M[i][n])
        for j in range(i + 1, n):
            total -= Fraction(M[i][j]) * x[j]
        x[i] = total / Fraction(M[i][i])
    return x


# Settle the verdict in integers, so no floating point decides it.
def certify_witness(D, measured, family, witness, block=2000):
    """Prove <v, beta^P> > 0 for every P, with no floating point in the verdict.

    The argument, in five steps:

    1. round `witness` to an integer vector z (or keep it, if already integral);
    2. certify v = (I - Pi)z instead, which lies in ker(D^T) identically;
    3. then <v, beta^P> = <z, s^P> - c^T(D^T s^P), where c solves the single
       216 x 216 system (D^T D)c = D^T z;
    4. solve that exactly and clear c's denominator d, which turns every
       product into the integer d<z, s^P> - C^T(D^T s^P);
    5. check all of them are positive.

    The floats on the way are sums of 0/1 products, small enough to be exact in
    float64 and bounds-checked against that. Returns (proved, worst, checked),
    worst being an exact Fraction.

    A pass is unconditional: v is a strict witness, so branch (i) of Thm. 6.5
    holds.
    """
    D = np.asarray(D).astype(np.int64)
    w = np.asarray(witness, dtype=float)
    rows = D.shape[0]

    # Integer seed. A witness that is already integral - the summed straddles,
    # for one - is kept as it is, so nothing is rounded at all.
    if np.array_equal(w, np.rint(w)) and abs(w).max() * rows < 2**52:
        z = np.rint(w).astype(np.int64)
    else:
        z = np.rint(w / abs(w).max() * (2**50 / rows)).astype(np.int64)

    G = np.rint(D.T.astype(float) @ D.astype(float)).astype(np.int64)
    a = np.rint(D.T.astype(float) @ z.astype(float)).astype(np.int64)
    if max(int(abs(G).max()), int(abs(a).max())) >= 2**52:
        raise ValueError("witness too large for the exact float64 sums")

    c = _exact_solve(G.tolist(), a.tolist())
    d = 1
    for value in c:
        d = d * value.denominator // gcd(d, value.denominator)
    C = [int(value * d) for value in c]

    # v = d*z - D C lies in ker(D^T) exactly iff d*(D^T z) == G C. Checking that
    # in integers is what makes the straddles below stand in for the shadows.
    columns = G.shape[0]
    for i in range(columns):
        if d * int(a[i]) != sum(int(G[i][j]) * C[j] for j in range(columns)):
            raise RuntimeError("the exact projection failed its own check")

    worst, checked, family = None, 0, np.asarray(family)
    for start in range(0, len(family), block):
        S = commutators(np.asarray(measured), family[start:start + block]).astype(float)
        zS = np.rint(z.astype(float) @ S).astype(np.int64).tolist()
        DS = np.rint(D.T.astype(float) @ S).astype(np.int64).tolist()
        for col in range(len(zS)):
            product = d * zS[col] - sum(C[i] * DS[i][col] for i in range(columns))
            worst = product if worst is None or product < worst else worst
            checked += 1
    return (worst is not None and worst > 0), Fraction(worst or 0, d), checked


# Split absorbed from resolved, then one LP decides the dichotomy.
def gordan(D, measured, family, tol=1e-9):
    """Remark 6.6: solve for lambda with B^T lambda >= 1 over the resolved shadows.

    Returns (certified, absorbed, resolved, witness):

    certified  True iff the programme is feasible, i.e. branch (i) -- no physical
               channel of this family with resolved content is ever invisible.
    absorbed   indices of family members with zero shadow, which hide alone.
    resolved   indices of the rest, the ones the programme is about.
    witness    the certifying lambda when feasible, otherwise None.

    The programme is stated on shadows where Theorem 6.5 states it on straddles;
    the two differ by a vector in im(D), which every lambda in ker(D^T)
    annihilates, so the test is the same.
    """
    B = shadows(D, measured, family)
    sizes = np.linalg.norm(B, axis=0)
    absorbed = list(np.flatnonzero(sizes <= tol))
    resolved = list(np.flatnonzero(sizes > tol))
    if not resolved:
        return False, absorbed, resolved, None  # nothing to certify: all hide alone

    program = linprog(
        c=np.zeros(B.shape[0]),
        A_ub=-B[:, resolved].T,  # -B^T lambda <= -1 is B^T lambda >= 1
        b_ub=-np.ones(len(resolved)),
        bounds=(None, None),
    )
    feasible = program.status == 0
    return feasible, absorbed, resolved, (program.x if feasible else None)


# When the LP is too big: hunt a witness by min-norm point instead.
def witness_search(B, iterations=20000, tol=1e-12):
    """Search for a strict Gordan witness when the programme is too big to solve.

    Theorem 6.5(i) asks for some lambda with <lambda, beta^P> > 0 for every
    resolved P. That holds exactly when the origin lies outside the convex hull
    of the shadows, so the search hunts for the weighted average of the shadows
    lying closest to the origin, by away-step Frank-Wolfe. Two properties make
    it the right formulation: a weighted average of shadows is already in
    ker(D^T), and the moment its smallest inner product with any shadow turns
    positive it *is* a witness and there is no need to reach the optimum.

    Unlike `gordan`, a failure here is not a verdict: it means no witness was
    found within `iterations`, not that none exists.

    Returns (witness, margin) on success and (None, best margin seen) on
    failure. The margin is a *cosine*: the worst inner product divided by the
    length of the point. Dividing matters. When no witness exists the point
    shrinks towards zero, its raw inner products shrink with it, and their
    signs are then only rounding noise. So a test on the raw products
    eventually reads all-positive and reports a witness that is not there.
    """
    # 1. Unit columns. Scaling a shadow does not change the sign of its inner
    #    product, and it makes the geometry (and the arithmetic) uniform.
    #    Absorbed errors have zero shadow and are dropped: Theorem 6.5 speaks
    #    only of the resolved ones, and they would divide by zero here.
    B = np.asarray(B)
    B = B[:, np.linalg.norm(B, axis=0) > tol]
    B = B / np.linalg.norm(B, axis=0)

    # 2. Start at the average of all shadows - the summed witness, which is
    #    this algorithm's first iterate rather than a rival to it.
    weights = np.full(B.shape[1], 1.0 / B.shape[1], dtype=B.dtype)
    point = B @ weights
    best = -np.inf

    for _ in range(iterations):
        # 3. Inner product of the current point with every shadow, divided by
        #    the point's own length so the test is a cosine. A collapsing
        #    point means the origin is inside the hull - branch (ii) -- and
        #    there is nothing to find.
        products = B.T @ point
        size = float(np.linalg.norm(point))
        if size <= tol:
            break
        worst = int(products.argmin())
        margin = float(products[worst]) / size
        best = max(best, margin)
        if margin > tol:
            return point, margin                          # a strict witness

        # 4. Two candidate directions. The Frank-Wolfe one pulls towards the
        #    shadow that is currently worst; the away one pushes off the active
        #    shadow that is best, which is what stops the zigzagging that
        #    plain Frank-Wolfe suffers near a boundary.
        active = np.flatnonzero(weights > 0)
        away = int(active[products[active].argmax()])
        square = float(point @ point)
        if square - products[worst] >= products[away] - square:
            direction, cap, index, is_away = B[:, worst] - point, 1.0, worst, False
        else:
            direction, index, is_away = point - B[:, away], away, True
            cap = weights[away] / (1 - weights[away]) if weights[away] < 1 else 1.0

        # 5. Exact line search: the step minimising ||point + gamma*direction||.
        denominator = float(direction @ direction)
        if denominator <= tol:
            break
        gamma = min(max(-float(point @ direction) / denominator, 0.0), cap)

        # 6. Move, and keep the weights consistent with the point.
        point = point + gamma * direction
        if is_away:
            weights = weights * (1 + gamma)
            weights[index] -= gamma
        else:
            weights = weights * (1 - gamma)
            weights[index] += gamma

    return None, best


# Print one measured set: what hides alone, and the LP's verdict.
def _show_dichotomy(title, D, measured, family, labels):
    certified, absorbed, resolved, witness = gordan(D, measured, family)
    print(f"{title}: {len(measured)} rows, family of {len(family)}")
    print(f"  absorbed ({len(absorbed)}): "
          + (", ".join(str(labels[i]) for i in absorbed) or "none"))
    print(f"  resolved: {len(resolved)}")
    if certified:
        print(f"  LP FEASIBLE -> branch (i): certified safe for this family")
        print(f"     witness lambda, first entries: {np.round(witness[:5], 3)}\n")
    else:
        print("  LP INFEASIBLE -> branch (ii): hiding configurations exist\n")


if __name__ == "__main__":
    n = 7
    doubles, pairs = pair_family(n)

    D, _, measured = design_matrix(n, 6, bit_flip_chain(n))
    D21, _, measured21 = design_matrix(n, 6, bit_flip_chain(n), max_order=6)

    _show_dichotomy("eleven-row set, where Algorithm 1 stops",
                    D, measured, doubles, pairs)
    _show_dichotomy("twenty-one-row set, the complete weight-2 set",
                    D21, measured21, doubles, pairs)
