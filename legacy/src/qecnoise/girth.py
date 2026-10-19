"""How small can a hiding conspiracy be? This finds the physical girth m*.

Thesis §6.4.

In   D', the measured set, the declared family, and a cap on the search size.
Out  the errors that hide alone, the spark, m* itself, and the errors that
     attain it. m* is infinity when no resolved errors can hide together.

The search is exhaustive and so exponential. If the cap is reached first, the
bound 2 <= spark <= m* <= rank(B_R) + 1 still holds and is reported instead.
Run this file directly for both families on both measured sets of the chain.
"""

from itertools import combinations

import numpy as np

from qecnoise.Dmatrixsinglequbit import bit_flip_chain, design_matrix, flips
from qecnoise.gordanLP import gordan, shadows


# Sweep subsets by size: smallest dependent set, then smallest positive one.
def spark_and_girth(B, max_size=6, tol=1e-9):
    """Smallest dependent subset of the columns, and smallest positive circuit.

    Returns (spark, size, members). `size` is the girth when a positive circuit
    is found, and None when the search reached `max_size` without one. `spark`
    is None if no dependent subset at all was found within the cap.
    """
    spark = None
    for size in range(2, min(max_size, B.shape[1]) + 1):
        for subset in combinations(range(B.shape[1]), size):
            _, values, rows = np.linalg.svd(B[:, subset], full_matrices=False)
            if values[-1] > tol:
                continue  # independent
            if spark is None:
                spark = size
            if values[-2] <= tol:
                continue  # dependent, but rank drops by more than one
            direction = rows[-1]  # spans the one-dimensional null space
            if np.abs(direction).min() > tol and (
                np.all(direction > 0) or np.all(direction < 0)
            ):
                return spark, size, list(subset)
    return spark, None, None


# Ask gordan first; only search for a circuit if something can hide.
def physical_girth(D, measured, family, max_size=6):
    """Definition 6.7 for a measured set and a declared family of errors.

    Returns (absorbed, spark, m_star, members):

    absorbed  indices of family members that hide alone, treated separately by
              the T = empty clause of Lemma 6.4.
    spark     the linear bracket, None when the LP certifies safety.
    m_star    np.inf when the program is feasible (branch (i), nothing hides);
              an integer when a positive circuit is found; None when the search
              was capped before finding one, so m_star > max_size.
    members   family indices of the circuit attaining m_star.
    """
    certified, absorbed, resolved, _ = gordan(D, measured, family)
    if certified:
        return absorbed, None, np.inf, []

    B = shadows(D, measured, family)[:, resolved]
    spark, size, subset = spark_and_girth(B, max_size)
    members = [resolved[i] for i in subset] if subset else []
    return absorbed, spark, size, members


# Print one case: absorbed list, the brackets, and m* or its range.
def _show_girth(title, D, measured, family, labels, max_size=6):
    absorbed, spark, m_star, members = physical_girth(D, measured, family, max_size)
    resolved = len(family) - len(absorbed)

    print(f"{title}")
    print(f"  {len(measured)} rows, family of {len(family)}: "
          f"{len(absorbed)} absorbed, {resolved} resolved")
    print(f"  absorbed: "
          + (", ".join(str(labels[i]) for i in absorbed) or "none"))
    if m_star == np.inf:
        print("  LP feasible -> nothing in this family can hide, m* = infinity\n")
        return
    # Only needed on the branch that prints it, and the shadows are cheap here:
    # a family with a circuit to find is one the search above already ran on.
    ceiling = np.linalg.matrix_rank(shadows(D, measured, family)) + 1
    print(f"  spark = {spark}, upper bound rank(B_R) + 1 = {ceiling}")
    if m_star is None:
        print(f"  no positive circuit of size <= {max_size}, so "
              f"{max_size + 1} <= m* <= {ceiling}\n")
    else:
        print(f"  m* = {m_star}, attained by: "
              + ", ".join(str(labels[i]) for i in members) + "\n")


if __name__ == "__main__":
    n = 7
    pair_labels = list(combinations(range(n), 2))
    triple_labels = list(combinations(range(n), 3))
    doubles = [flips(n, *p) for p in pair_labels]
    mixed = doubles + [flips(n, *t) for t in triple_labels]
    mixed_labels = pair_labels + triple_labels

    D, _, measured = design_matrix(n, 6, bit_flip_chain(n))
    D21, _, measured21 = design_matrix(n, 6, bit_flip_chain(n), max_order=6)

    _show_girth("eleven-row set, pure pairs", D, measured, doubles, pair_labels)
    _show_girth("eleven-row set, mixed pairs and triples",
                D, measured, mixed, mixed_labels)
    _show_girth("twenty-one-row set, pure pairs", D21, measured21, doubles, pair_labels)
    # The mixed family on 21 rows has 56 resolved shadows, so size 6 means 32
    # million subsets. Capped at 4 here: enough to show that spark = 4 is not
    # attained, which is the gap positivity opens (§8.3).
    _show_girth("twenty-one-row set, mixed pairs and triples",
                D21, measured21, mixed, mixed_labels, max_size=5)
