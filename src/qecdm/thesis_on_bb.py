"""Apply the thesis pipeline to qldpc's bivariate bicycle codes: D', absorbed/resolved, Gordan LP.

Run this file directly for the [[72,12,6]] and [[144,12,12]] BB codes, and the
absorbed/resolved split (without the LP) of the [[288,12,18]] code.
"""

import time

import numpy as np
from qldpc import codes
from sympy.abc import x, y

from qecdm.dmatrix import design_matrix, pauli_errors
from qecdm.gordanLP import gordan, split_absorbed, straddles
from qecdm.gordanLP_large import gordan_large, pivot_rows, sparse_straddles, split_mod_p
from qecdm.gordanLP_symmetric import bb_translations, gordan_symmetric
from qecdm.stabilizer import label_to_stab, qldpc_stabilizers


def bb_code_analyse(orders, poly_a, poly_b, family=None, method="symmetric"):
    """orders, A, B -> dict: the thesis results for one BB code.

    The pipeline is BBCode -> qldpc_stabilizers -> design_matrix -> gordan.
    Every check is kept, not an independent subset: the subset would not be
    translation-invariant. `family` defaults to every two-qubit Pauli error.

    `method` is "symmetric" (gordanLP_symmetric: the LP reduced by the code's
    l*m translations, for BB72 and up), "large" (gordanLP_large: mod p and
    sparse, for small BB codes) or "exact" (gordanLP: Fractions, for tiny
    ones). All three give the same results.
    """
    times = {}

    start = time.perf_counter()
    code = codes.BBCode(orders, poly_a, poly_b)
    checks, qubits, _ = qldpc_stabilizers(code)
    times["stabilizers"] = time.perf_counter() - start

    start = time.perf_counter()
    D, merged, measured, subsets = design_matrix(checks)
    times["design_matrix"] = time.perf_counter() - start

    if family is None:
        family, _ = pauli_errors(len(qubits), 2)

    start = time.perf_counter()
    if method == "symmetric":
        result = gordan_symmetric(D, measured, bb_translations(code), family)
        certified, absorbed, resolved, witness, physical, unphysical = result
    elif method == "large":
        certified, absorbed, resolved, witness, physical, unphysical = gordan_large(D, measured, family)
    elif method == "exact":
        certified, absorbed, resolved, witness = gordan(D, measured, family)
        physical, unphysical, _ = split_absorbed(D, straddles(measured, family), absorbed)
    else:
        raise ValueError(f'method must be "symmetric", "large" or "exact", not {method!r}')
    times["gordan"] = time.perf_counter() - start

    return {
        "orders": code.orders,
        "qubits": len(qubits),
        "logicals": code.dimension,
        "checks": len(checks),
        "rows": D.shape[0],
        "classes": D.shape[1],                          # |C*|
        "kappa": D.shape[0] - np.linalg.matrix_rank(D),
        "max_order": max(len(subset) for subset in subsets),
        "family": len(family),
        "absorbed": len(absorbed),
        "resolved": len(resolved),
        "physical": len(physical),
        "unphysical": len(unphysical),
        "certified": certified,
        "times": times,
        # The full outputs, for looking at which errors and rows are involved.
        "code": code, "D": D, "merged": merged, "measured": measured, "subsets": subsets,
        "absorbed_indices": absorbed, "resolved_indices": resolved, "witness": witness,
    }


def anchored_errors(code):
    """BB code -> np.ndarray (2 * (n-1) * 9, 2, n): two-qubit errors with one qubit at a block's origin.

    Each is an ordered pair: letter s on an origin u (qubit 0 of L or of R),
    letter t on any other qubit v. Translating any ordered pair so its first
    qubit is at the origin gives one of these, and exactly l*m ordered pairs
    give each one.
    """
    n = code.num_qudits
    labels = [f"{s}{u}{t}{v}" if u < v else f"{t}{v}{s}{u}"  # labels list qubits in order
              for u in (0, n // 2) for v in range(n) if v != u for s in "XYZ" for t in "XYZ"]
    return np.array([label_to_stab(label, n) for label in labels])


def bb_code_split(orders, poly_a, poly_b, block=500):
    """orders, A, B -> dict: the absorbed/resolved split of every two-qubit error, without the LP.

    Being absorbed, and being physical, is the same for an error and all its
    translates, so only the anchored errors are classified. Each error is two
    ordered pairs and each anchored pair stands for l*m of them, so
    #absorbed = (l*m / 2) * #absorbed anchored errors, and likewise for physical
    and unphysical. For [[288,12,18]] that is 5,166 errors instead of 371,952.
    """
    times = {}

    start = time.perf_counter()
    code = codes.BBCode(orders, poly_a, poly_b)
    checks, qubits, _ = qldpc_stabilizers(code)
    times["stabilizers"] = time.perf_counter() - start

    start = time.perf_counter()
    D, merged, measured, subsets = design_matrix(checks)
    times["design_matrix"] = time.perf_counter() - start

    start = time.perf_counter()
    anchored = anchored_errors(code)
    S = sparse_straddles(measured, anchored)
    split = split_mod_p(D, S, pivot_rows(D), block=block)
    times["split"] = time.perf_counter() - start

    n, translations = len(qubits), len(qubits) // 2
    family = 9 * n * (n - 1) // 2
    absorbed, physical, unphysical = (translations * len(split[i]) // 2 for i in (0, 2, 3))
    return {
        "orders": code.orders,
        "qubits": n,
        "logicals": code.dimension,
        "checks": len(checks),
        "rows": D.shape[0],
        "classes": D.shape[1],                          # |C*|
        "kappa": D.shape[0] - D.shape[1],               # rank = |C*|: pivot_rows found them all
        "max_order": max(len(subset) for subset in subsets),
        "family": family,
        "anchored": len(anchored),
        "absorbed": absorbed,
        "resolved": family - absorbed,
        "physical": physical,
        "unphysical": unphysical,
        "times": times,
        "code": code, "D": D, "measured": measured, "anchored_errors": anchored,
        "anchored_split": split,
    }


if __name__ == "__main__":
    columns = ["orders", "qubits", "logicals", "checks", "rows", "classes", "kappa", "max_order",
               "family", "absorbed", "resolved", "physical", "unphysical", "certified"]
    print(" ".join(f"{c:>10}" for c in columns) + "   times (s)")
    for orders in ({x: 6, y: 6},    # [[72,12,6]]
                   {x: 12, y: 6}):  # [[144,12,12]], about 3 minutes
        r = bb_code_analyse(orders, x**3 + y + y**2, y**3 + x + x**2)
        times = ", ".join(f"{step} {t:.1f}" for step, t in r["times"].items())
        print(" ".join(f"{str(r[c]):>10}" for c in columns) + f"   {times}", flush=True)

    columns = [c for c in columns if c != "certified"] + ["anchored"]
    print("\nSplit only, no LP:")
    print(" ".join(f"{c:>10}" for c in columns) + "   times (s)")
    r = bb_code_split({x: 12, y: 12}, x**3 + y**2 + y**7, y**3 + x + x**2)  # [[288,12,18]], ~12 min
    times = ", ".join(f"{step} {t:.1f}" for step, t in r["times"].items())
    print(" ".join(f"{str(r[c]):>10}" for c in columns) + f"   {times}", flush=True)
