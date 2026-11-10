"""Apply the thesis pipeline to stim's surface codes: D', absorbed/resolved, Gordan LP.

Run this file directly for the d = 3, 5 and 7 rotated surface codes.
"""

import time

import numpy as np

from qecdm.stabilizer.dmatrix import design_matrix, pauli_errors
from qecdm.stabilizer.gordanLP import gordan, split_absorbed, straddles
from qecdm.stabilizer.gordanLP_large import gordan_large
from qecdm.stabilizer.stabilizers import independent_stabilizers, stim_stabilizers


def surface_code_analyse(distance, rounds=1, name="surface_code:rotated_memory_z", family=None,
                         method="exact"):
    """int -> dict: the thesis results for one stim surface code.

    The pipeline is stim_stabilizers -> independent_stabilizers -> design_matrix
    -> gordan, with split_absorbed on the absorbed errors. `family` defaults to
    every two-qubit Pauli error. One round is enough for the surface code, since
    every round measures every check.

    `method` is "exact" (gordanLP: Fractions, simple, for small codes) or
    "large" (gordanLP_large: mod p and sparse, for d >= 5). Both give the same
    results.
    """
    times = {}

    start = time.perf_counter()
    generators, qubits, coords = stim_stabilizers(name, distance=distance, rounds=rounds)
    generators = independent_stabilizers(generators)  # already independent here; a safeguard
    times["stabilizers"] = time.perf_counter() - start

    start = time.perf_counter()
    D, merged, measured, subsets = design_matrix(generators)
    times["design_matrix"] = time.perf_counter() - start

    if family is None:
        family, _ = pauli_errors(len(qubits), 2)

    # Both methods take the measured set (the rows of D'), not the generators.
    start = time.perf_counter()
    if method == "exact":
        certified, absorbed, resolved, witness = gordan(D, measured, family)
        physical, unphysical, _ = split_absorbed(D, straddles(measured, family), absorbed)
    elif method == "large":
        certified, absorbed, resolved, witness, physical, unphysical = gordan_large(D, measured, family)
    else:
        raise ValueError(f'method must be "exact" or "large", not {method!r}')
    times["gordan"] = time.perf_counter() - start

    return {
        "distance": distance,
        "qubits": len(qubits),
        "generators": len(generators),
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
        "D": D, "merged": merged, "measured": measured, "subsets": subsets,
        "absorbed_indices": absorbed, "resolved_indices": resolved,
        "witness": witness, "data_qubits": qubits, "coords": coords,
    }


if __name__ == "__main__":
    columns = ["distance", "qubits", "generators", "rows", "classes", "kappa", "max_order",
               "family", "absorbed", "resolved", "physical", "unphysical", "certified"]
    print(" ".join(f"{c:>10}" for c in columns) + "   times (s)")
    for d in (3, 5, 7):
        r = surface_code_analyse(d, method="large")
        times = ", ".join(f"{step} {t:.1f}" for step, t in r["times"].items())
        print(" ".join(f"{str(r[c]):>10}" for c in columns) + f"   {times}", flush=True)
