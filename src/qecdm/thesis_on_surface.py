"""Apply the thesis pipeline to stim's surface codes: D', absorbed/resolved, Gordan LP.

Run this file directly for the d = 3 and d = 5 rotated surface codes.
"""

import time

import numpy as np

from qecdm.dmatrix import design_matrix, pauli_errors
from qecdm.gordanLP import gordan, split_absorbed, straddles
from qecdm.stabilizer import independent_stabilizers, stim_stabilizers


def surface_code_analyse(distance, rounds=1, name="surface_code:rotated_memory_z", family=None):
    """int -> dict: the thesis results for one stim surface code.

    The pipeline is stim_stabilizers -> independent_stabilizers -> design_matrix
    -> gordan, with split_absorbed on the absorbed errors. `family` defaults to
    every two-qubit Pauli error. One round is enough for the surface code, since
    every round measures every check.
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

    # gordan takes the measured set (the rows of D'), not the generators.
    start = time.perf_counter()
    certified, absorbed, resolved, witness = gordan(D, measured, family)
    times["gordan"] = time.perf_counter() - start

    start = time.perf_counter()
    physical, unphysical, _ = split_absorbed(D, straddles(measured, family), absorbed)
    times["split_absorbed"] = time.perf_counter() - start

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
        r = surface_code_analyse(d)
        times = ", ".join(f"{step} {t:.1f}" for step, t in r["times"].items())
        print(" ".join(f"{str(r[c]):>10}" for c in columns) + f"   {times}", flush=True)
