"""Apply the absorption test to idle crosstalk on the codes of the stabilizer version.

Run this file directly for the repetition, Steane, rotated surface and [[72,12,6]]
BB codes, under each tying rule: how much two-qubit data crosstalk the window D'
absorbs, resolves, or never sees.
"""

import time

import numpy as np
from qldpc import codes
from sympy.abc import x, y

from qecdm.detector.absorption import error_patterns, idle_crosstalk, window_straddles
from qecdm.detector.circuits import bb_memory, qldpc_memory, repetition_memory, surface_memory
from qecdm.detector.faults import qubit_roles
from qecdm.detector.windows import detector_rounds, window_design
from qecdm.stabilizer.gordanLP_large import pivot_rows, split_mod_p

NOISE = {"idle": 0.002, "gate_1q": 0.005, "gate_2q": 0.005, "measure": 0.01, "reset": 0.003, "readout": 0.02}
ROUNDS, START, SIZE = 9, 3, 2
RULES = ("rounds", "gates", "channels")

CODES = {
    "rep d=3": lambda: repetition_memory(3, ROUNDS, NOISE),
    "steane": lambda: qldpc_memory(codes.SteaneCode(), ROUNDS, NOISE),
    "surface d=3": lambda: surface_memory(3, ROUNDS, NOISE),
    "surface d=5": lambda: surface_memory(5, ROUNDS, NOISE),
    "bb72": lambda: bb_memory({x: 6, y: 6}, x**3 + y + y**2, y**3 + x + x**2, ROUNDS, NOISE),
}


def crosstalk_analyse(circuit, rules=RULES, start=START, size=SIZE):
    """stim.Circuit -> list of dicts, one per rule: the absorption results for its window D'.

    The same steps as absorbed_crosstalk, with the family and its patterns built
    once for all rules, since neither depends on the tying. A rule whose D' is
    rank deficient ("gates", usually) cannot be split by split_mod_p and gets
    None for absorbed and resolved.
    """
    t0 = time.perf_counter()
    roles = qubit_roles(circuit)
    errors = idle_crosstalk(circuit)
    patterns = error_patterns(circuit, errors)
    rounds = detector_rounds(circuit)
    family_time = time.perf_counter() - t0

    seen = {(e.position, e.qubits, e.pauli) for e, pattern in zip(errors, patterns) if pattern}
    here = {(e.position, e.qubits, e.pauli) for e in errors if start <= e.round < start + size}

    results = []
    for rule in rules:
        t0 = time.perf_counter()
        design, detectors = window_design(circuit, start, size, rule)
        S, keys = window_straddles(design, rounds, start, size, errors, patterns)
        try:
            absorbed, resolved, _, _ = split_mod_p(design.D, S, pivot_rows(design.D))
            absorbed, resolved = len(absorbed), len(resolved)
        except ValueError:  # D' is rank deficient: some directions cannot be learned
            absorbed = resolved = None
        results.append({
            "rule": rule,
            "data": sum(role == "data" for role in roles.values()),
            "detectors": len(detectors),                 # in the window
            "rows": design.D.shape[0],
            "params": design.D.shape[1],                 # after tying and merging
            "rank": int(np.linalg.matrix_rank(design.D)),
            "unlearnable": len(design.directions),
            "crosstalk": len(here),                      # tied keys in the window's rounds
            "absorbed": absorbed,
            "resolved": resolved,
            "invisible": len(here - seen),
            "time": family_time + time.perf_counter() - t0,
        })
    return results


if __name__ == "__main__":
    columns = ["rule", "data", "detectors", "rows", "params", "rank", "unlearnable",
               "crosstalk", "absorbed", "resolved", "invisible"]
    print(f"{'code':>12} " + " ".join(f"{c:>11}" for c in columns) + "   time (s)")
    for name, build in CODES.items():
        for r in crosstalk_analyse(build()):
            values = " ".join(f"{'-' if r[c] is None else str(r[c]):>11}" for c in columns)
            print(f"{name:>12} {values}   {r['time']:.1f}", flush=True)


#the qldpc codes only have Z-check stabilizers as its a Z-memory test
#in future add the X-check stabilizers as well