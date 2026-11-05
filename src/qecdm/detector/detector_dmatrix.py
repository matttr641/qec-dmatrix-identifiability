"""The detector D': rows are parities of connected sets of detectors, columns are parameters.

The row of a set R of detectors is <(-1)^(XOR of R)>, and a fault flips that
parity when it trips an odd number of R's detectors: the entry is
|R & pattern| mod 2. That is D'_raw = R H mod 2, one column per fault. Faults
with the same pattern have the same column, so the rows are built on the
distinct patterns (D_sig) and tied with the counts: D'_tied = D_sig M. The
mod 2 is taken per fault, before tying, so two tied faults that both flip a
row give 2, not 0.

Rows are walked order by order over the connected subsets of the detector
graph, until rank(D'_tied) reaches rank(M), the most any set of rows can give.
"""

from collections import namedtuple

import numpy as np
from scipy import sparse

from qecdm.detector.faults import circuit_faults
from qecdm.detector.tying import identifiability, merge_parameters, signature_counts, tying_matrix
from qecdm.dmatrix import connected_subsets

DetectorDesign = namedtuple("DetectorDesign", "D groups D_fixed fixed_names subsets target directions")


def detector_graph(H):
    """H (detectors x faults) -> np.ndarray (detectors x detectors) of bools.

    Detectors i and j are joined when some fault trips both: (H H^T)[i, j] > 0.
    """
    H = sparse.csr_matrix(H, dtype=np.int64)
    return (H @ H.T).toarray() > 0


def signature_matrix(signatures, num_detectors):
    """list of tuples, int -> np.ndarray (detectors x signatures): the 0/1 matrix of the patterns."""
    H_sig = np.zeros((num_detectors, len(signatures)), dtype=np.int64)
    for j, signature in enumerate(signatures):
        H_sig[list(signature), j] = 1
    return H_sig


def subset_rows(subsets, H_sig):
    """list of tuples, H_sig -> np.ndarray (subsets x signatures): |R & pattern| mod 2."""
    R = np.zeros((len(subsets), H_sig.shape[0]), dtype=np.int64)
    for i, subset in enumerate(subsets):
        R[i, list(subset)] = 1
    return R @ H_sig % 2


def detector_design_matrix(circuit, rule="rounds", fixed=("prep",), max_order=None):
    """stim.Circuit -> DetectorDesign: D'_tied, by walking connected detector subsets.

    - D: (rows x merged parameters) of ints; groups[k] are the names merged into column k;
    - D_fixed: (rows x fixed parameters), the calibrated part, named by fixed_names;
    - subsets: the detectors of each row; target: rank(M); directions: what cannot be learned.

    Stops when rank(D) reaches the target, or after exactly `max_order` orders if
    given (D may then fall short). Raises if the target is never reached.
    """
    H, faults = circuit_faults(circuit)
    return design_from_faults(H, faults, rule, fixed, max_order)


def design_from_faults(H, faults, rule="rounds", fixed=("prep",), max_order=None):
    """H (detectors x faults), list of Fault -> DetectorDesign, as detector_design_matrix.

    For any H and its faults, such as those of a window of rounds; subsets
    number the detectors by the rows of H.
    """
    T, names, T_fixed, fixed_names = tying_matrix(faults, rule, fixed)
    M, signatures = signature_counts(H, T)
    M_fixed, _ = signature_counts(H, T_fixed)  # the same signatures: they depend on H only
    M_merged, groups = merge_parameters(M, names)
    target, directions = identifiability(M_merged, groups)
    H_sig = signature_matrix(signatures, H.shape[0])

    levels = connected_subsets(detector_graph(H))
    rows, subsets, seen = [], [], set()
    order, rank = 0, 0
    while order < H.shape[0] and (rank < target if max_order is None else order < max_order):
        order += 1
        level = next(levels)
        for subset, row in zip(level, subset_rows(level, H_sig)):
            if not row.any() or row.tobytes() in seen:  # sees no fault, or repeats a row
                continue
            seen.add(row.tobytes())
            rows.append(row)
            subsets.append(subset)
        rank = np.linalg.matrix_rank(np.array(rows) @ M_merged)
    if max_order is None and rank < target:
        raise RuntimeError(f"rank {rank} < target {target} at every order")

    D_sig = np.array(rows)
    return DetectorDesign(D_sig @ M_merged, groups, D_sig @ M_fixed, fixed_names, subsets, target, directions)
