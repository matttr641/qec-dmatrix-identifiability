"""Which extra errors a window D' absorbs: errors outside the model that the fit can
mistake for a change in the model's own rates.

An extra error is absorbed when its straddle (the rows whose parity it flips)
lies in im(D'), and resolved otherwise, as in the stabilizer version.
"""

from collections import namedtuple
from itertools import combinations, product

import numpy as np
import stim

from qecdm.detector.detector_dmatrix import signature_matrix, subset_rows
from qecdm.detector.faults import _is_noise, _qubits, fault_round, measurement_moments, qubit_roles
from qecdm.detector.windows import detector_rounds, shift_subsets, window_design, window_detectors
from qecdm.stabilizer.gordanLP_large import pivot_rows, split_mod_p

Crosstalk = namedtuple("Crosstalk", "qubits pauli round position offset label")
Absorption = namedtuple("Absorption", "absorbed resolved invisible")

#The family: two-qubit Paulis on pairs of data qubits, once per stretch both sit idle

def idle_crosstalk(circuit):
    """stim.Circuit -> list of Crosstalk: every weight-2 Pauli, once per shared idle interval of a data pair.

    A data qubit is idle in a moment when no gate, measurement or reset acts on
    it there. A pair's shared idle interval is a run of consecutive moments in
    which both are idle; anywhere inside it a crosstalk has the same effect,
    so it gets one of each of the 9 Paulis (X, Y or Z on each), at the end of
    the interval: offset is just before the TICK of its last moment. round and
    position are those of that moment, as for a Fault, so (position, qubits,
    pauli) is the same crosstalk in every round.
    """
    circuit = circuit.flattened()
    roles = qubit_roles(circuit)
    data = sorted(q for q, role in roles.items() if role == "data")
    ends = measurement_moments(circuit, roles)
    errors, busy, moment = [], set(), 0
    last = {}  # pair -> (round, position, offset, moment) of the latest moment in its open idle interval

    def close(pair):
        """The pair's interval has ended: one crosstalk of each Pauli, at its last moment."""
        round_, position, offset, at = last.pop(pair)
        for p1, p2 in product("XYZ", repeat=2):
            label = f"{p1}{pair[0]}{p2}{pair[1]} r{round_} m{at}"
            errors.append(Crosstalk(pair, p1 + p2, round_, position, offset, label))

    for i, ins in enumerate(circuit):
        if ins.name != "TICK":
            if not _is_noise(ins):
                busy.update(_qubits(ins))  # a gate, measurement or reset acts on these
            continue
        idle_pairs = set(combinations([q for q in data if q not in busy], 2))
        for pair in sorted(set(last) - idle_pairs):
            close(pair)                    # one of the pair was busy in this moment
        for pair in idle_pairs:
            last[pair] = (*fault_round(moment, ends), i, moment)
        busy, moment = set(), moment + 1
    for pair in sorted(last):
        close(pair)                        # intervals still open when the circuit ends
    return errors

#The detectors each family error trips, from stim

def error_patterns(circuit, errors):
    """stim.Circuit, list of Crosstalk -> list of tuples: the detectors each error trips.

    Every error is put in a noiseless copy of the circuit as its own E
    instruction at its offset, and stim explains which detectors each one
    flips, as circuit_faults does for the model's faults. An error that trips
    no detector gets ().
    """
    circuit = circuit.flattened()
    errors_at = {}                            # circuit position -> the errors that go there
    for k, error in enumerate(errors):
        errors_at.setdefault(error.offset, []).append(k)

    noiseless, error_of = stim.Circuit(), {}  # error_of: position of an E in the copy -> its error
    for i, ins in enumerate(circuit):
        for k in errors_at.get(i, []):
            error_of[len(noiseless)] = k
            noiseless.append("E", [stim.target_pauli(q, p) for q, p in zip(errors[k].qubits, errors[k].pauli)], 0.01)
        if stim.gate_data(ins.name).produces_measurements:
            noiseless.append(ins.name, ins.targets_copy())  # drop a measurement's flip rate, as in M(p)
        elif not _is_noise(ins):
            noiseless.append(ins)

    patterns = [()] * len(errors)
    for explained in noiseless.explain_detector_error_model_errors(reduce_to_one_representative_error=False):
        detectors = tuple(sorted(t.dem_target.val for t in explained.dem_error_terms
                                 if t.dem_target.is_relative_detector_id()))
        for location in explained.circuit_error_locations:
            patterns[error_of[location.stack_frames[0].instruction_offset]] = detectors
    return patterns

#Each crosstalk's straddle over a window's rows, tied across rounds

def window_straddles(design, rounds, start, size, errors, patterns):
    """Window DetectorDesign, detector_rounds list, window, errors and their patterns -> (S, keys).

    The straddle of one error is |row & pattern| mod 2 for each row of the
    window at `start`, as for the model's faults. Copies of the same crosstalk
    in different rounds share a key (position, qubits, pauli) and their
    straddles are added, as tying adds the model's faults: S[:, j] is the
    column of keys[j]. Errors that trip none of the window's detectors are left out.
    """
    window = set(window_detectors(rounds, start, size))
    kept = [k for k, pattern in enumerate(patterns) if window.intersection(pattern)]
    rows = shift_subsets(design.subsets, rounds, start, size)
    parities = subset_rows(rows, signature_matrix([patterns[k] for k in kept], len(rounds)))  # rows x kept

    columns = {}                                  # key -> the positions in `kept` of its copies
    for i, k in enumerate(kept):
        columns.setdefault((errors[k].position, errors[k].qubits, errors[k].pauli), []).append(i)
    S = np.column_stack([parities[:, copies].sum(axis=1) for copies in columns.values()])
    return S, list(columns)

#Which crosstalk the window D' absorbs

def absorbed_crosstalk(circuit, rule="rounds", start=3, size=2):
    """stim.Circuit -> Absorption: each idle crosstalk, tied across rounds, sorted by what the fit does with it.

    Builds the window D' at `start` and the crosstalk straddles there, then
    split_mod_p decides for each one whether its straddle lies in im(D'):
    - absorbed: in im(D'), so the fit takes it as a change in the model's rates;
    - resolved: not in im(D'), so the fit would show it as a misfit;
    - invisible: in the window's rounds, but trips no detector at all.
    Each list holds keys (position, qubits, pauli). The window D' is the same
    at every position, so the answer holds for every window. D' needs full
    column rank ("rounds" or "channels"; "gates" raises).
    """
    design, _ = window_design(circuit, start, size, rule)
    errors = idle_crosstalk(circuit)
    patterns = error_patterns(circuit, errors)
    S, keys = window_straddles(design, detector_rounds(circuit), start, size, errors, patterns)
    absorbed, resolved, _, _ = split_mod_p(design.D, S, pivot_rows(design.D))
    seen = {(e.position, e.qubits, e.pauli) for e, pattern in zip(errors, patterns) if pattern}
    here = {(e.position, e.qubits, e.pauli) for e in errors if start <= e.round < start + size}  # not the run's ends
    invisible = sorted(here - seen)
    return Absorption([keys[j] for j in absorbed], [keys[j] for j in resolved], invisible)
