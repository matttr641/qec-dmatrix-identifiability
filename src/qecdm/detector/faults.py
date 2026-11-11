"""Every single fault in a noisy circuit: H (detectors x faults) and a record for each fault.

Each column of H is one circuit fault location (one Pauli of one noise channel
on one gate, or one measurement flip), kept separate: nothing is merged or
tied here. The records hold what tying needs: the fault's channel, qubits,
Pauli, round, and position within its round.
"""

from bisect import bisect_left
from collections import namedtuple

import numpy as np
import stim
from scipy import sparse

Fault = namedtuple("Fault", "channel qubits pauli round position offset label")

CHANNELS = ("idle", "gate_1q", "gate_2q", "measure", "reset", "prep", "readout")


def _qubits(ins):
    """stim.CircuitInstruction -> list of int: the qubits it acts on."""
    return [t.value for t in ins.targets_copy() if t.is_qubit_target]


def _is_noise(ins):
    """A noise channel (X_ERROR, DEPOLARIZE1, ...), not a gate or a noisy measurement like M(p)."""
    data = stim.gate_data(ins.name)
    return data.is_noisy_gate and not data.produces_measurements


def qubit_roles(circuit):
    """stim.Circuit -> dict int -> "data" | "ancilla".

    The data qubits are those of the final measurement (the readout); every
    other measured qubit is an ancilla. Qubits never measured are left out.
    """
    measurements = [ins for ins in circuit if stim.gate_data(ins.name).produces_measurements]
    roles = {q: "ancilla" for ins in measurements[:-1] for q in _qubits(ins)}
    roles.update({q: "data" for q in _qubits(measurements[-1])})
    return roles


def _neighbour(circuit, offset, qubit, step):
    """The index of the nearest gate on `qubit` before (step=-1) or after (step=+1) `offset`.

    Only the same moment is searched: noise channels, annotations (DETECTOR,
    ...) and gates on other qubits are skipped, and a TICK ends the search,
    giving None.
    """
    i = offset + step
    while 0 <= i < len(circuit):
        ins = circuit[i]
        if ins.name == "TICK":
            return None
        if not _is_noise(ins) and qubit in _qubits(ins):
            return i
        i += step
    return None


def fault_channel(circuit, offset, qubits, roles, readout_offset):
    """The channel of a fault on `qubits` made by the instruction at `offset`.

    - a measurement's own flip, M(p): readout if it is the final measurement, else measure;
    - DEPOLARIZE2: gate_2q;
    - DEPOLARIZE1: gate_1q straight after a one-qubit Clifford on that qubit, else idle;
    - X/Y/Z_ERROR straight after a reset: prep on a data qubit, reset on an ancilla;
    - X/Y/Z_ERROR straight before a measurement: readout or measure, as for M(p).
    """
    ins = circuit[offset]
    if stim.gate_data(ins.name).produces_measurements:
        return "readout" if offset == readout_offset else "measure"
    if ins.name == "DEPOLARIZE2":
        return "gate_2q"
    before = _neighbour(circuit, offset, qubits[0], -1)
    before = None if before is None else stim.gate_data(circuit[before].name)
    if ins.name == "DEPOLARIZE1":
        one_qubit_gate = before is not None and before.is_unitary and before.is_single_qubit_gate
        return "gate_1q" if one_qubit_gate else "idle"
    if ins.name in ("X_ERROR", "Y_ERROR", "Z_ERROR"):
        if before is not None and before.is_reset:
            return "prep" if roles[qubits[0]] == "data" else "reset"
        after = _neighbour(circuit, offset, qubits[0], +1)
        if after is not None and stim.gate_data(circuit[after].name).produces_measurements:
            return "readout" if after == readout_offset else "measure"
    raise ValueError(f"cannot classify the fault from instruction {offset}: {ins}")


def _moments(circuit):
    """stim.Circuit -> list: the moment (number of TICKs before it) of each instruction."""
    moments, tick = [], 0
    for ins in circuit:
        moments.append(tick)
        tick += ins.name == "TICK"
    return moments


def measurement_moments(circuit, roles):
    """The moments that measure ancillas: each one ends a round."""
    moments = _moments(circuit)
    return sorted({moments[i] for i, ins in enumerate(circuit)
                   if stim.gate_data(ins.name).produces_measurements
                   and any(roles.get(q) == "ancilla" for q in _qubits(ins))})


def fault_round(moment, ends):
    """int, list -> (round, position): which round a moment is in, and where in it.

    Round t ends at the t-th ancilla-measurement moment ends[t-1]. The position
    counts moments back from that end, so the measurement moment is 0 and the
    one before it -1: rounds line up from their measurements back, even when the
    first round starts differently. Moments after the last measurement (the
    readout) are round len(ends) + 1, with positions counted forward from 1.
    """
    t = bisect_left(ends, moment)  # the number of rounds ended before this moment
    if t == len(ends):
        return t + 1, moment - ends[-1]
    return t + 1, moment - ends[t]


def pauli_label(location):
    """stim.CircuitErrorLocation -> str: e.g. "X0*Z1", or "flip" for a measurement's own flip."""
    letters = []
    for target in location.flipped_pauli_product:
        g = target.gate_target
        letters.append(("X" if g.is_x_target else "Y" if g.is_y_target else "Z") + str(g.value))
    return "*".join(letters) or "flip"


def circuit_faults(circuit):
    """stim.Circuit -> (H, faults): H (detectors x faults) of 0/1, and one Fault per column.

    Every circuit fault that trips at least one detector is one column, so the
    same detector pattern can appear in many columns. Faults that trip no
    detector (even if they flip a logical observable) are left out.
    """
    circuit = circuit.flattened()
    roles = qubit_roles(circuit)
    moments = _moments(circuit)
    ends = measurement_moments(circuit, roles)
    readout_offset = max(i for i, ins in enumerate(circuit)
                         if stim.gate_data(ins.name).produces_measurements)

    rows, faults = [], []
    for error in circuit.explain_detector_error_model_errors(reduce_to_one_representative_error=False):
        detectors = [t.dem_target.val for t in error.dem_error_terms if t.dem_target.is_relative_detector_id()]
        if not detectors:
            continue
        for location in error.circuit_error_locations:
            offset = location.stack_frames[0].instruction_offset
            qubits = tuple(t.gate_target.value for t in location.instruction_targets.targets_in_range)
            channel = fault_channel(circuit, offset, qubits, roles, readout_offset)
            round_, position = fault_round(moments[offset], ends)
            pauli = pauli_label(location)
            rows.append(detectors)
            faults.append(Fault(channel, qubits, pauli, round_, position, offset,
                                f"{pauli} {channel} r{round_}"))

    cols = [j for j, detectors in enumerate(rows) for _ in detectors]
    H = sparse.csc_matrix((np.ones(len(cols), dtype=np.uint8), ([d for r in rows for d in r], cols)),
                          shape=(circuit.num_detectors, len(faults)))
    return H, faults


def check_round_structure(circuit):
    """stim.Circuit -> None: raise unless rounds 2 onwards have the same noise template.

    The template of a round is its noise instructions, each as (position,
    name, targets, arguments), up to and including its measurement moment.
    Round 1 may start differently (it holds the preparation), and the readout
    is not part of any round (stim puts it in the last round's measurement
    moment), so neither is compared. Tying a fault with "the same fault in
    another round" relies on this.
    """
    circuit = circuit.flattened()
    roles = qubit_roles(circuit)
    moments = _moments(circuit)
    ends = measurement_moments(circuit, roles)
    readout_offset = max(i for i, ins in enumerate(circuit)
                         if stim.gate_data(ins.name).produces_measurements)
    templates = {}
    for i, ins in enumerate(circuit):
        noisy = _is_noise(ins) or (stim.gate_data(ins.name).produces_measurements and ins.gate_args_copy())
        readout = i == readout_offset or (_is_noise(ins) and
                                          _neighbour(circuit, i, _qubits(ins)[0], +1) == readout_offset)
        round_, position = fault_round(moments[i], ends)
        if noisy and not readout and 2 <= round_ <= len(ends) and position <= 0:
            templates.setdefault(round_, []).append(
                (position, ins.name, tuple(_qubits(ins)), tuple(ins.gate_args_copy())))
    rounds = sorted(templates)
    for t in rounds[1:]:
        if sorted(templates[t]) != sorted(templates[rounds[0]]):
            raise ValueError(f"round {t} has a different noise template from round {rounds[0]}")
