"""Sliding windows: estimating the noise rates from a few rounds at a time."""

from bisect import bisect_right

import numpy as np
import stim

from qecdm.detector.detector_dmatrix import design_from_faults
from qecdm.detector.estimate import (NOISE_KEY, fit_thetas, rates_from_thetas, row_expectations,
                                     sample_detectors)
from qecdm.detector.faults import (_is_noise, _moments, circuit_faults, fault_channel, fault_round,
                                   measurement_moments, qubit_roles)

#Finds the rounds the detectors fall in, None if readout detector

def detector_rounds(circuit):
    """stim.Circuit -> list: the round of each detector, or None if it uses the readout.

    A detector belongs to the last round that ended before it. Detectors after
    the final measurement compare the readout with the last round, so they are
    not part of any window.
    """
    circuit = circuit.flattened()
    moments = _moments(circuit)
    ends = measurement_moments(circuit, qubit_roles(circuit))
    readout = max(i for i, ins in enumerate(circuit) if stim.gate_data(ins.name).produces_measurements)
    return [None if i > readout else bisect_right(ends, moments[i])
            for i, ins in enumerate(circuit) if ins.name == "DETECTOR"]

def window_detectors(rounds, start, size):
    """detector_rounds list, first round, number of rounds -> list: the circuit's detectors in that window, in order."""
    detectors = [d for d, r in enumerate(rounds) if r is not None and start <= r < start + size]
    if not detectors:
        raise ValueError(f"no detectors in rounds {start} to {start + size - 1}")
    return detectors

#Cuts H down to just the window detectors and any error that trips one of them, can have edge errors outside window

def window_faults(circuit, start, size):
    """stim.Circuit, first round, number of rounds -> (H, faults, detectors) for that window.

    detectors are the circuit's detectors in rounds start to start + size - 1;
    H keeps their rows, and only the faults (columns) that trip at least one of
    them. Row i of H is circuit detector detectors[i].
    """
    detectors = window_detectors(detector_rounds(circuit), start, size)
    H, faults = circuit_faults(circuit)
    H = H[detectors]                             # the window's rows
    keep = np.flatnonzero(H.getnnz(axis=0))      # faults that trip one of them
    return H[:, keep], [faults[j] for j in keep], detectors

#Builds a windows D'

def window_design(circuit, start, size, rule="rounds", max_order=None):
    """stim.Circuit, first round, number of rounds -> (DetectorDesign, detectors) for that window.

    The design is built from the window's faults alone, with nothing fixed: a
    window away from the ends has no prep or readout faults, and raises if it
    does. design.subsets number the window's detectors; row i is circuit
    detector detectors[i].
    """
    H, faults, detectors = window_faults(circuit, start, size)
    ends = {f.channel for f in faults} & {"prep", "readout"}
    if ends:
        raise ValueError(f"rounds {start} to {start + size - 1} see {sorted(ends)}: move the window away from the ends")
    return design_from_faults(H, faults, rule, fixed=(), max_order=max_order), detectors

#Maps row number (detector) of H back to the actual detector number in circuit
#Needed for sampling from the circuit

def shift_subsets(subsets, rounds, start, size):
    """A window design's subsets, detector_rounds list, first round, number of rounds -> circuit detector tuples.

    Window detector i becomes the i-th circuit detector of the window at
    `start`, so the same D' describes the rows there. This relies on every
    window listing its detectors in the same roles and order, which holds
    away from the first round.
    """
    detectors = window_detectors(rounds, start, size)
    if max(i for subset in subsets for i in subset) >= len(detectors):
        raise ValueError(f"rounds {start} to {start + size - 1} have only {len(detectors)} detectors")
    return [tuple(detectors[i] for i in subset) for subset in subsets]

#A circuit whose noise rates change from round to round

def drifting_circuit(circuit, noise_at):
    """stim.Circuit, function round -> noise dict -> stim.Circuit with rates that change by round.

    Every noisy instruction keeps its place and targets; its rate becomes
    noise_at(its round)[its channel], with the channel found as in faults.py
    (prep takes the reset rate). An instruction is split by target, so its
    qubits can belong to different channels; stim joins the pieces again
    when they end up with the same rate.
    """
    circuit = circuit.flattened()
    roles = qubit_roles(circuit)
    moments = _moments(circuit)
    ends = measurement_moments(circuit, roles)
    readout = max(i for i, ins in enumerate(circuit) if stim.gate_data(ins.name).produces_measurements)
    out = stim.Circuit()
    for i, ins in enumerate(circuit):
        noisy = _is_noise(ins) or (stim.gate_data(ins.name).produces_measurements and ins.gate_args_copy())
        if not noisy:
            out.append(ins)
            continue
        noise = noise_at(fault_round(moments[i], ends)[0])
        targets = ins.targets_copy()
        width = 2 if ins.name == "DEPOLARIZE2" else 1  # a two-qubit channel acts on pairs
        for k in range(0, len(targets), width):
            group = targets[k:k + width]
            channel = fault_channel(circuit, i, tuple(t.value for t in group), roles, readout)
            out.append(ins.name, group, noise[NOISE_KEY.get(channel, channel)])
    return out

#Fits every window position of a (drifting) circuit with the one window D'

def track_rates(design, circuit, size, shots, stride=1, seed=None):
    """Window DetectorDesign, stim.Circuit, window size, shots -> (starts, thetas, rates).

    Samples the circuit, then fits each window from round 2 to the last that
    fits, moving `stride` rounds at a time; stride = size + 1 leaves a round
    between windows, so no fault reaches two of them and they are
    independent. thetas[i] and rates[i] are the window starting at starts[i],
    one entry per column of design.D.
    """
    rounds = detector_rounds(circuit)
    last = max(r for r in rounds if r is not None)
    starts = list(range(2, last - size + 2, stride))
    samples = sample_detectors(circuit, shots, seed)
    thetas = []
    for start in starts:
        y = row_expectations(samples, shift_subsets(design.subsets, rounds, start, size))
        thetas.append(fit_thetas(design, y, np.zeros(0)))  # nothing is fixed in a window
    thetas = np.array(thetas)
    return starts, thetas, np.array([rates_from_thetas(design, theta) for theta in thetas])

#One long run: average the rows over many windows, then fit once

def long_run_rates(design, circuit, size, last=None, stride=1, shots=1, seed=None):
    """Window DetectorDesign, stim.Circuit, window size -> (theta, rates) from one long run.

    Each window position is another sample of the same rows, so y is averaged
    over the windows (and shots) before one fit. last=N uses only the last N
    windows, to follow drift; None uses every window from round 2 on.
    stride = size + 1 makes the windows independent.
    """
    rounds = detector_rounds(circuit)
    end = max(r for r in rounds if r is not None)
    starts = list(range(2, end - size + 2, stride))[-last if last else 0:]
    samples = sample_detectors(circuit, shots, seed)
    y = np.mean([row_expectations(samples, shift_subsets(design.subsets, rounds, start, size)) for start in starts], axis=0)
    theta = fit_thetas(design, y, np.zeros(0))  # nothing is fixed in a window
    return theta, rates_from_thetas(design, theta)
