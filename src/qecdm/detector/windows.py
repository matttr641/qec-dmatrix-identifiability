"""Sliding windows: estimating the noise rates from a few rounds at a time."""

from bisect import bisect_right

import numpy as np
import stim

from qecdm.detector.detector_dmatrix import design_from_faults
from qecdm.detector.faults import _moments, circuit_faults, measurement_moments, qubit_roles

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

def window_detectors(circuit, start, size):
    """stim.Circuit, first round, number of rounds -> list: the circuit's detectors in that window, in order."""
    detectors = [d for d, r in enumerate(detector_rounds(circuit)) if r is not None and start <= r < start + size]
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
    detectors = window_detectors(circuit, start, size)
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

def shift_subsets(subsets, circuit, start, size):
    """A window design's subsets, stim.Circuit, first round, number of rounds -> circuit detector tuples.

    Window detector i becomes the i-th circuit detector of the window at
    `start`, so the same D' describes the rows there. This relies on every
    window listing its detectors in the same roles and order, which holds
    away from the first round.
    """
    detectors = window_detectors(circuit, start, size)
    if max(i for subset in subsets for i in subset) >= len(detectors):
        raise ValueError(f"rounds {start} to {start + size - 1} have only {len(detectors)} detectors")
    return [tuple(detectors[i] for i in subset) for subset in subsets]
