"""Tests for the tying matrix: every fault lands in exactly one parameter, the rules give
the parameters worked out by hand, and fixed channels go to T_fixed."""

from collections import Counter

import numpy as np
import pytest
from sympy.abc import x, y

from qecdm.detector.circuits import bb_memory, repetition_memory, surface_memory, uniform_noise
from qecdm.detector.faults import circuit_faults
from qecdm.detector.tying import tying_matrix

BB = ({x: 2, y: 3}, 1 + y + x * y**2, 1 + y + x * y**2)  # [[12,4]], the small certified BB code
NOISE = uniform_noise(0.01)

CIRCUITS = {
    "rep_r2": lambda: repetition_memory(3, 2, NOISE),
    "rep_r5": lambda: repetition_memory(3, 5, NOISE),
    "surface_r3": lambda: surface_memory(3, 3, NOISE),
    "bb_r3": lambda: bb_memory(*BB, 3, NOISE),
}


@pytest.fixture(scope="module")
def faults():
    """name -> list of Fault, each built once and shared by every test."""
    return {name: circuit_faults(build())[1] for name, build in CIRCUITS.items()}


def members(T):
    """T -> list of lists: the faults (row indices) in each parameter (column)."""
    return [list(T[:, k].indices) for k in range(T.shape[1])]


# --- every fault exactly once ------------------------------------------------------

@pytest.mark.parametrize("name", list(CIRCUITS))
@pytest.mark.parametrize("rule", ["rounds", "channels"])
def test_every_fault_in_exactly_one_parameter(faults, name, rule):
    """Side by side, T and T_fixed have one 1 in every row: no fault lost or counted twice."""
    T, names, T_fixed, fixed_names = tying_matrix(faults[name], rule)
    both = np.hstack([T.toarray(), T_fixed.toarray()])
    assert both.shape == (len(faults[name]), len(names) + len(fixed_names))
    assert set(both.ravel()) == {0, 1}
    assert all(both.sum(axis=1) == 1)


@pytest.mark.parametrize("rule", ["rounds", "channels"])
def test_fixed_faults_are_only_in_T_fixed(faults, rule):
    """The prep faults have zero rows in T, and T_fixed holds nothing else."""
    fault_list = faults["rep_r2"]
    T, _, T_fixed, _ = tying_matrix(fault_list, rule)
    prep = [j for j, f in enumerate(fault_list) if f.channel == "prep"]
    assert len(prep) == 3
    assert T[prep].nnz == 0
    assert sorted(T_fixed.tocoo().row) == prep


# --- the parameter counts ------------------------------------------------------------

@pytest.mark.parametrize("name, rounds, rounds_fixed, channels, channels_fixed", [
    ("rep_r2", 61, 3, 5, 1),
    ("rep_r5", 61, 3, 5, 1),        # more rounds, the same parameters
    ("surface_r3", 412, 9, 6, 1),   # gate_1q from the H layers
    ("bb_r3", 1152, 12, 5, 1),
])
def test_parameter_counts(faults, name, rounds, rounds_fixed, channels, channels_fixed):
    T, _, T_fixed, _ = tying_matrix(faults[name], "rounds")
    assert (T.shape[1], T_fixed.shape[1]) == (rounds, rounds_fixed)
    T, _, T_fixed, _ = tying_matrix(faults[name], "channels")
    assert (T.shape[1], T_fixed.shape[1]) == (channels, channels_fixed)


def test_repetition_rounds_parameters_by_channel(faults):
    """The 61 by hand: idle 3 data qubits x X, Y (Z is undetected in a Z memory), CNOT 4 gates
    x 12 detectable Paulis, 1 measure and 1 reset rate per ancilla, 1 readout per data qubit."""
    _, names, _, fixed_names = tying_matrix(faults["rep_r2"], "rounds")
    assert Counter(key[0] for key in names) == {
        "idle": 6, "gate_2q": 48, "measure": 2, "reset": 2, "readout": 3}
    assert Counter(key[0] for key in fixed_names) == {"prep": 3}


def test_channels_names(faults):
    _, names, _, fixed_names = tying_matrix(faults["rep_r2"], "channels")
    assert set(names) == {"idle", "gate_2q", "measure", "reset", "readout"}
    assert fixed_names == ["prep"]


# --- what the rounds rule ties ------------------------------------------------------

def test_rounds_ties_one_fault_per_round(faults):
    """Each idle and CNOT parameter has one fault in each of the 5 rounds, all at the same
    position, on the same qubits, with the same Pauli."""
    fault_list = faults["rep_r5"]
    T, names, _, _ = tying_matrix(fault_list, "rounds")
    for key, rows in zip(names, members(T)):
        if key[0] in ("idle", "gate_2q"):
            group = [fault_list[j] for j in rows]
            assert sorted(f.round for f in group) == [1, 2, 3, 4, 5]
            assert len({(f.position, f.qubits, f.pauli) for f in group}) == 1


def test_rounds_ties_every_reset_of_an_ancilla(faults):
    """An ancilla's initial reset and its reset after each MR share one rate: 1 + 4 resets."""
    fault_list = faults["rep_r5"]
    T, names, _, _ = tying_matrix(fault_list, "rounds")
    resets = [rows for key, rows in zip(names, members(T)) if key[0] == "reset"]
    assert [len(rows) for rows in resets] == [5, 5]
    assert all(len({fault_list[j].qubits for j in rows}) == 1 for rows in resets)


def test_channels_ties_a_whole_channel(faults):
    """Every fault of a channel in one parameter, so each column holds that channel's count."""
    fault_list = faults["rep_r2"]
    T, names, _, _ = tying_matrix(fault_list, "channels")
    counts = Counter(f.channel for f in fault_list)
    assert {key: len(rows) for key, rows in zip(names, members(T))} == {
        key: counts[key] for key in names}


# --- options ---------------------------------------------------------------------------

def test_nothing_fixed(faults):
    """fixed=() puts every fault in T: T_fixed is (faults x 0), and prep gets its own parameters."""
    T, names, T_fixed, fixed_names = tying_matrix(faults["rep_r2"], "rounds", fixed=())
    assert T_fixed.shape == (len(faults["rep_r2"]), 0)
    assert fixed_names == []
    assert T.shape[1] == 64


def test_prep_and_readout_fixed(faults):
    T, names, T_fixed, fixed_names = tying_matrix(faults["rep_r2"], "rounds", fixed=("prep", "readout"))
    assert (T.shape[1], T_fixed.shape[1]) == (58, 6)
    assert {key[0] for key in fixed_names} == {"prep", "readout"}


def test_custom_rule(faults):
    """Any Fault -> key function: here one rate per (channel, qubits), ignoring Pauli and position."""
    fault_list = faults["rep_r2"]
    T, names, _, _ = tying_matrix(fault_list, lambda f: (f.channel, f.qubits))
    assert len(names) == len({(f.channel, f.qubits) for f in fault_list if f.channel != "prep"})


def test_unknown_fixed_channel_raises(faults):
    with pytest.raises(ValueError, match="unknown fixed channels"):
        tying_matrix(faults["rep_r2"], "rounds", fixed=("preparation",))


def test_unknown_rule_raises(faults):
    with pytest.raises(KeyError):
        tying_matrix(faults["rep_r2"], "round")
