"""Tests for the tying matrix: every fault lands in exactly one parameter, the rules give
the parameters worked out by hand, and fixed channels go to T_fixed. Then M, merging and
identifiability: the rank target and the directions the detectors cannot see."""

from collections import Counter

import numpy as np
import pytest
from scipy import sparse
from sympy.abc import x, y

from qecdm.detector.circuits import bb_memory, repetition_memory, surface_memory, uniform_noise
from qecdm.detector.faults import circuit_faults
from qecdm.detector.tying import identifiability, merge_parameters, signature_counts, tying_matrix

BB = ({x: 2, y: 3}, 1 + y + x * y**2, 1 + y + x * y**2)  # [[12,4]], the small certified BB code
NOISE = uniform_noise(0.01)

CIRCUITS = {
    "rep_r2": lambda: repetition_memory(3, 2, NOISE),
    "rep_r5": lambda: repetition_memory(3, 5, NOISE),
    "surface_r3": lambda: surface_memory(3, 3, NOISE),
    "bb_r3": lambda: bb_memory(*BB, 3, NOISE),
}


@pytest.fixture(scope="module")
def built():
    """name -> (H, faults), each built once and shared by every test."""
    return {name: circuit_faults(build()) for name, build in CIRCUITS.items()}


@pytest.fixture(scope="module")
def faults(built):
    """name -> list of Fault."""
    return {name: fault_list for name, (_, fault_list) in built.items()}


def merged(built, name, rule="rounds", fixed=("prep",)):
    """The merged M and its groups for one circuit and rule."""
    H, fault_list = built[name]
    T, names, _, _ = tying_matrix(fault_list, rule, fixed)
    M, _ = signature_counts(H, T)
    return merge_parameters(M, names)


def members(T):
    """T -> list of lists: the faults (row indices) in each parameter (column)."""
    return [list(T[:, k].indices) for k in range(T.shape[1])]


# --- every fault exactly once ------------------------------------------------------

@pytest.mark.parametrize("name", list(CIRCUITS))
@pytest.mark.parametrize("rule", ["rounds", "gates", "channels"])
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


def test_gates_ties_every_pauli_of_a_gate(faults):
    """One rate per CNOT: each gate_2q parameter holds its 12 detectable Paulis in both
    rounds, all on the same qubit pair at the same position."""
    fault_list = faults["rep_r2"]
    T, names, _, _ = tying_matrix(fault_list, "gates")
    assert Counter(key[0] for key in names) == {
        "idle": 3, "gate_2q": 4, "measure": 2, "reset": 2, "readout": 3}
    for key, rows in zip(names, members(T)):
        if key[0] == "gate_2q":
            group = [fault_list[j] for j in rows]
            assert len(group) == 24 and len({f.pauli for f in group}) == 12
            assert len({(f.position, f.qubits) for f in group}) == 1


@pytest.mark.parametrize("name, gates", [("rep_r2", 14), ("rep_r5", 14), ("surface_r3", 66),
                                         ("bb_r3", 252)])
def test_gates_parameter_counts(faults, name, gates):
    T, _, _, _ = tying_matrix(faults[name], "gates")
    assert T.shape[1] == gates


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


# --- signature_counts ------------------------------------------------------------------

# 5 faults on detectors D0, D2: patterns (0,), (0, 2), (0,), (2,), (2,); parameters
# A = {0, 3}, B = {1}, C = {2, 4}.
TOY_H = sparse.csc_matrix(np.array([[1, 1, 1, 0, 0],
                                    [0, 0, 0, 0, 0],
                                    [0, 1, 0, 1, 1]]))
TOY_T = sparse.csc_matrix(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1]]))


def test_signature_counts_by_hand():
    """M[sigma, k] counts parameter k's faults with pattern sigma."""
    M, signatures = signature_counts(TOY_H, TOY_T)
    assert signatures == [(0,), (0, 2), (2,)]
    assert M.tolist() == [[1, 0, 1],
                          [0, 1, 0],
                          [1, 0, 1]]


@pytest.mark.parametrize("name", list(CIRCUITS))
def test_signature_counts_keep_every_fault(built, name):
    """Each fault has one pattern, so M's columns add up to each parameter's number of
    faults, and the signatures are H's distinct columns."""
    H, fault_list = built[name]
    T, _, _, _ = tying_matrix(fault_list, "rounds")
    M, signatures = signature_counts(H, T)
    assert M.sum(axis=0).tolist() == np.asarray(T.sum(axis=0)).ravel().tolist()
    assert len(signatures) == len(set(signatures)) == M.shape[0]


# --- merge_parameters ------------------------------------------------------------------

def test_merge_by_hand():
    """A and C have the same column, so they merge; B stays alone."""
    M, _ = signature_counts(TOY_H, TOY_T)
    M_merged, groups = merge_parameters(M, ["A", "B", "C"])
    assert groups == [["A", "C"], ["B"]]
    assert M_merged.tolist() == [[1, 0], [0, 1], [1, 0]]


@pytest.mark.parametrize("name", ["rep_r2", "surface_r3"])
def test_merge_keeps_the_rank(built, name):
    """Merging only removes repeated columns, so the rank is unchanged."""
    H, fault_list = built[name]
    T, names, _, _ = tying_matrix(fault_list, "rounds")
    M, _ = signature_counts(H, T)
    M_merged, groups = merge_parameters(M, names)
    assert np.linalg.matrix_rank(M_merged) == np.linalg.matrix_rank(M)
    assert sorted(n for group in groups for n in group) == sorted(names)  # every name in one group


@pytest.mark.parametrize("name", ["rep_r2", "surface_r3", "bb_r3"])
def test_measure_and_reset_always_merge(built, name):
    """An ancilla reset flip and its next measurement flip trip the same detectors."""
    _, groups = merged(built, name, "channels")
    assert ["measure", "reset"] in groups


# --- identifiability -------------------------------------------------------------------

def test_identifiability_by_hand():
    """Columns a + b = c: the step (+1, +1, -1) changes no count, so the rank is 2."""
    M = np.array([[1, 0, 1], [0, 1, 1]])
    rank, directions = identifiability(M, ["a", "b", "c"])
    assert rank == 2
    assert directions in ([[("a", -1), ("b", -1), ("c", 1)]], [[("a", 1), ("b", 1), ("c", -1)]])


@pytest.mark.parametrize("name, rule, fixed, columns, rank", [
    ("rep_r2", "rounds", ("prep",), 12, 12),
    ("rep_r2", "rounds", (), 15, 12),
    ("rep_r2", "channels", ("prep",), 4, 4),
    ("rep_r5", "rounds", ("prep",), 12, 12),          # more rounds, the same target
    ("surface_r3", "rounds", ("prep",), 135, 126),
    ("surface_r3", "channels", ("prep",), 5, 5),
    ("rep_r2", "gates", ("prep",), 12, 12),
    ("surface_r3", "gates", ("prep",), 52, 48),
    ("bb_r3", "rounds", ("prep",), 63, 57),
    ("bb_r3", "gates", ("prep",), 111, 57),
    ("bb_r3", "channels", ("prep",), 4, 4),
])
def test_rank_targets(built, name, rule, fixed, columns, rank):
    M_merged, groups = merged(built, name, rule, fixed)
    found, directions = identifiability(M_merged, groups)
    assert (M_merged.shape[1], found) == (columns, rank)
    assert found == np.linalg.matrix_rank(M_merged)
    assert len(directions) == columns - rank


@pytest.mark.parametrize("name", ["rep_r2", "surface_r3", "bb_r3"])
def test_directions_change_no_count(built, name):
    """Rebuild each direction as a vector over the merged columns: M v = 0."""
    M_merged, groups = merged(built, name)
    _, directions = identifiability(M_merged, groups)
    for direction in directions:
        v = np.zeros(len(groups), dtype=int)
        for group, coefficient in direction:
            v[groups.index(group)] = coefficient
        assert v.any() and not (M_merged @ v).any()


def test_repetition_gauge_without_prep(built):
    """Nothing fixed: one direction per data qubit, each holding that qubit's prep and its
    readout with opposite signs (the prep, idle, after-check CNOT, readout gauge)."""
    _, directions = identifiability(*merged(built, "rep_r2", fixed=()))
    assert len(directions) == 3
    for direction in directions:
        sign = {group[0][0]: c for group, c in direction if group[0][0] in ("prep", "readout")}
        assert len(direction) == 4
        assert sign["prep"] == -sign["readout"]
    preps = {group[0][1] for direction in directions for group, _ in direction if group[0][0] == "prep"}
    assert preps == {(0,), (2,), (4,)}  # a different data qubit in each


def test_surface_gates_directions_are_the_hadamards(built):
    """Under "gates", the surface code's 4 directions are its 4 X-check ancillas. The
    depolarizing after each of the ancilla's two Hs has two Paulis that only flip its
    measurement (Z, Y after the first H; X, Y after the last), so both Hs merge into one
    group whose column is twice the measure (+ reset) column."""
    _, directions = identifiability(*merged(built, "surface_r3", "gates"))
    assert len(directions) == 4
    for direction in directions:
        assert sorted((group[0][0], c) for group, c in direction) == [("gate_1q", 1), ("measure", -2)]
        assert len({group[0][2] for group, _ in direction}) == 1  # both keys: (channel, position, qubits, ...)
