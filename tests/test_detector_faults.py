"""Tests for the fault list: H has one column per circuit fault, each fault gets the right
channel, round and position, and rounds share one noise template."""

from collections import Counter

import pytest
import stim
from sympy.abc import x, y

from qecdm.detector.circuits import bb_memory, repetition_memory, surface_memory, uniform_noise
from qecdm.detector.faults import (CHANNELS, check_round_structure, circuit_faults, fault_round,
                                   qubit_roles)

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
    """name -> (circuit, H, faults), each built once and shared by every test."""
    out = {}
    for name, build in CIRCUITS.items():
        circuit = build()
        out[name] = (circuit, *circuit_faults(circuit))
    return out


def columns(H):
    """H -> list of tuples: the detectors of each column."""
    return [tuple(sorted(H[:, j].indices)) for j in range(H.shape[1])]


# --- H and the fault list ---------------------------------------------------------

def test_repetition_counts(built):
    """d = 3, 2 rounds: 122 detectable faults in 15 distinct patterns, by channel as worked
    out by hand (3 + 2 initial resets, 12 idle, 96 CNOT components, 4 measurement flips,
    2 resets after MR, 3 readout flips)."""
    _, H, faults = built["rep_r2"]
    assert H.shape == (6, 122)
    assert len(set(columns(H))) == 15
    assert Counter(f.channel for f in faults) == {
        "prep": 3, "reset": 4, "idle": 12, "gate_2q": 96, "measure": 4, "readout": 3}


@pytest.mark.parametrize("name", ["rep_r2", "rep_r5", "surface_r3", "bb_r3"])
def test_columns_are_the_dem_patterns(built, name):
    """The distinct columns of H are exactly stim's DEM detector patterns (stim also splits
    patterns by observable, which H ignores, so compare detector sets only)."""
    circuit, H, _ = built[name]
    dem = {tuple(sorted(t.val for t in ins.targets_copy() if t.is_relative_detector_id()))
           for ins in circuit.detector_error_model().flattened() if ins.type == "error"}
    assert set(columns(H)) == dem - {()}


@pytest.mark.parametrize("name", ["rep_r2", "rep_r5", "surface_r3", "bb_r3"])
def test_every_column_has_a_fault(built, name):
    _, H, faults = built[name]
    assert len(faults) == H.shape[1]
    assert all(H[:, j].nnz > 0 for j in range(H.shape[1]))  # no undetectable faults kept
    assert {f.channel for f in faults} <= set(CHANNELS)
    assert all(H.data == 1)


def test_diagonal_faults_are_cnot_faults(built):
    """The space-time diagonals D0 D3 and D2 D5 come only from data 2 being hit between
    its two CNOTs: 8 two-qubit components in each round."""
    _, H, faults = built["rep_r2"]
    diagonal = [f for f, c in zip(faults, columns(H)) if c in ((0, 3), (2, 5))]
    assert len(diagonal) == 16
    assert {f.channel for f in diagonal} == {"gate_2q"}


def test_measurement_flips_are_paulis_in_stim_and_flips_in_qldpc(built):
    """stim's measurement errors are X_ERRORs, so they get a Pauli; qldpc's are M(p), so 'flip'."""
    _, _, rep = built["rep_r2"]
    assert {f.pauli for f in rep if f.channel in ("measure", "readout")} == {"X0", "X1", "X2", "X3", "X4"}
    _, _, bb = built["bb_r3"]
    assert {f.pauli for f in bb if f.channel in ("measure", "readout")} == {"flip"}


def test_qldpc_channels(built):
    """Z-memory: only the 6 Z-check ancillas' faults are detected (6 x 3 rounds of measure
    and reset), all 12 data qubits' prep and readout; idle on every idle qubit every moment."""
    _, _, faults = built["bb_r3"]
    assert Counter(f.channel for f in faults) == {
        "gate_2q": 2448, "idle": 912, "prep": 12, "measure": 18, "reset": 18, "readout": 12}


# --- roles, rounds and positions ---------------------------------------------------

def test_qubit_roles():
    assert qubit_roles(repetition_memory(3, 2, NOISE)) == {
        0: "data", 2: "data", 4: "data", 1: "ancilla", 3: "ancilla"}
    roles = qubit_roles(surface_memory(3, 2, NOISE))
    assert Counter(roles.values()) == {"data": 9, "ancilla": 8}  # the 9 unused grid slots are left out


def test_fault_round():
    """Round ends at moments 3 and 6: positions count back from each round's measurement,
    and moments after the last one are an extra round with positive positions."""
    ends = [3, 6]
    assert [fault_round(m, ends) for m in range(8)] == [
        (1, -3), (1, -2), (1, -1), (1, 0), (2, -2), (2, -1), (2, 0), (3, 1)]


def test_repetition_templates(built):
    """Every (channel, round, position) in the 2-round repetition code. Round 1 alone has
    the preparation step (-3); both rounds have idle and CNOT at -2, CNOT at -1, and their
    measurement and reset flips at 0; stim puts the readout in round 2's measurement step."""
    _, _, faults = built["rep_r2"]
    assert {(f.channel, f.round, f.position) for f in faults} == {
        ("prep", 1, -3), ("reset", 1, -3),
        ("idle", 1, -2), ("gate_2q", 1, -2), ("gate_2q", 1, -1), ("measure", 1, 0), ("reset", 1, 0),
        ("idle", 2, -2), ("gate_2q", 2, -2), ("gate_2q", 2, -1), ("measure", 2, 0), ("readout", 2, 0)}


def test_rounds_line_up(built):
    """The same faults, keyed by (channel, position, qubits, Pauli), appear in every bulk round:
    this is what per-round tying will rely on."""
    _, _, faults = built["rep_r5"]
    by_round = {}
    for f in faults:
        by_round.setdefault(f.round, set()).add((f.channel, f.position, f.qubits, f.pauli))
    assert by_round[2] == by_round[3] == by_round[4]
    assert len(by_round[2]) == 58


def test_qldpc_readout_is_its_own_round(built):
    """qldpc measures the data in a later moment than the last MX, so the readout is round 4."""
    _, _, faults = built["bb_r3"]
    assert Counter(f.channel for f in faults if f.round == 4) == {"readout": 12}
    assert all(f.position > 0 for f in faults if f.round == 4)


# --- failures ------------------------------------------------------------------------

def test_unclassifiable_fault_raises():
    """A flip alone in its moment: no reset before it, no measurement after it."""
    circuit = stim.Circuit("R 0\nTICK\nX_ERROR(0.1) 0\nTICK\nM 0\nDETECTOR rec[-1]")
    with pytest.raises(ValueError, match="cannot classify"):
        circuit_faults(circuit)


@pytest.mark.parametrize("name", ["rep_r5", "surface_r3", "bb_r3"])
def test_round_structure_holds(built, name):
    check_round_structure(built[name][0])  # raises if not


def test_round_structure_catches_a_changed_round():
    """Give round 3's first CNOT layer a different rate: rounds 2 and 3 no longer match."""
    circuit = repetition_memory(3, 4, NOISE)
    out, seen = stim.Circuit(), 0
    for ins in circuit:
        if ins.name == "DEPOLARIZE2":
            seen += 1
            if seen == 5:  # rounds have 2 CNOT layers each, so the 5th is round 3's first
                out.append(ins.name, ins.targets_copy(), 0.02)
                continue
        out.append(ins)
    with pytest.raises(ValueError, match="round 3 has a different noise template"):
        check_round_structure(out)
