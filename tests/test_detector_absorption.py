"""Tests for crosstalk absorption: the family (one crosstalk per shared idle interval of a
data pair), its detector patterns, its tied straddles over a window, and which ones the
window D' absorbs."""

from collections import Counter

import numpy as np
import pytest
import stim

from qecdm.detector.absorption import (Crosstalk, absorbed_crosstalk, error_patterns, idle_crosstalk,
                                       window_straddles)
from qecdm.detector.circuits import repetition_memory, surface_memory
from qecdm.detector.detector_dmatrix import DetectorDesign
from qecdm.detector.estimate import exact_expectations, fit_thetas
from qecdm.detector.windows import detector_rounds, shift_subsets, window_design

NOISE = {"idle": 0.002, "gate_1q": 0.005, "gate_2q": 0.005, "measure": 0.01, "reset": 0.003, "readout": 0.02}
ROUNDS = 9

# Data 0, 1, 2 and one ancilla, 3. The data qubits idle together:
#   moment 1 (CX 0 3): 1, 2      moment 2 (H 3), moment 3 (MR 3): 0, 1, 2
#   moment 4 (CX 1 3): 0, 2      moment 5 (MR 3): 0, 1, 2      then the readout
SMALL = stim.Circuit("""
R 0 1 2 3
TICK
CX 0 3
TICK
H 3
TICK
MR 3
TICK
CX 1 3
TICK
MR 3
TICK
M 0 1 2
""")


def key(error):
    return error.position, error.qubits, error.pauli


@pytest.fixture(scope="module")
def surface():
    return surface_memory(3, ROUNDS, NOISE)


@pytest.fixture(scope="module")
def results(surface):
    """(name, rule) -> Absorption, each worked out once."""
    rep = repetition_memory(3, ROUNDS, NOISE)
    return {("rep", "rounds"): absorbed_crosstalk(rep, "rounds"),
            ("surface", "rounds"): absorbed_crosstalk(surface, "rounds"),
            ("surface", "channels"): absorbed_crosstalk(surface, "channels")}


# --- idle_crosstalk --------------------------------------------------------------------

def test_one_crosstalk_per_shared_idle_interval():
    """Pairs (0,1) and (1,2) are idle together in moments 1-3 or 2-3, and again in moment 5;
    (0,2) from moment 2 right through to 5. Each interval ends at its last moment."""
    intervals = {(e.qubits, e.label.split()[-1]) for e in idle_crosstalk(SMALL)}
    assert intervals == {((0, 1), "m3"), ((1, 2), "m3"), ((0, 1), "m5"), ((1, 2), "m5"), ((0, 2), "m5")}


def test_each_interval_gets_the_nine_paulis():
    paulis = Counter((e.qubits, e.offset) for e in idle_crosstalk(SMALL))
    assert set(paulis.values()) == {9}
    assert {e.pauli for e in idle_crosstalk(SMALL)} == {a + b for a in "XYZ" for b in "XYZ"}


def test_crosstalk_goes_just_before_the_tick_ending_its_interval():
    """Moment 3 ends at the TICK at index 7, moment 5 at index 11; both moments end a round."""
    for e in idle_crosstalk(SMALL):
        assert SMALL[e.offset].name == "TICK"
        assert (e.offset, e.round, e.position) in {(7, 1, 0), (11, 2, 0)}


@pytest.mark.parametrize("name, per_position", [
    ("rep", {0: 27}),                                # all of it in the gap between rounds
    ("surface", {-6: 297, -5: 27, -4: 27, -3: 27}),  # the gap, then CX layers 1-3
])
def test_crosstalk_per_round(surface, name, per_position):
    circuit = repetition_memory(3, ROUNDS, NOISE) if name == "rep" else surface
    assert Counter(e.position for e in idle_crosstalk(circuit) if e.round == 5) == per_position


# --- error_patterns --------------------------------------------------------------------

TINY = stim.Circuit("""
R 0 1 2 3
TICK
M(0.1) 0 1 2 3
DETECTOR rec[-4]
DETECTOR rec[-3]
DETECTOR rec[-2]
""")  # detector i is qubit i's measurement; the measurement's own flip rate must be ignored


def test_error_patterns_by_hand():
    """X flips a Z measurement and Z does not: X0X1 trips D0 and D1, Z0Z1 trips nothing."""
    errors = [Crosstalk(pair, pauli, 1, 0, 1, "")
              for pair, pauli in [((0, 1), "XX"), ((1, 2), "XX"), ((0, 1), "ZZ"), ((0, 2), "XX")]]
    assert error_patterns(TINY, errors) == [(0, 1), (1, 2), (), (0, 2)]


def test_error_patterns_match_stim_one_at_a_time():
    """Each repetition-code crosstalk on its own, in stim's own noiseless copy, gives the same detectors."""
    circuit = repetition_memory(3, 4, NOISE)
    errors = idle_crosstalk(circuit)
    flat = circuit.flattened()
    for error, pattern in zip(errors, error_patterns(circuit, errors)):
        ticks_before = sum(1 for ins in flat[:error.offset] if ins.name == "TICK")
        alone, seen = stim.Circuit(), 0
        for ins in flat.without_noise():
            if ins.name == "TICK":
                if seen == ticks_before:
                    alone.append("E", [stim.target_pauli(q, p) for q, p in zip(error.qubits, error.pauli)], 0.01)
                seen += 1
            alone.append(ins)
        flips = [e for e in alone.detector_error_model().flattened() if e.type == "error"]
        expected = tuple(sorted(t.val for e in flips for t in e.targets_copy() if t.is_relative_detector_id()))
        assert pattern == expected


# --- window_straddles ------------------------------------------------------------------

def test_window_straddles_by_hand():
    """Window rounds 4-5 (detectors 6-9). A's copies in rounds 3, 4, 5 each flip row (7,) or (9,)
    once, so A gets 2 there. B trips 6 and 8: it flips (6,) and (8,), but not (6, 8)."""
    design = DetectorDesign(None, None, None, None, [(0,), (1,), (2,), (3,), (0, 2)], None, None)
    rounds = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, None, None]
    errors = [Crosstalk((0, 2), "XX", r, 0, 0, "") for r in (3, 4, 5)] + \
             [Crosstalk((0, 4), "XZ", r, -1, 0, "") for r in (4, 9)] + [Crosstalk((2, 4), "ZZ", 4, 0, 0, "")]
    patterns = [(5, 7), (7, 9), (9, 11), (6, 8), (16, 17), ()]
    S, keys = window_straddles(design, rounds, 4, 2, errors, patterns)
    assert keys == [(0, (0, 2), "XX"), (-1, (0, 4), "XZ")]
    assert S.tolist() == [[0, 1], [2, 0], [0, 1], [2, 0], [0, 0]]


def test_straddles_are_the_same_at_every_window(surface):
    """One design and one family give the same column for each crosstalk at every start."""
    design, _ = window_design(surface, 3, 2, "channels")
    errors = idle_crosstalk(surface)
    patterns = error_patterns(surface, errors)
    rounds = detector_rounds(surface)
    columns = []
    for start in (2, 4, 7):
        S, keys = window_straddles(design, rounds, start, 2, errors, patterns)
        columns.append({k: S[:, j].tolist() for j, k in enumerate(keys)})
    assert columns[0] == columns[1] == columns[2]


# --- absorbed_crosstalk ----------------------------------------------------------------

@pytest.mark.parametrize("name, rule, counts", [
    ("rep", "rounds", (24, 0, 3)),
    ("surface", "rounds", (126, 248, 4)),
    ("surface", "channels", (0, 374, 4)),
])
def test_absorption_counts(results, name, rule, counts):
    """(absorbed, resolved, invisible). Under "channels" nothing can hide."""
    result = results[name, rule]
    assert (len(result.absorbed), len(result.resolved), len(result.invisible)) == counts


def test_every_crosstalk_has_one_verdict(results):
    result = results["surface", "rounds"]
    keys = result.absorbed + result.resolved + result.invisible
    assert len(keys) == len(set(keys))


def with_crosstalk(circuit, crosstalk, p):
    """The noisy circuit with this crosstalk added, at rate p, in every round."""
    places = {e.offset: e for e in idle_crosstalk(circuit) if key(e) == crosstalk}
    out = stim.Circuit()
    for i, ins in enumerate(circuit.flattened()):
        if i in places:
            out.append("E", [stim.target_pauli(q, s) for q, s in zip(places[i].qubits, places[i].pauli)], p)
        out.append(ins)
    return out


@pytest.mark.parametrize("verdict, misfit", [("absorbed", False), ("resolved", True)])
def test_the_verdict_holds_in_stims_exact_data(surface, results, verdict, misfit):
    """Add one crosstalk at p = 0.004 and fit stim's exact y for a window: an absorbed one is
    fitted perfectly by moving the model's rates, a resolved one leaves a misfit."""
    crosstalk = getattr(results["surface", "rounds"], verdict)[0]
    design, _ = window_design(surface, 3, 2, "rounds")
    rows = shift_subsets(design.subsets, detector_rounds(surface), 4, 2)
    log_y = np.log(exact_expectations(with_crosstalk(surface, crosstalk, 0.004), rows))
    theta = fit_thetas(design, np.exp(log_y), np.zeros(0))
    assert (np.abs(log_y - design.D @ theta).max() > 1e-6) == misfit


def test_a_rank_deficient_design_raises():
    """Under "gates" the window D' has unlearnable directions, which split_mod_p cannot handle."""
    with pytest.raises(ValueError, match="rank deficient"):
        absorbed_crosstalk(repetition_memory(3, ROUNDS, NOISE), "gates")
