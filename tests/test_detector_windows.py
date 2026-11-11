"""Tests for sliding windows: which round each detector is in, a window's faults and D',
moving it along the run, drifting rates, and fitting them (per window, and from one long run)."""

import numpy as np
import pytest
import stim
from sympy.abc import x, y

from qecdm.detector.circuits import bb_memory, repetition_memory, surface_memory
from qecdm.detector.estimate import (exact_expectations, fit_thetas, row_expectations, sample_detectors,
                                     true_thetas)
from qecdm.detector.faults import check_round_structure, circuit_faults
from qecdm.detector.windows import (detector_rounds, drifting_circuit, long_run_rates, shift_subsets,
                                    track_rates, true_window_thetas, window_design, window_faults,
                                    window_starts)

BB = ({x: 2, y: 3}, 1 + y + x * y**2, 1 + y + x * y**2)  # [[12,4]], the small certified BB code
NOISE = {"idle": 0.002, "gate_1q": 0.005, "gate_2q": 0.005, "measure": 0.01, "reset": 0.003, "readout": 0.02}
ROUNDS = 9


def ramp(t):
    """Three channels drifting at once: gates and idle up, measurement down."""
    return dict(NOISE, gate_2q=0.002 + 0.0004 * t, idle=0.001 + 0.0002 * t, measure=0.02 - 0.0005 * t)


CIRCUITS = {
    "rep": lambda: repetition_memory(3, ROUNDS, NOISE),
    "surface": lambda: surface_memory(3, ROUNDS, NOISE),
}


@pytest.fixture(scope="module")
def built():
    """(name, rule) -> (constant circuit, drifting circuit, window design at rounds 3-4), built once."""
    out = {}
    for name, make in CIRCUITS.items():
        circuit = make()
        drift = drifting_circuit(circuit, ramp)
        for rule in ("rounds", "channels"):
            out[name, rule] = circuit, drift, window_design(circuit, 3, 2, rule)[0]
    return out


# --- detector_rounds -------------------------------------------------------------------

def test_detector_rounds_of_the_repetition_code():
    """2 detectors a round for 5 rounds, then 2 that use the readout."""
    assert detector_rounds(repetition_memory(3, 5, NOISE)) == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, None, None]


@pytest.mark.parametrize("name, counts", [("surface", [4, 8, 8, 8, 8, 4]), ("bb", [6, 6, 6, 6, 6, 6])])
def test_detectors_per_round(name, counts):
    """Surface's first round has only the Z checks; the last entry is the readout detectors."""
    circuit = surface_memory(3, 5, NOISE) if name == "surface" else bb_memory(*BB, 5, NOISE)
    rounds = detector_rounds(circuit)
    assert [rounds.count(r) for r in [1, 2, 3, 4, 5, None]] == counts


@pytest.mark.parametrize("name", CIRCUITS)
def test_detector_rounds_match_stims_time_coordinate(name):
    """stim's generated circuits give each detector a time: its round - 1 (the readout's is the last)."""
    circuit = CIRCUITS[name]()
    rounds = detector_rounds(circuit)
    for d, coords in circuit.get_detector_coordinates().items():
        assert coords[-1] == (ROUNDS if rounds[d] is None else rounds[d] - 1)


# --- window_faults and window_design ---------------------------------------------------

@pytest.mark.parametrize("name, expected", [("rep", 144), ("surface", 1036)])
def test_window_keeps_the_faults_that_reach_it(name, expected):
    """Every fault that trips one of the window's detectors, and no other."""
    circuit = CIRCUITS[name]()
    H, faults, detectors = window_faults(circuit, 3, 2)
    full_H, _ = circuit_faults(circuit)
    reach = sum(1 for j in range(full_H.shape[1]) if set(full_H[:, j].indices) & set(detectors))
    assert len(faults) == H.shape[1] == reach == expected


def test_window_design_reaches_its_target_with_nothing_fixed(built):
    for (name, rule), (_, _, design) in built.items():
        assert np.linalg.matrix_rank(design.D) == design.target
        assert design.D_fixed.shape[1] == 0


def test_a_window_at_the_start_raises():
    """Round 1 holds the data preparation."""
    with pytest.raises(ValueError, match="prep"):
        window_design(repetition_memory(3, ROUNDS, NOISE), 1, 2)


# --- shift_subsets and window_starts ---------------------------------------------------

def test_shift_subsets_by_hand():
    """Window detectors 0 and 2 are the first of each round: circuit 4, 6 at round 3, and 8, 10 at round 5."""
    rounds = detector_rounds(repetition_memory(3, ROUNDS, NOISE))
    assert shift_subsets([(0, 2)], rounds, 3, 2) == [(4, 6)]
    assert shift_subsets([(0, 2)], rounds, 5, 2) == [(8, 10)]


def test_shift_subsets_raises_when_the_window_is_too_short(built):
    """Surface's rounds 1-2 have 12 detectors, but the design uses 16."""
    circuit, _, design = built["surface", "channels"]
    with pytest.raises(ValueError, match="only 12 detectors"):
        shift_subsets(design.subsets, detector_rounds(circuit), 1, 2)


@pytest.mark.parametrize("name, rule", [("rep", "rounds"), ("rep", "channels"),
                                        ("surface", "rounds"), ("surface", "channels")])
def test_one_design_fits_every_position(built, name, rule):
    """The design built at round 3, moved to each start, still matches stim exactly:
    log y = D theta, with theta from that position's own faults."""
    circuit, _, design = built[name, rule]
    rounds = detector_rounds(circuit)
    for start in window_starts(rounds, 2):
        _, faults, _ = window_faults(circuit, start, 2)
        theta, _ = true_thetas(faults, NOISE, design, rule)
        log_y = np.log(exact_expectations(circuit, shift_subsets(design.subsets, rounds, start, 2)))
        assert np.abs(log_y - design.D @ theta).max() < 1e-12


def test_window_starts_by_hand():
    """9 rounds, windows of 2: starts 2 to 8."""
    rounds = detector_rounds(repetition_memory(3, ROUNDS, NOISE))
    assert window_starts(rounds, 2) == [2, 3, 4, 5, 6, 7, 8]
    assert window_starts(rounds, 2, stride=3) == [2, 5, 8]
    assert window_starts(rounds, 2, stride=3, last=2) == [5, 8]


@pytest.mark.parametrize("name", CIRCUITS)
def test_windows_a_round_apart_share_no_faults(name):
    """Rounds 3-4 and 6-7 have nothing in common, so they are independent; 3-4 and 5-6 do share."""
    circuit = CIRCUITS[name]()
    faults = {start: {(f.offset, f.qubits, f.pauli) for f in window_faults(circuit, start, 2)[1]}
              for start in (3, 5, 6)}
    assert faults[3] & faults[5]
    assert not faults[3] & faults[6]


# --- drifting_circuit ------------------------------------------------------------------

@pytest.mark.parametrize("name", CIRCUITS)
def test_constant_rates_give_back_the_same_circuit(name):
    circuit = CIRCUITS[name]()
    assert str(drifting_circuit(circuit, lambda t: NOISE)) == str(circuit.flattened())


def test_a_ramp_sets_each_rounds_rate(built):
    """The repetition code's CNOT noise, in circuit order, steps through the ramp one round at a time."""
    _, drift, _ = built["rep", "rounds"]
    rates = [ins.gate_args_copy()[0] for ins in drift if ins.name == "DEPOLARIZE2"]
    assert list(dict.fromkeys(rates)) == pytest.approx([ramp(t)["gate_2q"] for t in range(1, ROUNDS + 1)])


def test_a_ramp_breaks_the_round_structure(built):
    circuit, drift, _ = built["rep", "rounds"]
    check_round_structure(circuit)
    with pytest.raises(ValueError, match="different noise template"):
        check_round_structure(drift)


# --- true_window_thetas ----------------------------------------------------------------

@pytest.mark.parametrize("name, rule", [("rep", "rounds"), ("surface", "channels")])
def test_constant_noise_gives_the_ordinary_truth(built, name, rule):
    """With no drift, every window's true value is Step 7's true theta (compared as D theta)."""
    circuit, _, design = built[name, rule]
    _, faults, _ = window_faults(circuit, 3, 2)
    static, _ = true_thetas(faults, NOISE, design, rule)
    truth = true_window_thetas(design, circuit, lambda t: NOISE, 2, window_starts(detector_rounds(circuit), 2))
    assert np.abs((truth - static) @ design.D.T).max() < 1e-12


@pytest.mark.parametrize("name, rule", [("rep", "rounds"), ("rep", "channels"),
                                        ("surface", "rounds"), ("surface", "channels")])
def test_true_values_match_fits_of_stims_exact_data(built, name, rule):
    """Our faults with each round's rate, against stim's own model of the drifting circuit:
    per window, and averaged over the last 2 windows a round apart."""
    _, drift, design = built[name, rule]
    rounds = detector_rounds(drift)

    def stim_y(start):
        return exact_expectations(drift, shift_subsets(design.subsets, rounds, start, 2))

    starts = window_starts(rounds, 2)
    stim_fits = [fit_thetas(design, stim_y(start), np.zeros(0)) for start in starts]
    assert np.abs(true_window_thetas(design, drift, ramp, 2, starts) - stim_fits).max() < 1e-12

    starts = window_starts(rounds, 2, stride=3, last=2)
    stim_fit = fit_thetas(design, np.mean([stim_y(start) for start in starts], axis=0), np.zeros(0))
    assert np.abs(true_window_thetas(design, drift, ramp, 2, starts, average=True) - stim_fit).max() < 1e-12


# --- track_rates and long_run_rates ----------------------------------------------------

def test_track_rates_returns_one_fit_per_window(built):
    _, drift, design = built["rep", "channels"]
    starts, thetas, rates = track_rates(design, drift, 2, 1000, stride=3, seed=1)
    assert starts == [2, 5, 8]
    assert thetas.shape == rates.shape == (3, len(design.groups))


def test_track_rates_follows_the_drift(built):
    """200,000 shots: every window's D theta within 0.01 of its true value."""
    _, drift, design = built["surface", "channels"]
    starts, thetas, _ = track_rates(design, drift, 2, 200_000, seed=1)
    truth = true_window_thetas(design, drift, ramp, 2, starts)
    assert np.abs((thetas - truth) @ design.D.T).max() < 0.01


@pytest.mark.parametrize("stride, last", [(1, None), (3, None), (1, 3)])
def test_long_run_averages_the_windows_then_fits(built, stride, last):
    """The same as averaging row_expectations over the chosen windows, on the same samples."""
    circuit, _, design = built["surface", "rounds"]
    rounds = detector_rounds(circuit)
    samples = sample_detectors(circuit, 20, seed=3)
    y = np.mean([row_expectations(samples, shift_subsets(design.subsets, rounds, start, 2))
                 for start in window_starts(rounds, 2, stride, last)], axis=0)
    theta, _ = long_run_rates(design, circuit, 2, last=last, stride=stride, shots=20, seed=3)
    assert theta == pytest.approx(fit_thetas(design, y, np.zeros(0)), abs=1e-12)
