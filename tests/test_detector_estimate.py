"""Tests for the true values: the rate of one fault, the true thetas, stim's exact
expectations, and that the model D theta + D_fixed theta_fixed reproduces them. Then
the sampled expectations, against the exact ones, and the fit from data back to thetas and rates."""

import numpy as np
import pytest
import stim
from sympy.abc import x, y

from qecdm.detector.circuits import bb_memory, repetition_memory, surface_memory, uniform_noise
from qecdm.detector.detector_dmatrix import DetectorDesign, detector_design_matrix
from qecdm.detector.estimate import (exact_expectations, fault_probability, fit_thetas, rates_from_thetas,
                                     row_expectations, sample_detectors, true_thetas)
from qecdm.detector.faults import Fault, circuit_faults

BB = ({x: 2, y: 3}, 1 + y + x * y**2, 1 + y + x * y**2)  # [[12,4]], the small certified BB code
# Different rates per channel, so a mix-up between two channels cannot go unseen. (stim's
# generated circuits need one rate for all gates; the formulas still differ between them.)
NOISE = {"idle": 0.002, "gate_1q": 0.005, "gate_2q": 0.005, "measure": 0.01, "reset": 0.003, "readout": 0.02}

CIRCUITS = {
    "rep_r2": lambda: repetition_memory(3, 2, NOISE),
    "surface_r3": lambda: surface_memory(3, 3, NOISE),
    "bb_r3": lambda: bb_memory(*BB, 3, NOISE),
}
RULES = ("rounds", "gates", "channels")
CASES = [(name, rule) for name in CIRCUITS for rule in RULES]


@pytest.fixture(scope="module")
def built():
    """(name, rule) -> (circuit, faults, DetectorDesign), each built once and shared by every test."""
    out = {}
    for name, make in CIRCUITS.items():
        circuit = make()
        _, faults = circuit_faults(circuit)
        for rule in RULES:
            out[name, rule] = circuit, faults, detector_design_matrix(circuit, rule)
    return out


def fault(channel):
    """A Fault of this channel: fault_probability reads nothing else."""
    return Fault(channel, (0,), "X0", 1, 0, 0, "")


def dem_probabilities(circuit):
    """stim circuit -> the probability of each error in its detector error model."""
    return [e.args_copy()[0] for e in circuit.detector_error_model().flattened() if e.type == "error"]


# --- fault_probability -----------------------------------------------------------------

@pytest.mark.parametrize("channel", ["measure", "reset", "readout"])
def test_a_flip_keeps_its_own_rate(channel):
    assert fault_probability(fault(channel), NOISE) == NOISE[channel]


def test_prep_uses_the_reset_rate():
    assert fault_probability(fault("prep"), NOISE) == NOISE["reset"]


@pytest.mark.parametrize("channel", ["idle", "gate_1q"])
def test_one_qubit_depolarizing_by_hand(channel):
    assert fault_probability(fault(channel), {channel: 0.1}) == pytest.approx(0.0345253, abs=1e-7)


def test_two_qubit_depolarizing_by_hand():
    assert fault_probability(fault("gate_2q"), {"gate_2q": 0.1}) == pytest.approx(0.0070003, abs=1e-7)


def test_p_solves_the_defining_equations():
    """Each anticommuting independent fault gives a factor q = 1 - 2p; the channel gives 1 - 4r/3 or 1 - 16r/15."""
    r = 0.1
    assert (1 - 2 * fault_probability(fault("idle"), {"idle": r})) ** 2 == pytest.approx(1 - 4 * r / 3)
    assert (1 - 2 * fault_probability(fault("gate_2q"), {"gate_2q": r})) ** 8 == pytest.approx(1 - 16 * r / 15)


def test_one_qubit_p_matches_stim():
    """A Bell pair around the channel gives each of the 3 Paulis its own detector pattern."""
    r = 0.1
    circuit = stim.Circuit(f"""
        R 0 1
        H 0
        CNOT 0 1
        DEPOLARIZE1({r}) 0
        CNOT 0 1
        H 0
        M 0 1
        DETECTOR rec[-2]
        DETECTOR rec[-1]
    """)
    p = fault_probability(fault("idle"), {"idle": r})
    assert dem_probabilities(circuit) == pytest.approx([p] * 3)


def test_two_qubit_p_matches_stim():
    """Two Bell pairs give each of the 15 Paulis its own detector pattern.
    One distinct line in dem corresponds to each of the 15 2 qubit Paulis."""
    r = 0.1
    circuit = stim.Circuit(f"""
        R 0 1 2 3
        H 0 2
        CNOT 0 1 2 3
        DEPOLARIZE2({r}) 0 2
        CNOT 0 1 2 3
        H 0 2
        M 0 1 2 3
        DETECTOR rec[-4]
        DETECTOR rec[-3]
        DETECTOR rec[-2]
        DETECTOR rec[-1]
    """)
    p = fault_probability(fault("gate_2q"), {"gate_2q": r})
    assert dem_probabilities(circuit) == pytest.approx([p] * 15)


# --- true_thetas -----------------------------------------------------------------------

def test_true_thetas_by_hand(built):
    """Repetition under "channels": each column's theta, with measure and reset merged."""
    _, faults, design = built["rep_r2", "channels"]
    theta, theta_fixed = true_thetas(faults, NOISE, design, "channels")
    p_idle = fault_probability(fault("idle"), NOISE)
    assert theta[design.groups.index(["idle"])] == pytest.approx(np.log(1 - 2 * p_idle))
    assert theta[design.groups.index(["readout"])] == pytest.approx(np.log(1 - 2 * NOISE["readout"]))
    assert theta[design.groups.index(["measure", "reset"])] == pytest.approx(
        np.log(1 - 2 * NOISE["measure"]) + np.log(1 - 2 * NOISE["reset"]))
    assert design.fixed_names == ["prep"]
    assert theta_fixed == pytest.approx([np.log(1 - 2 * NOISE["reset"])])


@pytest.mark.parametrize("name, rule", CASES)
def test_true_thetas_line_up_with_the_columns(built, name, rule):
    _, faults, design = built[name, rule]
    theta, theta_fixed = true_thetas(faults, NOISE, design, rule)
    assert theta.shape == (design.D.shape[1],)
    assert theta_fixed.shape == (design.D_fixed.shape[1],)


def test_tying_faults_with_different_rates_raises(built):
    """One key for every fault says one rate, but the channels have different rates."""
    _, faults, design = built["rep_r2", "channels"]
    with pytest.raises(ValueError, match="different rates"):
        true_thetas(faults, NOISE, design, lambda f: "everything")


# --- exact_expectations ----------------------------------------------------------------

TOY = stim.Circuit("""
    R 0 1 2
    X_ERROR(0.1) 0
    X_ERROR(0.05) 1
    X_ERROR(0.2) 2
    CNOT 2 0 2 1
    M 0 1
    DETECTOR rec[-2]
    DETECTOR rec[-1]
""")  # errors: 0.1 on D0, 0.05 on D1, 0.2 on both D0 and D1


def test_exact_expectations_by_hand():
    """Each error with odd overlap gives a factor 1 - 2p: the 0.2 error flips both detectors,
    so it leaves the row (0, 1) alone, giving 0.8 * 0.9 there and no factor of 0.6."""
    assert exact_expectations(TOY, [(0,), (1,), (0, 1)]) == pytest.approx([0.8 * 0.6, 0.9 * 0.6, 0.8 * 0.9])


def test_exact_expectations_follow_the_order_of_the_subsets():
    assert exact_expectations(TOY, [(0, 1), (1,), (0,)]) == pytest.approx([0.8 * 0.9, 0.9 * 0.6, 0.8 * 0.6])


def test_no_noise_gives_one():
    circuit = repetition_memory(3, 2, uniform_noise(0))
    assert exact_expectations(circuit, [(0,), (1,), (0, 2)]) == pytest.approx([1, 1, 1])


# --- the model against stim ------------------------------------------------------------

@pytest.mark.parametrize("name, rule", CASES)
def test_model_reproduces_stim_exactly(built, name, rule):
    """log y from stim's own error model = D theta + D_fixed theta_fixed from ours."""
    circuit, faults, design = built[name, rule]
    theta, theta_fixed = true_thetas(faults, NOISE, design, rule)
    log_y = np.log(exact_expectations(circuit, design.subsets))
    assert np.abs(log_y - (design.D @ theta + design.D_fixed @ theta_fixed)).max() < 1e-12


# --- samples ---------------------------------------------------------------------------

def test_samples_have_one_row_per_shot_and_one_column_per_detector():
    assert sample_detectors(TOY, 10, seed=1).shape == (10, 2)


def test_the_seed_fixes_the_samples():
    a, b, c = (sample_detectors(TOY, 1000, seed=seed) for seed in (5, 5, 6))
    assert (a == b).all()
    assert (a != c).any()


def test_no_noise_never_fires():
    circuit = repetition_memory(3, 2, uniform_noise(0))
    assert not sample_detectors(circuit, 100, seed=1).any()


def test_row_expectations_by_hand():
    """Row (0, 1): only shot 1 has odd parity, as both detectors fired in shot 2. 1 - 2 * 1/4 = 0.5."""
    samples = np.array([[1, 0],
                        [1, 1],
                        [0, 0],
                        [0, 0]], dtype=bool)
    assert row_expectations(samples, [(0,), (1,), (0, 1)]) == pytest.approx([0, 0.5, 0.5])


@pytest.mark.parametrize("name", CIRCUITS)
def test_samples_approach_the_exact_expectations(built, name):
    """Every row within 5 standard errors of stim's exact value, sqrt((1 - y^2) / shots)."""
    circuit, _, design = built[name, "rounds"]
    shots = 100_000
    exact = exact_expectations(circuit, design.subsets)
    sampled = row_expectations(sample_detectors(circuit, shots, seed=1), design.subsets)
    assert (np.abs(sampled - exact) < 5 * np.sqrt((1 - exact**2) / shots)).all()


# --- fit_thetas ------------------------------------------------------------------------

def small_design(D, D_fixed=None, groups=(), directions=()):
    """A DetectorDesign by hand, with only the fields the fit and the rates read."""
    D = np.array(D)
    D_fixed = np.zeros((len(D), 0)) if D_fixed is None else np.array(D_fixed)
    return DetectorDesign(D, list(groups), D_fixed, [], [], 0, list(directions))


def test_fit_by_hand():
    """Three rows and two unknowns, with the known prep moved to the other side."""
    D, D_fixed = [[1, 0], [1, 1], [0, 1]], [[1], [0], [0]]
    theta, theta_fixed = np.array([-0.1, -0.2]), np.array([-0.3])
    y = np.exp(np.array(D) @ theta + np.array(D_fixed) @ theta_fixed)
    assert fit_thetas(small_design(D, D_fixed), y, theta_fixed) == pytest.approx(theta)


def test_fit_is_least_squares():
    """Two rows measure one theta and disagree: the best fit is their average."""
    y = np.exp([-0.1, -0.3])
    assert fit_thetas(small_design([[1], [1]]), y, np.array([])) == pytest.approx([-0.2])


def test_fit_picks_the_smallest_theta_in_a_null_space():
    """Only 2a + b = c is measured, and (2c/5, c/5) is the smallest solution."""
    c = -0.5
    assert fit_thetas(small_design([[2, 1]]), np.exp([c]), np.array([])) == pytest.approx([2 * c / 5, c / 5])


@pytest.mark.parametrize("bad", [0.0, -0.1])
def test_fit_raises_when_a_row_has_no_log(bad):
    with pytest.raises(ValueError, match="y <= 0"):
        fit_thetas(small_design([[1], [1]]), np.array([0.9, bad]), np.array([]))


@pytest.mark.parametrize("name, rule", CASES)
def test_fit_of_exact_data_reproduces_every_row(built, name, rule):
    """D theta is always determined, even where theta itself is not."""
    circuit, faults, design = built[name, rule]
    true, theta_fixed = true_thetas(faults, NOISE, design, rule)
    fit = fit_thetas(design, exact_expectations(circuit, design.subsets), theta_fixed)
    assert np.abs(design.D @ (fit - true)).max() < 1e-12


@pytest.mark.parametrize("name, rule", [case for case in CASES if case[1] == "channels"])
def test_fit_of_exact_data_gives_the_true_thetas(built, name, rule):
    """With no unlearnable directions, each theta itself comes back."""
    circuit, faults, design = built[name, rule]
    true, theta_fixed = true_thetas(faults, NOISE, design, rule)
    assert not design.directions
    fit = fit_thetas(design, exact_expectations(circuit, design.subsets), theta_fixed)
    assert np.abs(fit - true).max() < 1e-12


@pytest.mark.parametrize("name, rule", [("rep_r2", "channels"), ("surface_r3", "channels"), ("surface_r3", "gates")])
def test_fit_of_samples_is_close(built, name, rule):
    """A million shots put every row of D theta within 0.01 of the truth."""
    circuit, faults, design = built[name, rule]
    true, theta_fixed = true_thetas(faults, NOISE, design, rule)
    y = row_expectations(sample_detectors(circuit, 1_000_000, seed=1), design.subsets)
    assert np.abs(design.D @ (fit_thetas(design, y, theta_fixed) - true)).max() < 0.01


# --- rates_from_thetas -----------------------------------------------------------------

def test_rates_under_channels_by_hand():
    """Each formula undone, and a merged column's flips: 0.01 + 0.003 - 2 * 0.01 * 0.003."""
    groups = [["gate_2q"], ["idle"], ["measure", "reset"], ["readout"]]
    theta = np.array([np.log(1 - 16 * 0.005 / 15) / 8, np.log(1 - 4 * 0.002 / 3) / 2,
                      np.log(1 - 2 * 0.01) + np.log(1 - 2 * 0.003), np.log(1 - 2 * 0.02)])
    assert rates_from_thetas(small_design([[0]], groups=groups), theta) == pytest.approx([0.005, 0.002, 0.01294, 0.02])


def test_a_gates_parameter_gives_the_channel_rate():
    groups = [[("idle", -2, (0,))], [("gate_2q", -2, (0, 1))]]
    theta = np.array([np.log(1 - 4 * 0.002 / 3) / 2, np.log(1 - 16 * 0.005 / 15) / 8])
    assert rates_from_thetas(small_design([[0]], groups=groups), theta) == pytest.approx([0.002, 0.005])


@pytest.mark.parametrize("name", [("idle", -2, (0,), "X0"), ("gate_2q", -2, (0, 1), "X0*X1"),
                                  ("measure", 0, (1,), "X1"), ("readout", (0,), "X0")])
def test_one_pauli_or_flip_gives_its_own_probability(name):
    """A key ending in its Pauli: p = (1 - q) / 2, with no depolarizing formula."""
    assert rates_from_thetas(small_design([[0]], groups=[[name]]), np.array([np.log(1 - 2 * 0.001)])) == (
        pytest.approx([0.001]))


@pytest.mark.parametrize("names", [["idle", "gate_1q"], [("idle", -2, (0,)), ("gate_1q", -1, (0,))]])
def test_merged_depolarizing_gives_the_flip_probability(names):
    """A sum of thetas is no one channel's, even when the first of them is depolarizing."""
    assert rates_from_thetas(small_design([[0]], groups=[names]), np.array([np.log(1 - 2 * 0.001)])) == (
        pytest.approx([0.001]))


def test_a_column_in_a_direction_has_no_rate():
    groups = [["gate_1q"], ["measure", "reset"], ["readout"]]
    directions = [[(["gate_1q"], 1), (["measure", "reset"], -2)]]
    rates = rates_from_thetas(small_design([[0]], groups=groups, directions=directions), np.log([0.9, 0.9, 0.96]))
    assert np.isnan(rates[:2]).all()
    assert rates[2] == pytest.approx(0.02)


@pytest.mark.parametrize("name, rule, undetermined", [("rep_r2", "channels", 0), ("surface_r3", "gates", 8),
                                                      ("surface_r3", "rounds", 32), ("bb_r3", "rounds", 18),
                                                      ("bb_r3", "gates", 111)])
def test_number_of_columns_without_a_rate(built, name, rule, undetermined):
    _, _, design = built[name, rule]
    assert np.isnan(rates_from_thetas(design, np.zeros(len(design.groups)))).sum() == undetermined


@pytest.mark.parametrize("name, rule", CASES)
def test_rates_of_exact_data_are_the_true_rates(built, name, rule):
    """The whole chain: stim's exact y, the fit, and the rates, per kind of column."""
    circuit, faults, design = built[name, rule]
    true, theta_fixed = true_thetas(faults, NOISE, design, rule)
    rates = rates_from_thetas(design, fit_thetas(design, exact_expectations(circuit, design.subsets), theta_fixed))
    for rate, group, value in zip(rates, design.groups, true):
        channel = group[0] if isinstance(group[0], str) else group[0][0]
        if np.isnan(rate):
            continue
        if len(group) > 1:
            assert rate == pytest.approx((1 - np.exp(value)) / 2)
        elif rule == "rounds":
            assert rate == pytest.approx(fault_probability(fault(channel), NOISE))
        else:
            assert rate == pytest.approx(NOISE[channel])
