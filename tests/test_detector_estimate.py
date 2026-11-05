"""Tests for the true values: the rate of one fault, the true thetas, stim's exact
expectations, and that the model D theta + D_fixed theta_fixed reproduces them."""

import numpy as np
import pytest
import stim
from sympy.abc import x, y

from qecdm.detector.circuits import bb_memory, repetition_memory, surface_memory, uniform_noise
from qecdm.detector.detector_dmatrix import detector_design_matrix
from qecdm.detector.estimate import exact_expectations, fault_probability, true_thetas
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


def test_q_solves_the_defining_equations():
    """Each anticommuting independent fault gives a factor 1 - 2q; the channel gives 1 - 4p/3 or 1 - 16p/15."""
    p = 0.1
    assert (1 - 2 * fault_probability(fault("idle"), {"idle": p})) ** 2 == pytest.approx(1 - 4 * p / 3)
    assert (1 - 2 * fault_probability(fault("gate_2q"), {"gate_2q": p})) ** 8 == pytest.approx(1 - 16 * p / 15)


def test_one_qubit_q_matches_stim():
    """A Bell pair around the channel gives each of the 3 Paulis its own detector pattern."""
    p = 0.1
    circuit = stim.Circuit(f"""
        R 0 1
        H 0
        CNOT 0 1
        DEPOLARIZE1({p}) 0
        CNOT 0 1
        H 0
        M 0 1
        DETECTOR rec[-2]
        DETECTOR rec[-1]
    """)
    q = fault_probability(fault("idle"), {"idle": p})
    assert dem_probabilities(circuit) == pytest.approx([q] * 3)


def test_two_qubit_q_matches_stim():
    """Two Bell pairs give each of the 15 Paulis its own detector pattern.
    One distinct line in dem corresponds to each of the 15 2 qubit Paulis."""
    p = 0.1
    circuit = stim.Circuit(f"""
        R 0 1 2 3
        H 0 2
        CNOT 0 1 2 3
        DEPOLARIZE2({p}) 0 2
        CNOT 0 1 2 3
        H 0 2
        M 0 1 2 3
        DETECTOR rec[-4]
        DETECTOR rec[-3]
        DETECTOR rec[-2]
        DETECTOR rec[-1]
    """)
    q = fault_probability(fault("gate_2q"), {"gate_2q": p})
    assert dem_probabilities(circuit) == pytest.approx([q] * 15)


# --- true_thetas -----------------------------------------------------------------------

def test_true_thetas_by_hand(built):
    """Repetition under "channels": each column's theta, with measure and reset merged."""
    _, faults, design = built["rep_r2", "channels"]
    theta, theta_fixed = true_thetas(faults, NOISE, design, "channels")
    q_idle = fault_probability(fault("idle"), NOISE)
    assert theta[design.groups.index(["idle"])] == pytest.approx(np.log(1 - 2 * q_idle))
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
