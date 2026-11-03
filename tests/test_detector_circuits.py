"""Tests for the detector circuits: the noise dict, the readout rewrite, and that each
builder puts every rate on the instructions it should."""

import pytest
import stim
from qldpc import codes
from sympy.abc import x, y

from qecdm.detector.circuits import (NOISE_KEYS, bb_memory, qldpc_memory, repetition_memory,
                                     set_readout_error, surface_memory, uniform_noise)

BB = ({x: 2, y: 3}, 1 + y + x * y**2, 1 + y + x * y**2)  # [[12,4]], the small certified BB code

# Every rate different, so each can be traced to the instructions it lands on.
# stim needs gate_1q == gate_2q; qldpc does not.
DISTINCT = dict(idle=0.001, gate_1q=0.002, gate_2q=0.002, measure=0.003, reset=0.004, readout=0.005)

BUILDERS = [
    pytest.param(lambda noise: repetition_memory(3, 2, noise), 5, 6, id="repetition_d3"),
    pytest.param(lambda noise: surface_memory(3, 2, noise), 26, 16, id="surface_d3"),
    pytest.param(lambda noise: bb_memory(*BB, 2, noise), 24, 18, id="bb_12_4"),
]


def noise_args(circuit):
    """stim.Circuit -> dict: each noisy instruction's name -> the set of rates it uses."""
    found = {}
    for ins in circuit:
        if ins.name not in ("DETECTOR", "OBSERVABLE_INCLUDE", "QUBIT_COORDS", "SHIFT_COORDS") \
                and ins.gate_args_copy():
            found.setdefault(ins.name, set()).update(ins.gate_args_copy())
    return found


# --- uniform_noise ---------------------------------------------------------------

def test_uniform_noise_sets_every_rate():
    assert uniform_noise(0.01) == dict.fromkeys(NOISE_KEYS, 0.01)


def test_uniform_noise_overrides_named_rates():
    noise = uniform_noise(0.01, readout=0.02, reset=0.005)
    assert noise == dict(idle=0.01, gate_1q=0.01, gate_2q=0.01, measure=0.01, reset=0.005, readout=0.02)


def test_uniform_noise_rejects_unknown_rates():
    with pytest.raises(ValueError, match="unknown noise rates"):
        uniform_noise(0.01, cnot=0.1)


# --- set_readout_error ------------------------------------------------------------

def test_readout_changes_only_the_final_flip_in_stim():
    """The X_ERROR before each ancilla MR keeps the measure rate; only the one before
    the final M changes."""
    circuit = set_readout_error(repetition_memory(3, 2, uniform_noise(0.01)), 0.05)
    flips = [(circuit[i - 1].gate_args_copy(), ins.name) for i, ins in enumerate(circuit)
             if ins.name in ("M", "MR")]
    assert flips == [([0.01], "MR"), ([0.01], "MR"), ([0.05], "M")]


def test_readout_changes_only_the_final_measurement_in_qldpc():
    """qldpc's errors are the measurements' own arguments: the ancilla MX(p) keep the
    measure rate, the final data M gets the readout rate."""
    circuit = set_readout_error(bb_memory(*BB, 2, uniform_noise(0.01)), 0.05)
    measurements = [(ins.name, ins.gate_args_copy()) for ins in circuit if ins.name in ("M", "MX")]
    assert measurements == [("MX", [0.01]), ("MX", [0.01]), ("M", [0.05])]


def test_readout_equal_to_measure_changes_nothing():
    generated = stim.Circuit.generated(
        "repetition_code:memory", distance=3, rounds=2, before_round_data_depolarization=0.01,
        after_clifford_depolarization=0.01, before_measure_flip_probability=0.01,
        after_reset_flip_probability=0.01)
    assert repetition_memory(3, 2, uniform_noise(0.01)) == generated.flattened()


@pytest.mark.parametrize("build, qubits, detectors", BUILDERS)
def test_readout_rewrite_keeps_the_circuit_structure(build, qubits, detectors):
    """Only a rate changes: the noiseless circuit (gates, detectors, observables) is identical."""
    circuit = build(uniform_noise(0.01))
    assert set_readout_error(circuit, 0.05).without_noise() == circuit.without_noise()


# --- the builders ------------------------------------------------------------------

@pytest.mark.parametrize("build, qubits, detectors", BUILDERS)
def test_builders_are_flat_with_known_sizes(build, qubits, detectors):
    circuit = build(uniform_noise(0.01))
    assert "REPEAT" not in str(circuit)
    assert (circuit.num_qubits, circuit.num_detectors) == (qubits, detectors)


def test_repetition_code_has_the_15_known_columns():
    """Circuit-level + reset noise, d = 3, 2 rounds: 15 distinct detector patterns."""
    assert repetition_memory(3, 2, uniform_noise(0.01)).detector_error_model().num_errors == 15


def test_stim_builders_put_each_rate_in_place():
    """idle -> DEPOLARIZE1 on data, gate_2q -> DEPOLARIZE2 (and DEPOLARIZE1 after the
    surface code's H), measure/reset/readout -> X_ERROR flips."""
    assert noise_args(repetition_memory(3, 2, DISTINCT)) == {
        "DEPOLARIZE1": {0.001}, "DEPOLARIZE2": {0.002}, "X_ERROR": {0.003, 0.004, 0.005}}
    assert noise_args(surface_memory(3, 2, DISTINCT)) == {
        "DEPOLARIZE1": {0.001, 0.002}, "DEPOLARIZE2": {0.002}, "X_ERROR": {0.003, 0.004, 0.005}}


def test_qldpc_builder_puts_each_rate_in_place():
    """reset -> X_ERROR after R and Z_ERROR after RX; measure -> MX(p); readout -> M(p).
    This BB circuit has no one-qubit gates, so gate_1q never appears."""
    noise = dict(DISTINCT, gate_1q=0.0015)
    assert noise_args(bb_memory(*BB, 2, noise)) == {
        "DEPOLARIZE1": {0.001}, "DEPOLARIZE2": {0.002}, "X_ERROR": {0.004}, "Z_ERROR": {0.004},
        "MX": {0.003}, "M": {0.005}}


def test_stim_rejects_different_gate_rates():
    with pytest.raises(ValueError, match="gate_1q must equal gate_2q"):
        surface_memory(3, 2, uniform_noise(0.01, gate_1q=0.001))


def test_surface_basis():
    assert surface_memory(3, 2, uniform_noise(0.01), basis="X").num_detectors == 16
    with pytest.raises(ValueError, match='basis must be "x" or "z"'):
        surface_memory(3, 2, uniform_noise(0.01), basis="y")


def test_bb_memory_is_qldpc_memory_of_the_bb_code():
    noise = uniform_noise(0.01)
    assert bb_memory(*BB, 2, noise) == qldpc_memory(codes.BBCode(*BB), 2, noise)
