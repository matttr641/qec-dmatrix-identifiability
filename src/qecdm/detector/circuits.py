"""Noisy memory-experiment circuits for the detector D': repetition, surface and BB codes.

Every builder takes the same noise dict (see `uniform_noise`) and returns a
flattened stim circuit, so the rest of the detector code never needs to know
which generator made it. The repetition and surface codes come from stim's
generated circuits; BB codes come from qldpc's get_memory_experiment.

Only which faults exist matters for building D', not their rates, so every
rate should be > 0: a zero-rate channel has no faults and so no columns.
"""

import stim
from qldpc import circuits, codes

NOISE_KEYS = ("idle", "gate_1q", "gate_2q", "measure", "reset", "readout")
MEASUREMENTS = {"M", "MX", "MY", "MR", "MRX", "MRY"}
FLIPS = {"X_ERROR", "Y_ERROR", "Z_ERROR"}


def uniform_noise(p, **rates):
    """float -> dict: every noise rate p, except any given by name, e.g. readout=0.02.

    - idle: idling depolarization (stim: on the data qubits before each round;
      qldpc: on every idle qubit in every moment);
    - gate_1q, gate_2q: depolarization after one- and two-qubit Cliffords;
    - measure: flip of each mid-circuit (ancilla) measurement;
    - reset: flip after each reset, including the data qubits' preparation;
    - readout: flip of the final data-qubit measurement.
    """
    noise = dict.fromkeys(NOISE_KEYS, p)
    unknown = set(rates) - set(NOISE_KEYS)
    if unknown:
        raise ValueError(f"unknown noise rates {sorted(unknown)}: use {NOISE_KEYS}")
    for key, value in rates.items():
        noise[key] = value
    return noise


def set_readout_error(circuit, p):
    """stim.Circuit, float -> stim.Circuit: the final data measurement's flip rate set to p.

    The generators give the readout the same rate as the ancilla measurements.
    The final measurement is the last measurement in the circuit. Its error is
    either a flip just before it (stim: X_ERROR before M) or its own argument
    (qldpc: M(p)); whichever it has is set to p.
    """
    circuit = circuit.flattened()
    last = max(i for i, ins in enumerate(circuit) if ins.name in MEASUREMENTS)
    readout, before = circuit[last], circuit[last - 1]
    flip_before = before.name in FLIPS and before.targets_copy() == readout.targets_copy()

    out = stim.Circuit()
    for i, ins in enumerate(circuit):
        if i == last - 1 and flip_before:
            out.append(ins.name, ins.targets_copy(), p)
        elif i == last and ins.gate_args_copy():
            out.append(ins.name, ins.targets_copy(), p)
        else:
            out.append(ins)
    return out


def _stim_memory(name, distance, rounds, noise):
    """The shared part of the stim builders."""
    if noise["gate_1q"] != noise["gate_2q"]:
        raise ValueError("stim's generated circuits use one rate for all Clifford gates: "
                         "gate_1q must equal gate_2q")
    circuit = stim.Circuit.generated(
        name, distance=distance, rounds=rounds,
        before_round_data_depolarization=noise["idle"],
        after_clifford_depolarization=noise["gate_2q"],
        before_measure_flip_probability=noise["measure"],
        after_reset_flip_probability=noise["reset"])
    return set_readout_error(circuit, noise["readout"])


def repetition_memory(distance, rounds, noise):
    """int, int, dict -> stim.Circuit: stim's repetition-code memory (Z basis), flattened."""
    return _stim_memory("repetition_code:memory", distance, rounds, noise)


def surface_memory(distance, rounds, noise, basis="z"):
    """int, int, dict -> stim.Circuit: stim's rotated surface-code memory, flattened."""
    if basis.lower() not in ("x", "z"):
        raise ValueError(f'basis must be "x" or "z", not {basis!r}')
    return _stim_memory(f"surface_code:rotated_memory_{basis.lower()}", distance, rounds, noise)


def qldpc_memory(code, rounds, noise, basis="z"):
    """qldpc code, int, dict -> stim.Circuit: qldpc's memory experiment for any of its codes, flattened."""
    model = circuits.NoiseModel(
        clifford_1q_error=noise["gate_1q"],
        clifford_2q_error=noise["gate_2q"],
        readout_error=noise["measure"],  # qldpc uses one rate for every measurement
        reset_error=noise["reset"],
        idle_error=noise["idle"])
    circuit = circuits.get_memory_experiment(code, basis=basis.upper(), num_rounds=rounds,
                                             noise_model=model)
    return set_readout_error(circuit, noise["readout"])


def bb_memory(orders, poly_a, poly_b, rounds, noise, basis="z"):
    """orders, A, B, int, dict -> stim.Circuit: a BB code's memory experiment, flattened.

    Same arguments as codes.BBCode, e.g. ({x: 6, y: 6}, x**3 + y + y**2, y**3 + x + x**2).
    """
    return qldpc_memory(codes.BBCode(orders, poly_a, poly_b), rounds, noise, basis)
