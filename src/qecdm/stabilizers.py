import re

import numpy as np
import stim


# The Pauli each single-qubit measurement gate reads out.
MEASURED_PAULI = {"M": "Z", "MZ": "Z", "MR": "Z", "MRZ": "Z",
                  "MX": "X", "MRX": "X", "MY": "Y", "MRY": "Y"}


def stim_stabilizers(circuit, distance=None, rounds=None):
    """str (+ int, int) -> (np.ndarray (count, 2, n), list, list): stabilizers, qubits, coords.

    Pass a circuit's text, or a stim generated name like
    "surface_code:rotated_memory_z" with `distance` and `rounds`.
    Rows are the distinct stabilizers, in the order first measured. Column j
    is stim qubit `qubits[j]`, at `coords[j]` (None if the circuit gives none).
    Data qubits are those read out by the final measurement; every other
    measured qubit is an ancilla.
    """
    if distance is None:
        circuit = stim.Circuit(circuit)
    else:
        circuit = stim.Circuit.generated(circuit, distance=distance, rounds=rounds)
    circuit = circuit.without_noise().flattened()
    n = circuit.num_qubits

    # The last measurement reads out the data; the others measure ancillas.
    measurements = [(i, instruction) for i, instruction in enumerate(circuit)
                    if instruction.name in MEASURED_PAULI]

    data = sorted(t.value for t in measurements[-1][1].targets_copy())

    ancillas = {t.value for _, instruction in measurements[:-1]
                for t in instruction.targets_copy()} - set(data)

    is_ancilla = np.isin(np.arange(n), list(ancillas))

    # Round 1 starts at the first gate on an ancilla: data-only gates before
    # it are state preparation, not part of measuring the stabilizers.
    start = next(i for i, instruction in enumerate(circuit)
                 if stim.gate_data(instruction.name).is_unitary
                 and any(t.value in ancillas for t in instruction.targets_copy()))

    stabilizers = []
    for i, instruction in measurements[:-1]:
        for t in instruction.targets_copy():
            if t.value not in ancillas:
                continue
            measured = stim.PauliString(n)
            measured[t.value] = MEASURED_PAULI[instruction.name]
            # Trace back to the start of round 1. Earlier rounds' resets absorb
            # their ancilla parts, and stim raises if one doesn't fit its reset.
            x, z = measured.before(circuit[start:i]).to_numpy()
            # The same check for round 1: its ancillas' preparation must fix
            # whatever is left on them.
            leftover = stim.PauliString.from_numpy(xs=x & is_ancilla, zs=z & is_ancilla)
            leftover.before(circuit[:start])
            stabilizers.append([x[data], z[data]])

    # Later rounds repeat earlier ones: keep each stabilizer once, in order.
    stabilizers = np.array(stabilizers, dtype=int)
    _, first = np.unique(stabilizers, axis=0, return_index=True)

    # Where each column's qubit sits: its stim index, and its coordinates.
    coords = circuit.get_final_qubit_coordinates()
    return stabilizers[np.sort(first)], data, [coords.get(q) for q in data]


def stab_to_label(stabilizer):
    """np.ndarray of shape (2, n) -> str, e.g. [[1], [1]] -> "Y0".

    Names each non-identity Pauli with its qubit, skipping identities.
    """
    letters = {(1, 0): "X", (1, 1): "Y", (0, 1): "Z"}
    x, z = np.asarray(stabilizer)
    return "".join(f"{letters[(x[q], z[q])]}{q}" for q in np.flatnonzero(x | z))


def label_to_stab(label, n):
    """str, int -> np.ndarray of shape (2, n), e.g. ("Y0", 1) -> [[1], [1]].

    The reverse of stab_to_label; `n` is needed as the label omits identities.
    """
    stabilizer = np.zeros((2, n), dtype=int)
    for letter, q in re.findall(r"([XYZ])(\d+)", label):
        stabilizer[:, int(q)] = {"X": (1, 0), "Y": (1, 1), "Z": (0, 1)}[letter]
    return stabilizer

