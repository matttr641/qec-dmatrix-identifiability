import re

import numpy as np
import stim
from qldpc import codes
from sympy.abc import x, y


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


# The rank over GF(2); np.linalg.matrix_rank works over the reals instead.
def gf2_rank(M):
    """np.ndarray (rows, columns) of 0/1 -> int: the rank with arithmetic mod 2."""
    M = np.array(M, dtype=int) % 2
    rank = 0
    for column in range(M.shape[1]):
        pivots = np.flatnonzero(M[rank:, column])
        if len(pivots) == 0:
            continue
        M[[rank, rank + pivots[0]]] = M[[rank + pivots[0], rank]]
        below_or_above = np.flatnonzero(M[:, column])
        M[below_or_above[below_or_above != rank]] ^= M[rank]
        rank += 1
        if rank == len(M):
            break
    return rank


def independent_stabilizers(stabilizers):
    """np.ndarray (k, 2, n) -> np.ndarray (m, 2, n): an independent generating set.

    Keeps each stabilizer that isn't a product of the ones kept before it, so
    the first listed wins: stim's color code output (Z, Y, X faces) keeps the
    Z and Y faces. The kept set generates the same group as the input.
    """
    G = np.asarray(stabilizers)
    flat = G.reshape(len(G), -1)  # each [x, z] row laid out as one vector
    kept = []
    for i in range(len(G)):
        # Independent if adding it raises the rank above the number kept.
        if gf2_rank(flat[kept + [i]]) > len(kept):
            kept.append(i)
    return G[kept]


def qldpc_stabilizers(code):
    """qldpc code -> (np.ndarray (count, 2, n), list, list): stabilizers, qubits, coords.

    Pass an already-built code, e.g. codes.SurfaceCode(5), codes.SteaneCode() or
    codes.BBCode({x: 6, y: 6}, x**3 + y + y**2, y**3 + x + x**2), so no
    code-specific arguments are needed here. The rows are qldpc's checks as
    given, which may be redundant (the [[72,12,6]] BB code lists 72 of rank 60): pass them
    through independent_stabilizers before design_matrix. The output matches
    stim_stabilizers; qldpc gives no coordinates, so coords are all None.
    """
    if code.field.order != 2:
        raise ValueError(f"only qubit codes are supported, not GF({code.field.order})")
    n = code.num_qudits
    # qldpc's parity-check matrix is [X part | Z part], one check per row.
    H = np.asarray(code.matrix, dtype=int)
    G = np.stack([H[:, :n], H[:, n:]], axis=1)  # each check as [x, z]
    return G, list(range(n)), [None] * n