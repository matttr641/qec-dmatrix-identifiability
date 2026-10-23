"""Tests for stim_stabilizers and the label helpers.

Known answers are written out in full, so any change in the output, including
the row or column order, fails a test.
"""

import numpy as np
import pytest
import stim

from cases import steane_41row, steane_generators
from qecdm.stabilizer import (
    gf2_rank, independent_stabilizers, label_to_stab, stab_to_label, stim_stabilizers,
)
from qecnoise.Dmatrixsinglequbit import commutators, design_matrix


# Every stim generated memory circuit, by name.
GENERATED = [
    "repetition_code:memory",
    "surface_code:rotated_memory_z",
    "surface_code:rotated_memory_x",
    "surface_code:unrotated_memory_z",
    "surface_code:unrotated_memory_x",
    "color_code:memory_xyz",
]


# --------------------------------------------------------------------------- #
# Known answers
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name, distance, labels, qubits", [
    ("surface_code:rotated_memory_z", 3,
     ["X0X1", "Z0Z1Z3Z4", "X1X2X4X5", "Z2Z5",
      "Z3Z6", "X3X4X6X7", "Z4Z5Z7Z8", "X7X8"],
     [1, 3, 5, 8, 10, 12, 15, 17, 19]),
    ("repetition_code:memory", 5,
     ["Z0Z1", "Z1Z2", "Z2Z3", "Z3Z4"],
     [0, 2, 4, 6, 8]),
])
def test_known_stabilizers(name, distance, labels, qubits):
    """Each stabilizer once, in measurement order, on the right stim qubits."""
    G, data, _ = stim_stabilizers(name, distance=distance, rounds=3)
    assert [stab_to_label(s) for s in G] == labels
    assert data == qubits

#at d = 3 the color code and the Steane code are the same, but stim cycles through
#mesuring each of the three Paulis on each of the 3 faces
def test_color_code_is_thesis_steane():
    """Relabelled, stim's d=3 color code gives the thesis Steane generators and D'."""
    G, _, _ = stim_stabilizers("color_code:memory_xyz", distance=3, rounds=3)
    assert [stab_to_label(s)[0] for s in G] == list("ZZZYYYXXX")
    #just checks the ordering of which Pauli is measured (Z first, then Y, then X)

    # Thesis qubit j is stim column [0, 2, 1, 6, 5, 4, 3][j]: stim's shared
    # qubit 3 is the thesis's qubit 6.
    G = G[:, :, [0, 2, 1, 6, 5, 4, 3]]
    # The X-faces then the Z-faces, in the order of the thesis's Hamming rows.
    G = G[[8, 6, 7, 2, 0, 1]]
    np.testing.assert_array_equal(G, steane_generators())

    D, _, _ = design_matrix(7, 6, G)
    assert D.shape == (41, 21)
    np.testing.assert_array_equal(D, steane_41row()["D"])


# --------------------------------------------------------------------------- #
# Properties of every generated code
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("distance", [3, 5, 7])
@pytest.mark.parametrize("name", GENERATED)
def test_stabilizers_commute(name, distance):
    """Every pair of stabilizers commutes."""
    G, _, _ = stim_stabilizers(name, distance=distance, rounds=3)
    assert not commutators(G, G).any()


@pytest.mark.parametrize("distance", [3, 5, 7])
@pytest.mark.parametrize("name", GENERATED)
def test_stabilizers_generate_whole_group(name, distance):
    """n qubits, one logical qubit: the rows span n - 1 independent generators."""
    G, _, _ = stim_stabilizers(name, distance=distance, rounds=3)
    n = G.shape[2]
    # Each row [x, z] laid out flat as one length-2n vector.
    assert gf2_rank(G.reshape(len(G), 2 * n)) == n - 1


@pytest.mark.parametrize("distance", [3, 5, 7])
@pytest.mark.parametrize("name", GENERATED)
def test_stabilizers_are_distinct(name, distance):
    """Repeat rounds are removed: no stabilizer appears twice."""
    G, _, _ = stim_stabilizers(name, distance=distance, rounds=3)
    assert len(np.unique(G, axis=0)) == len(G)


# --------------------------------------------------------------------------- #
# Consistency
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", GENERATED)
def test_name_and_text_agree(name):
    """A generated name gives the same answer as that circuit's text."""
    by_name = stim_stabilizers(name, distance=3, rounds=3)
    text = str(stim.Circuit.generated(name, distance=3, rounds=3))
    by_text = stim_stabilizers(text)
    np.testing.assert_array_equal(by_name[0], by_text[0]) #the stabilizers
    assert by_name[1:] == by_text[1:] #qubits and coords


@pytest.mark.parametrize("name", GENERATED)
def test_noise_does_not_change_stabilizers(name):
    """Noise channels are stripped before tracing back."""
    noisy = stim.Circuit.generated(
        name, distance=3, rounds=3,
        after_clifford_depolarization=0.01,
        before_measure_flip_probability=0.01,
        after_reset_flip_probability=0.01,
    )
    G_noisy, _, _ = stim_stabilizers(str(noisy))
    G_clean, _, _ = stim_stabilizers(name, distance=3, rounds=3)
    np.testing.assert_array_equal(G_noisy, G_clean)


@pytest.mark.parametrize("name", [n for n in GENERATED if n.startswith(("surface", "repetition"))])
def test_rounds_do_not_change_stabilizers(name):
    """Every round measures the same stabilizers, so more rounds add nothing.

    Not the color code: its stabilizers cycle Z, Y, X over the first 3 rounds.
    """
    G_short, _, _ = stim_stabilizers(name, distance=3, rounds=2)
    G_long, _, _ = stim_stabilizers(name, distance=3, rounds=5)
    np.testing.assert_array_equal(G_short, G_long)


def test_coordinates():
    """Coordinates come from QUBIT_COORDS, and are None when there are none."""
    _, _, coords = stim_stabilizers("surface_code:rotated_memory_z", distance=3, rounds=3)
    assert coords[0] == [1.0, 1.0]
    assert coords[-1] == [5.0, 5.0]

    _, _, coords = stim_stabilizers("R 0 1 2\nCX 0 1 2 1\nMR 1\nM 0 2")
    assert coords == [None, None]


# --------------------------------------------------------------------------- #
# Edge cases: small hand-written circuits
# --------------------------------------------------------------------------- #
# Each is a 3-qubit repetition code: data qubits 0, 2, 4 and ancillas 1, 3.
# Ancilla 1 checks data 0 and 2; ancilla 3 checks data 2 and 4.


def test_split_ancilla_measurement():
    """Ancillas measured one at a time (a TICK stops stim merging the two MRs)."""
    circuit = """
        R 0 1 2 3 4
        CX 0 1 2 1 2 3 4 3
        MR 1
        TICK
        MR 3
        M 0 2 4
    """
    G, _, _ = stim_stabilizers(circuit)
    assert [stab_to_label(s) for s in G] == ["Z0Z1", "Z1Z2"]


def test_ancilla_measured_in_x():
    """An ancilla reset and measured in X: the check is X-type, not Z-type."""
    circuit = """
        RX 1
        R 0 2
        CX 1 0 1 2
        MX 1
        M 0 2
    """
    G, _, _ = stim_stabilizers(circuit)
    assert [stab_to_label(s) for s in G] == ["X0X1"]


def test_data_preparation_is_skipped():
    """H on the data before round 1 is data state prep, so the checks stay Z-type."""
    circuit = """
        R 0 1 2 3 4
        H 0 2 4
        CX 0 1 2 1 2 3 4 3
        MR 1 3
        MX 0 2 4
    """
    G, _, _ = stim_stabilizers(circuit)
    assert [stab_to_label(s) for s in G] == ["Z0Z1", "Z1Z2"]


def test_bad_ancilla_raises():
    """The ancilla is put in |+> but measured in Z, so it measures nothing."""
    circuit = """
        R 0 1 2
        H 1
        CX 1 0
        MR 1
        M 0
    """
    with pytest.raises(ValueError):
        stim_stabilizers(circuit)


# --------------------------------------------------------------------------- #
# Label helpers
# --------------------------------------------------------------------------- #


def test_stab_to_label():
    assert stab_to_label([[1], [0]]) == "X0"
    assert stab_to_label([[1], [1]]) == "Y0"
    assert stab_to_label([[0], [1]]) == "Z0"
    assert stab_to_label([[1, 0, 0], [0, 0, 1]]) == "X0Z2"  # identity skipped
    assert stab_to_label([[0, 0], [0, 0]]) == ""            # all identity


def test_label_to_stab():
    np.testing.assert_array_equal(label_to_stab("Y0", 1), [[1], [1]])
    np.testing.assert_array_equal(label_to_stab("X0Z2", 3), [[1, 0, 0], [0, 0, 1]])
    np.testing.assert_array_equal(label_to_stab("", 2), [[0, 0], [0, 0]])


def test_label_to_stab_multi_digit_qubits():
    """Qubit numbers of more than one digit, e.g. 10 and 11."""
    stabilizer = label_to_stab("X10Z11", 12)
    np.testing.assert_array_equal(stabilizer[:, 10:], [[1, 0], [0, 1]])
    assert not stabilizer[:, :10].any()   # every other qubit is identity


def test_labels_round_trip():
    """label_to_stab undoes stab_to_label, for every stabilizer of a d=5 code."""
    G, _, _ = stim_stabilizers("color_code:memory_xyz", distance=5, rounds=3)
    n = G.shape[2]
    for s in G:
        np.testing.assert_array_equal(label_to_stab(stab_to_label(s), n), s)


# --------------------------------------------------------------------------- #
# Independent generating sets
# --------------------------------------------------------------------------- #


def test_independent_drops_products_and_identity():
    """Z0Z2 = Z0Z1 * Z1Z2 and the identity add nothing, so both are dropped."""
    G = [label_to_stab(label, 4) for label in ["Z0Z1", "Z1Z2", "Z0Z2", "", "Z2Z3"]]
    kept = independent_stabilizers(G)
    assert [stab_to_label(s) for s in kept] == ["Z0Z1", "Z1Z2", "Z2Z3"]


def test_independent_keeps_already_independent_set():
    """The surface code's stabilizers are already independent: all are kept."""
    G, _, _ = stim_stabilizers("surface_code:rotated_memory_z", distance=5, rounds=3)
    np.testing.assert_array_equal(independent_stabilizers(G), G)


def test_independent_color_code_gives_thesis_shape():
    """Of the color code's Z, Y, X faces, the first listed win: Z and Y faces.

    Any 6 independent faces generate the Steane group, so D' has the thesis shape.
    """
    G, _, _ = stim_stabilizers("color_code:memory_xyz", distance=3, rounds=3)
    kept = independent_stabilizers(G)
    np.testing.assert_array_equal(kept, G[:6])  # the 3 Z faces, then the 3 Y faces

    D, _, _ = design_matrix(7, 6, kept)
    assert D.shape == (41, 21)
