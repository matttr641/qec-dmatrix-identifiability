"""Tests for the D-matrix builder and its helpers.

The thesis cases are checked against the saved answer key in tests/fixtures,
so the new builder must reproduce the legacy results exactly.
"""

from pathlib import Path

import numpy as np
import pytest

from cases import chain_11row, chain_21row, steane_41row, steane_generators
from qecdm.dmatrix import (
    commutators, connected, correlation_graph, design_matrix, pauli_errors, stab_products,
)
from qecdm.stabilizer import independent_stabilizers, label_to_stab, stab_to_label, stim_stabilizers
from qecnoise.Dmatrixsinglequbit import bit_flip_chain, two_qubit_family


FIXTURES = Path(__file__).parent / "fixtures"

# Inputs for the property tests. The color code's 9 rows are redundant, so
# different subsets make the same product: that exercises the duplicate remover.
GENERATORS = {
    "chain": bit_flip_chain(7),
    "steane": steane_generators(),
    "surface_d3": stim_stabilizers("surface_code:rotated_memory_z", distance=3, rounds=3)[0],
    "color_d3_all_9": stim_stabilizers("color_code:memory_xyz", distance=3, rounds=3)[0],
}


# --------------------------------------------------------------------------- #
# Thesis fixtures
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("case, stabilizers, max_order", [
    (chain_11row, bit_flip_chain(7), None),   # stops when rank(D') = |C*|
    (chain_21row, bit_flip_chain(7), 6),      # walks every order
    (steane_41row, steane_generators(), None),
], ids=["chain_11row", "chain_21row", "steane_41row"])
def test_matches_thesis_fixture(case, stabilizers, max_order):
    """D', the measured set and the merged columns are exactly the thesis's."""
    saved = np.load(FIXTURES / f"{case.__name__}.npz")
    D, merged, measured, _ = design_matrix(stabilizers, max_order=max_order)

    np.testing.assert_array_equal(D, saved["D"])
    np.testing.assert_array_equal(measured, saved["measured"])
    # The fixtures don't save the column labels, so compare with the legacy run.
    assert merged == case()["merged"]


# --------------------------------------------------------------------------- #
# From stim circuits
# --------------------------------------------------------------------------- #


def test_color_code_d3_gives_thesis_steane_shape():
    """The d=3 color code is the Steane code: D' is 41 x 21 with kappa = 20.

    Stim measures Z, Y and X faces; independent_stabilizers keeps the Z and Y
    faces, which generate the same group as the thesis's X and Z faces.
    """
    G, _, _ = stim_stabilizers("color_code:memory_xyz", distance=3, rounds=3)
    D, _, _, _ = design_matrix(independent_stabilizers(G))

    assert D.shape == (41, 21)
    assert D.shape[0] - np.linalg.matrix_rank(D) == 20  # kappa = rows - rank


def test_rotated_surface_code_d3():
    """No thesis fixture, so check what can be worked out by hand.

    Two single-qubit errors share a column when their product is a stabilizer.
    The four weight-2 boundary stabilizers X0X1, X7X8, Z2Z5 and Z3Z6 each merge
    one pair, so the 27 single-qubit errors fall into 27 - 4 = 23 columns.
    """
    G, _, _ = stim_stabilizers("surface_code:rotated_memory_z", distance=3, rounds=3)
    D, merged, _, _ = design_matrix(G)

    assert len(merged) == 23
    assert [group for group in merged if len(group) > 1] == [
        ("X0", "X1"), ("Z2", "Z5"), ("Z3", "Z6"), ("X7", "X8"),
    ]
    assert np.linalg.matrix_rank(D) == 23           # the walk reached |C*|
    assert np.unique(D, axis=1).shape[1] == D.shape[1]  # every column is distinct


# --------------------------------------------------------------------------- #
# Properties of D'
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", GENERATORS)
def test_rows_match_direct_commutation(name):
    """Each row, built as the sum of its generators' rows, equals commuting the
    measured stabilizer with the errors directly, for every error in each column.
    """
    G = np.asarray(GENERATORS[name])
    D, merged, measured, _ = design_matrix(G)
    n = G.shape[2]
    for column, group in enumerate(merged):
        errors = [label_to_stab(label, n) for label in group]
        direct = commutators(measured, errors)       # (rows, errors in this column)
        for e in range(len(group)):
            np.testing.assert_array_equal(direct[:, e], D[:, column])


@pytest.mark.parametrize("name", GENERATORS)
def test_subsets_multiply_to_measured(name):
    """Multiplying each row's generators together gives that row's stabilizer."""
    G = np.asarray(GENERATORS[name])
    _, _, measured, subsets = design_matrix(G)
    assert len(subsets) == len(measured)
    for stabilizer, subset in zip(measured, subsets):
        np.testing.assert_array_equal(G[list(subset)].sum(axis=0) % 2, stabilizer)


@pytest.mark.parametrize("name", GENERATORS)
def test_no_repeated_measured_stabilizers(name):
    """The duplicate remover leaves each measured stabilizer once."""
    _, _, measured, _ = design_matrix(GENERATORS[name])
    assert len(np.unique(measured, axis=0)) == len(measured)


@pytest.mark.parametrize("name", GENERATORS)
def test_merged_holds_each_detectable_error_once(name):
    """Every detectable single-qubit error is in exactly one column, and no
    undetectable error (one that commutes with every generator) is in any.
    """
    G = np.asarray(GENERATORS[name])
    _, merged, _, _ = design_matrix(G)
    errors, labels = pauli_errors(G.shape[2], 1)
    detectable = [label for label, syndrome in zip(labels, commutators(G, errors).T)
                  if syndrome.any()]

    in_columns = [label for group in merged for label in group]
    assert sorted(in_columns) == sorted(detectable)   # each once, nothing extra


def test_chain_z_errors_are_undetectable():
    """The bit-flip chain only has Z checks, so no Z error gets a column."""
    _, merged, _, _ = design_matrix(bit_flip_chain(7))
    assert not any(label.startswith("Z") for group in merged for label in group)


# --------------------------------------------------------------------------- #
# Two-qubit errors
# --------------------------------------------------------------------------- #


def test_steane_with_two_qubit_errors():
    """weights=(1, 2): single- and two-qubit errors share columns by syndrome.

    The Steane code has 6 generators, so 63 non-trivial syndromes, and every
    one is reached: D' is 63 x 63 at full rank.
    """
    D, merged, _, _ = design_matrix(steane_generators(), weights=(1, 2))

    assert D.shape == (63, 63)
    assert np.linalg.matrix_rank(D) == 63
    # X0 shares a syndrome with each pair whose product with it commutes with
    # every stabilizer: here X0X1X2, X0X3X4, X0X5X6, weight-3 logical operators.
    assert merged[0] == ("X0", "X1X2", "X3X4", "X5X6")

    # Single-qubit errors are listed first, so they represent their column.
    def weight(label):
        return sum(letter in "XYZ" for letter in label)
    for group in merged:
        if any(weight(label) == 1 for label in group):
            assert weight(group[0]) == 1


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def stabs(labels, n):
    """Build an array of stabilizers from labelS, e.g. stabs(["Z0", "X1"], 2)."""
    return np.array([label_to_stab(label, n) for label in labels])


def test_pauli_errors_single_qubit():
    """3n errors, ordered X0, Y0, Z0, X1, ... as in the legacy code."""
    errors, labels = pauli_errors(3, 1)
    assert labels == ["X0", "Y0", "Z0", "X1", "Y1", "Z1", "X2", "Y2", "Z2"]
    assert [stab_to_label(e) for e in errors] == labels


def test_pauli_errors_two_qubit_matches_legacy():
    """9 * C(n, 2) errors, in exactly the legacy two_qubit_family order."""
    errors, labels = pauli_errors(7, 2)
    legacy_errors, legacy_labels = two_qubit_family(7)
    assert labels == legacy_labels
    np.testing.assert_array_equal(errors, legacy_errors)


def test_commutators():
    """1 where an odd number of qubits anticommute, 0 otherwise."""
    A = stabs(["X0", "X0", "X0Z1", "Y0"], 2)
    B = stabs(["Z0", "X0", "Z0X1", "Z0"], 2)
    pairs = np.diag(commutators(A, B))  # A[i] against B[i]
    # X vs Z: anticommute | same Pauli: commute | two anticommuting qubits
    # cancel: commute | Y vs Z: anticommute
    assert pairs.tolist() == [1, 0, 0, 1]


def test_correlation_graph():
    """Joined when sharing a qubit, whatever the Paulis; identity joins nothing."""
    graph = correlation_graph(stabs(["Z0", "X0", "Z1Z2", "Y2", ""], 3))
    assert graph.astype(int).tolist() == [
        [1, 1, 0, 0, 0],   # Z0 and X0 share qubit 0
        [1, 1, 0, 0, 0],
        [0, 0, 1, 1, 0],   # Z1Z2 and Y2 share qubit 2
        [0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0],   # the identity: not even joined to itself
    ]


def test_connected():
    """On the chain's line graph g0 - g1 - g2 - ..., a gap disconnects."""
    graph = correlation_graph(bit_flip_chain(7))
    assert connected(graph, (0, 1, 2))
    assert connected(graph, (3,))
    assert not connected(graph, (0, 2))      # g1 would link them, but isn't in the subset
    assert not connected(graph, (0, 1, 3))


def test_stab_products_order_2():
    """Neighbouring chain generators multiply: Z0Z1 * Z1Z2 = Z0Z2, and so on."""
    products, subsets = stab_products(bit_flip_chain(7), 2)
    assert [stab_to_label(p) for p in products] == ["Z0Z2", "Z1Z3", "Z2Z4", "Z3Z5", "Z4Z6"]
    assert subsets == [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]


def test_stab_products_order_above_k_is_empty():
    """Order of product > no. stabilizers: no products, but still shaped (0, 2, n)."""
    products, subsets = stab_products(bit_flip_chain(7), 7)
    assert products.shape == (0, 2, 7)
    assert subsets == []


def test_stab_products_drops_identity():
    """Z0Z1 * Z1Z2 * Z0Z2 is the identity, so that subset makes no product."""
    G = stabs(["Z0Z1", "Z1Z2", "Z0Z2"], 3)
    products, subsets = stab_products(G, 3)
    assert len(products) == 0 and subsets == []


def test_stab_products_custom_graph():
    """An extra edge g0 - g5 lets that otherwise disconnected pair multiply."""
    G = bit_flip_chain(7)
    graph = correlation_graph(G)
    graph[0, 5] = graph[5, 0] = True
    products, subsets = stab_products(G, 2, graph)
    assert (0, 5) in subsets
    assert stab_to_label(products[subsets.index((0, 5))]) == "Z0Z1Z5Z6"


# --------------------------------------------------------------------------- #
# Errors raised
# --------------------------------------------------------------------------- #


def test_identity_stabilizers_raise():
    """Identity stabilizers commute with every error, so nothing is detectable."""
    with pytest.raises(ValueError, match="no error is detectable"):
        design_matrix(stabs(["", ""], 3))
