"""Tests for the D-matrix builder and its helpers.

The thesis cases are checked against the saved answer key in tests/fixtures,
so the new builder must reproduce the legacy results exactly.
"""

from pathlib import Path

import numpy as np
import pytest

from cases import chain_11row, chain_21row, steane_41row, steane_generators
from qecdm.dmatrix import design_matrix
from qecdm.stabilizer import independent_stabilizers, stim_stabilizers
from qecnoise.Dmatrixsinglequbit import bit_flip_chain


FIXTURES = Path(__file__).parent / "fixtures"


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
