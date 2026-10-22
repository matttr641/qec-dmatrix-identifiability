"""Tests for the D-matrix builder and its helpers.

The thesis cases are checked against the saved answer key in tests/fixtures,
so the new builder must reproduce the legacy results exactly.
"""

from pathlib import Path

import numpy as np
import pytest

from cases import chain_11row, chain_21row, steane_41row, steane_generators
from qecdm.dmatrix import design_matrix
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
