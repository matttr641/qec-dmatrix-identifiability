"""Tests for the exact Gordan programme.

The thesis cases are checked against the saved answer key in tests/fixtures
and the verdicts of the thesis (Chapters 8 and 9).
"""

from pathlib import Path

import numpy as np
import pytest

from cases import chain_11row, chain_21row, steane_41row
from qecdm.gordanLP import gordan, left_kernel, straddles
from qecnoise.Dmatrixsinglequbit import pair_family, two_qubit_family


FIXTURES = Path(__file__).parent / "fixtures"


# --------------------------------------------------------------------------- #
# Thesis cases
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("case, family, certified, n_absorbed, n_resolved, kappa", [
    # Bit-flip code, 11 rows: ten pairs hide alone, the rest can conspire.
    (chain_11row, pair_family(7)[0], False, 10, 11, 4),
    # Bit-flip code, 21 rows: nothing hides alone, but groups of six can.
    (chain_21row, pair_family(7)[0], False, 0, 21, 14),
    # Steane: a third of the two-qubit errors absorbed, the rest certified safe.
    (steane_41row, two_qubit_family(7)[0], True, 63, 126, 20),
], ids=["chain_11row", "chain_21row", "steane_41row"])
def test_thesis_verdicts(case, family, certified, n_absorbed, n_resolved, kappa):
    """The absorbed set matches the saved fixture, and the verdict the thesis's."""
    saved = np.load(FIXTURES / f"{case.__name__}.npz")
    result = case()
    D, measured = result["D"], result["measured"]

    got_certified, absorbed, resolved, witness = gordan(D, measured, family)

    assert absorbed == list(saved["absorbed"])
    assert (len(absorbed), len(resolved)) == (n_absorbed, n_resolved)
    assert left_kernel(D).shape[1] == kappa  # kappa = rows - rank(D)
    assert got_certified is certified
    assert (witness is not None) == certified


# --------------------------------------------------------------------------- #
# The witness
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def steane():
    """Steane's D', measured set and the gordan result, computed once for the module."""
    result = steane_41row()
    family = two_qubit_family(7)[0]
    certified, _, resolved, witness = gordan(result["D"], result["measured"], family)
    assert certified  # the tests below need a witness
    return result["D"], result["measured"], family, resolved, witness


def test_witness_is_in_kernel_exactly(steane):
    """D^T lambda = 0 with no rounding: every entry is exactly zero."""
    D, _, _, _, witness = steane
    assert all(entry == 0 for entry in np.asarray(D, dtype=object).T @ witness)


def test_witness_is_integer(steane):
    """Every entry of lambda is a Python int, so the products below are exact."""
    _, _, _, _, witness = steane
    assert all(type(entry) is int for entry in witness)


def test_witness_is_positive_on_every_resolved_straddle(steane):
    """<lambda, s^P> > 0 for every resolved P, from the straddles themselves.

    This is branch (i) of Theorem 6.5 in the thesis's own terms, not through
    the kernel coordinates the programme was solved in.
    """
    _, measured, family, resolved, witness = steane
    S = np.asarray(straddles(measured, family), dtype=object)
    products = witness @ S[:, resolved]
    assert all(product > 0 for product in products)
