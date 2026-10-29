"""Tests for the exact Gordan programme.

The thesis cases are checked against the saved answer key in tests/fixtures
and the verdicts of the thesis (Chapters 8 and 9).
"""

from math import gcd
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import qecdm.gordanLP
from cases import chain_11row, chain_21row, steane_41row
from qecdm.gordanLP import (
    gordan, gordan_lp, left_kernel, rref, split_absorbed, split_shadows, straddles,
)
from qecdm.stabilizer import label_to_stab
from qecnoise.Dmatrixsinglequbit import pair_family, two_qubit_family
from qecnoise.gordanLP import shadows as legacy_shadows


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


# --------------------------------------------------------------------------- #
# left_kernel
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ["chain_11row", "chain_21row", "steane_41row"])
def test_left_kernel_properties(name):
    """On each thesis D': D^T K = 0 exactly, K has full column rank kappa, and
    each column is in lowest terms.
    """
    D = np.load(FIXTURES / f"{name}.npz")["D"]
    K = left_kernel(D)
    kappa = D.shape[0] - np.linalg.matrix_rank(D)

    assert K.shape == (D.shape[0], kappa)
    assert not (np.asarray(D, dtype=object).T @ K != 0).any()   # exactly zero
    assert np.linalg.matrix_rank(K.astype(float)) == kappa         # independent columns
    for column in K.T:
        assert gcd(*column) == 1                                   # smallest integers


def test_left_kernel_four_qubit_chain():
    """Rows Z0Z1, Z1Z2, Z2Z3, Z0Z2, Z1Z3: the one redundancy is
    -Z0Z1 - Z2Z3 + Z0Z2 + Z1Z3 = 0, since Z0Z1.Z2Z3 = Z0Z2.Z1Z3.
    """
    D = np.array([[1, 1, 0, 0],
                  [0, 1, 1, 0],
                  [0, 0, 1, 1],
                  [1, 0, 1, 0],
                  [0, 1, 0, 1]])
    assert left_kernel(D).tolist() == [[-1], [0], [-1], [1], [1]]


def test_left_kernel_empty_when_full_rank():
    """A full-rank square D has no left kernel: K has no columns."""
    K = left_kernel(np.eye(3, dtype=int))
    assert K.shape == (3, 0)


# --------------------------------------------------------------------------- #
# split_shadows
# --------------------------------------------------------------------------- #


def test_split_shadows_four_qubit_chain():
    """The six X pairs on the four-qubit chain: X0X3 and X1X2 look exactly like
    two independent single flips (straddle = X0 column + X3 column), so they
    are absorbed; the other four pairs are resolved.
    """
    D = np.array([[1, 1, 0, 0],
                  [0, 1, 1, 0],
                  [0, 0, 1, 1],
                  [1, 0, 1, 0],
                  [0, 1, 0, 1]])
    measured = np.array([label_to_stab(label, 4)
                         for label in ["Z0Z1", "Z1Z2", "Z2Z3", "Z0Z2", "Z1Z3"]])
    family = np.array([label_to_stab(label, 4)
                       for label in ["X0X1", "X0X2", "X0X3", "X1X2", "X1X3", "X2X3"]])

    absorbed, resolved, C = split_shadows(left_kernel(D), straddles(measured, family))

    assert C.tolist() == [[2, -2, 0, 0, -2, 2]]
    assert absorbed == [2, 3]          # X0X3, X1X2
    assert resolved == [0, 1, 4, 5]


@pytest.mark.parametrize("case, family", [
    (chain_11row, pair_family(7)[0]),
    (chain_21row, pair_family(7)[0]),
    (steane_41row, two_qubit_family(7)[0]),
], ids=["chain_11row", "chain_21row", "steane_41row"])
def test_split_shadows_agrees_with_float_shadows(case, family):
    """The exact test c^P = 0 agrees with the definition beta^P = (I - Pi_D) s^P
    computed in floating point: absorbed shadows are zero, resolved ones are
    clearly not.
    """
    result = case()
    D, measured = result["D"], result["measured"]
    absorbed, resolved, _ = split_shadows(left_kernel(D), straddles(measured, family))

    sizes = np.linalg.norm(legacy_shadows(D, measured, family), axis=0)
    assert sorted(absorbed + resolved) == list(range(len(family)))  # every error, once
    assert (sizes[absorbed] < 1e-9).all()
    assert (sizes[resolved] > 1e-6).all()


# --------------------------------------------------------------------------- #
# split_absorbed
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("case, family", [
    (chain_11row, pair_family(7)[0]),
    (steane_41row, two_qubit_family(7)[0]),
], ids=["chain_11row", "steane_41row"])
def test_absorbed_fits_reproduce_straddles(case, family):
    """Each absorbed straddle is exactly D g, and on the thesis cases every fit
    is nonnegative: every absorbed error alone gives a physical report.
    """
    result = case()
    D, measured = result["D"], result["measured"]
    S = straddles(measured, family)
    absorbed, _, _ = split_shadows(left_kernel(D), S)

    physical, unphysical, fits = split_absorbed(D, S, absorbed)

    np.testing.assert_array_equal(np.asarray(D, dtype=object) @ fits, S[:, absorbed])
    assert physical == absorbed
    assert unphysical == []


def test_negative_fit_is_unphysical():
    """A straddle equal to column 0 minus column 1 of D needs a negative rate.

    Not a straddle a real error could have (it has a -1 entry): it exercises
    the unphysical branch, which no thesis case reaches.
    """
    D = np.array([[1, 0],
                  [1, 1],
                  [0, 1]])
    S = np.array([[1,  1],     # column 0: D[:, 0], fit g = (1, 0)
                  [1,  0],     # column 1: D[:, 0] - D[:, 1], fit g = (1, -1)
                  [0, -1]])

    physical, unphysical, fits = split_absorbed(D, S, [0, 1])

    assert physical == [0]
    assert unphysical == [1]
    assert fits.tolist() == [[1, 1], [0, -1]]


def test_rank_deficient_D_raises():
    """Two equal columns: the fit is not unique, so it cannot be read for signs."""
    D = np.array([[1, 1],
                  [1, 1],
                  [0, 0]])
    S = np.array([[1], [1], [0]])
    with pytest.raises(ValueError, match="rank deficient"):
        split_absorbed(D, S, [0])


# --------------------------------------------------------------------------- #
# gordan_lp
# --------------------------------------------------------------------------- #


def ints(rows):
    """An object array of Python ints, the form split_shadows and left_kernel return."""
    return np.array(rows, dtype=object)


def test_nothing_resolved_skips_the_solver(monkeypatch):
    """With no resolved errors there is nothing to certify, so the LP never runs."""
    def no_solver(*args, **kwargs):
        raise AssertionError("linprog should not be called")
    monkeypatch.setattr(qecdm.gordanLP, "linprog", no_solver)

    assert gordan_lp(ints([[0, 0]]), [], ints([[1]])) == (False, None)


def test_four_qubit_chain_is_infeasible():
    """kappa = 1 and coordinates 2, -2, -2, 2: no y has 2y >= 1 and -2y >= 1.

    Branch (ii): X0X1 and X0X2 at equal rates hide together.
    """
    C = ints([[2, -2, 0, 0, -2, 2]])
    K = ints([[-1], [0], [-1], [1], [1]])
    assert gordan_lp(C, [0, 1, 4, 5], K) == (False, None)


def test_feasible_case_gives_integer_witness():
    """Constraints y0 >= 1, y1 >= 1, y0 - y1 >= 1 are feasible, e.g. y = (2, 1).

    With K the identity the witness is y itself, so its products with the
    coordinates can be checked directly.
    """
    C = ints([[1, 0, 1],
              [0, 1, -1]])
    K = ints([[1, 0],
              [0, 1]])

    certified, witness = gordan_lp(C, [0, 1, 2], K)

    assert certified
    assert all(type(entry) is int for entry in witness)
    assert all(product > 0 for product in C.T @ witness)


def test_solver_failure_raises(monkeypatch):
    """A status other than solved (0) or infeasible (2) is an error, not a verdict."""
    def failing_solver(*args, **kwargs):
        return SimpleNamespace(status=1, message="Iteration limit reached.")
    monkeypatch.setattr(qecdm.gordanLP, "linprog", failing_solver)

    with pytest.raises(RuntimeError, match="Iteration limit"):
        gordan_lp(ints([[1]]), [0], ints([[1]]))


# --------------------------------------------------------------------------- #
# rref
# --------------------------------------------------------------------------- #


def test_rref_pivots_skip_a_free_column():
    """Column 1 has no pivot, so row 1's pivot sits in column 2: pivots[i] is
    the column of row i's pivot, not i itself.
    """
    A, pivots = rref(np.array([[1, 1, 0],
                               [0, 0, 1]]))
    assert pivots == [0, 2]
    assert A == [[1, 1, 0], [0, 0, 1]]  # already reduced


def test_rref_columns_keeps_pivots_out_of_the_augmented_part():
    """[D | s] with D = (1, 0) and s = (0, 1). Searched everywhere, column 1 (s)
    takes a pivot; limited to D's one column, it is only carried along.
    """
    M = np.array([[1, 0],
                  [0, 1]])
    assert rref(M)[1] == [0, 1]
    A, pivots = rref(M, columns=1)
    assert pivots == [0]
    assert A == [[1, 0], [0, 1]]  # the s column is untouched by any pivot


# --------------------------------------------------------------------------- #
# gordan
# --------------------------------------------------------------------------- #


def test_default_family_is_every_two_qubit_error():
    """Leaving out the family is the same as passing all two-qubit errors."""
    result = steane_41row()
    D, measured = result["D"], result["measured"]

    default = gordan(D, measured)
    explicit = gordan(D, measured, two_qubit_family(7)[0])

    assert default[:3] == explicit[:3]  # certified, absorbed, resolved
