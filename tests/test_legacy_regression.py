"""Regression tests: the legacy code must reproduce the thesis results.

Expected values come from the thesis, not from running the code.
"""

from pathlib import Path

import numpy as np
import pytest

from cases import chain_11row, chain_21row, steane_41row, steane_generators
from qecnoise.Dmatrixsinglequbit import commutators, design_matrix, pair_family, two_qubit_family
from qecnoise.gordanLP import certify_witness, gordan
from qecnoise.girth import physical_girth


# --------------------------------------------------------------------------- #
# Thesis values
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("case, rows, classes", [
    (chain_11row, 11, 7),
    (chain_21row, 21, 7),
    (steane_41row, 41, 21),
])
def test_shape_of_D(case, rows, classes):
    """Rows = size of the measured set; columns = |C*|, the syndrome classes."""
    D = case()["D"]
    assert D.shape == (rows, classes)


def test_steane_kappa_is_20():
    """kappa = rows - rank(D') = 41 - 21."""
    D = steane_41row()["D"]
    assert D.shape[0] - np.linalg.matrix_rank(D) == 20


def test_chain_11row_pairs():
    """Eleven-row set: 10 pairs hide alone, and the girth is 2."""
    case = chain_11row()
    pairs, _ = pair_family(7)
    absorbed, spark, m_star, members = physical_girth(case["D"], case["measured"], pairs)
    assert len(absorbed) == 10
    assert m_star == 2


def test_chain_21row_pairs():
    """Complete weight-2 set: no pair hides alone, and the girth is 6."""
    case = chain_21row()
    pairs, _ = pair_family(7)
    absorbed, spark, m_star, members = physical_girth(case["D"], case["measured"], pairs)
    assert len(absorbed) == 0
    assert m_star == 6


def test_steane_two_qubit_errors():
    """Steane: the 63 matched errors (XX, YY, ZZ) hide alone; nothing else can hide."""
    case = steane_41row()
    family, labels = two_qubit_family(7)
    absorbed, spark, m_star, members = physical_girth(case["D"], case["measured"], family)
    assert len(absorbed) == 63
    assert all(labels[i][0] == labels[i][2] for i in absorbed)  # e.g. "X0X3"
    assert m_star == np.inf


# --------------------------------------------------------------------------- #
# Fixtures: the full outputs must match the saved answer key
# --------------------------------------------------------------------------- #

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("case, family", [
    (chain_11row, pair_family(7)[0]),
    (chain_21row, pair_family(7)[0]),
    (steane_41row, two_qubit_family(7)[0]),
], ids=["chain_11row", "chain_21row", "steane_41row"])
def test_matches_fixture(case, family):
    """D', the measured set and the absorbed errors are exactly as saved by generate_fixtures.py."""
    saved = np.load(FIXTURES / f"{case.__name__}.npz")
    result = case()
    _, absorbed, _, _ = gordan(result["D"], result["measured"], family)

    np.testing.assert_array_equal(result["D"], saved["D"])
    np.testing.assert_array_equal(result["measured"], saved["measured"])
    np.testing.assert_array_equal(absorbed, saved["absorbed"])

# --------------------------------------------------------------------------- #
# Properties that must hold whatever the thesis says
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("case", [chain_11row, chain_21row, steane_41row])
def test_D_is_binary_and_matches_measured_set(case):
    result = case()
    assert set(np.unique(result["D"])) <= {0, 1}
    assert len(result["D"]) == len(result["measured"])


@pytest.mark.parametrize("case", [chain_11row, chain_21row, steane_41row])
def test_measured_stabilizers_commute(case):
    measured = case()["measured"]
    assert not commutators(measured, measured).any()


def test_wrong_shape_is_rejected():
    with pytest.raises(ValueError):
        design_matrix(7, 5, steane_generators())  # says 5 generators, gives 6


def test_steane_lp_verdict_is_exact():
    """The floating-point LP says Steane is safe; the exact integer check must agree.

    certify_witness checks every error it is given, so pass the resolved ones only:
    the absorbed ones have zero shadow and would make the check fail.
    """
    case = steane_41row()
    family, _ = two_qubit_family(7)
    certified, absorbed, resolved, witness = gordan(case["D"], case["measured"], family)
    assert certified
    resolved_errors = np.asarray(family)[resolved]
    proved, worst, checked = certify_witness(case["D"], case["measured"], resolved_errors, witness)
    assert proved
