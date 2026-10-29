"""Tests for the large-code Gordan programme.

The large route must give exactly the results of the exact route in
gordanLP.py, which is itself checked against the thesis. Its pieces are then
checked one by one.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import qecdm.gordanLP_large
from cases import chain_11row, chain_21row, steane_41row
from qecdm.dmatrix import design_matrix, pauli_errors
from qecdm.gordanLP import gordan, rref, split_absorbed, straddles
from qecdm.gordanLP_large import (
    PRIME, _exact_fit, gordan_large, inverse_mod, pivot_rows, sparse_lp,
    sparse_straddles, split_rows, verify_witness, verify_witness_blocks,
)
from qecdm.stabilizer import stim_stabilizers
from qecnoise.Dmatrixsinglequbit import pair_family, two_qubit_family


FIXTURES = Path(__file__).parent / "fixtures"
THESIS = ["chain_11row", "chain_21row", "steane_41row"]


def surface(distance):
    """D', the measured set and every two-qubit error, for a rotated surface code."""
    generators, qubits, _ = stim_stabilizers("surface_code:rotated_memory_z",
                                             distance=distance, rounds=1)
    D, _, measured, _ = design_matrix(generators)
    return D, measured, pauli_errors(len(qubits), 2)[0]


def thesis(case, family):
    result = case()
    return result["D"], result["measured"], family


# --------------------------------------------------------------------------- #
# The two routes agree
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("inputs", [
    pytest.param(lambda: thesis(chain_11row, pair_family(7)[0]), id="chain_11row"),
    pytest.param(lambda: thesis(chain_21row, pair_family(7)[0]), id="chain_21row"),
    pytest.param(lambda: thesis(steane_41row, two_qubit_family(7)[0]), id="steane_41row"),
    pytest.param(lambda: surface(3), id="surface_d3"),
    pytest.param(lambda: surface(5), id="surface_d5", marks=pytest.mark.slow),
])
def test_large_route_matches_exact_route(inputs):
    """Same absorbed, resolved, physical and unphysical errors, and the same verdict."""
    D, measured, family = inputs()

    certified, absorbed, resolved, _ = gordan(D, measured, family)
    physical, unphysical, _ = split_absorbed(D, straddles(measured, family), absorbed)
    large = gordan_large(D, measured, family)

    assert large[0] == certified
    assert large[1:3] == (absorbed, resolved)
    assert large[4:] == (physical, unphysical)


# --------------------------------------------------------------------------- #
# The witness is exact
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def steane_witness():
    D, measured, family = thesis(steane_41row, two_qubit_family(7)[0])
    certified, _, resolved, witness, _, _ = gordan_large(D, measured, family)
    assert certified  # the tests below need a witness
    return D, measured, family, resolved, witness


def test_witness_is_integer(steane_witness):
    _, _, _, _, witness = steane_witness
    assert all(type(entry) is int for entry in witness)


def test_witness_is_in_kernel_exactly(steane_witness):
    D, _, _, _, witness = steane_witness
    assert all(entry == 0 for entry in np.asarray(D, dtype=object).T @ witness)


def test_witness_is_positive_on_every_resolved_straddle(steane_witness):
    """Checked against the dense straddles, independently of the sparse code."""
    _, measured, family, resolved, witness = steane_witness
    S = np.asarray(straddles(measured, family), dtype=object)
    assert all(product > 0 for product in witness @ S[:, resolved])


def test_verify_witness_accepts_the_witness(steane_witness):
    D, measured, family, resolved, witness = steane_witness
    assert verify_witness(D, sparse_straddles(measured, family), resolved, witness)


def test_verify_witness_rejects_outside_the_kernel(steane_witness):
    """One entry changed: D^T lambda is no longer 0."""
    D, measured, family, resolved, witness = steane_witness
    broken = witness.copy()
    broken[0] += 1
    assert not verify_witness(D, sparse_straddles(measured, family), resolved, broken)


def test_verify_witness_rejects_the_negated_witness(steane_witness):
    """-lambda is still in ker(D^T), but every <lambda, s^P> is now negative."""
    D, measured, family, resolved, witness = steane_witness
    assert not verify_witness(D, sparse_straddles(measured, family), resolved, -witness)


# Steane has 126 resolved errors: blocks of 1, of 50 (the last one short), of 63
# (an exact divisor), and of 1000 (all in one block).
BLOCKS = [1, 50, 63, 1000]


@pytest.mark.parametrize("block", BLOCKS)
def test_verify_witness_blocks_accepts_the_witness(steane_witness, block):
    D, measured, family, resolved, witness = steane_witness
    assert len(resolved) == 126
    assert verify_witness_blocks(D, measured, family, resolved, witness, block=block)


@pytest.mark.parametrize("block", BLOCKS)
def test_verify_witness_blocks_rejects_outside_the_kernel(steane_witness, block):
    D, measured, family, resolved, witness = steane_witness
    broken = witness.copy()
    broken[0] += 1
    assert not verify_witness_blocks(D, measured, family, resolved, broken, block=block)


@pytest.mark.parametrize("block", BLOCKS)
def test_verify_witness_blocks_rejects_the_negated_witness(steane_witness, block):
    D, measured, family, resolved, witness = steane_witness
    assert not verify_witness_blocks(D, measured, family, resolved, -witness, block=block)


def test_verify_witness_blocks_checks_every_block(steane_witness):
    """An absorbed error put last fails, and only it: s = D g gives <lambda, s> =
    <D^T lambda, g> = 0. With blocks of 50 it sits alone in the short last block
    (50 + 50 + 27), so a loop that stopped early or skipped that block would pass."""
    D, measured, family, resolved, witness = steane_witness
    absorbed = sorted(set(range(len(family))) - set(resolved))[0]
    S = sparse_straddles(measured, family)
    assert not verify_witness(D, S, [absorbed], witness)
    assert not verify_witness_blocks(D, measured, family, resolved + [absorbed], witness, block=50)


def test_verify_witness_blocks_nothing_resolved(steane_witness):
    """With no errors to check, both versions report only the kernel check."""
    D, measured, family, _, witness = steane_witness
    S = sparse_straddles(measured, family)
    assert verify_witness_blocks(D, measured, family, [], witness) == verify_witness(D, S, [], witness)


# --------------------------------------------------------------------------- #
# The pieces
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", THESIS)
def test_pivot_rows_match_exact_pivots(name):
    """The same rows as exact elimination of D^T, and D' on them is invertible."""
    D = np.load(FIXTURES / f"{name}.npz")["D"]
    pivots = pivot_rows(D)
    assert pivots == rref(D.T)[1]
    assert np.linalg.matrix_rank(D[pivots]) == D.shape[1]


def test_pivot_rows_rank_deficient_raises():
    with pytest.raises(ValueError, match="fewer pivots"):
        pivot_rows(np.array([[1, 1], [1, 1], [0, 0]]))


@pytest.mark.parametrize("name", THESIS)
def test_inverse_mod(name):
    """M D_P = I modulo p."""
    D = np.load(FIXTURES / f"{name}.npz")["D"]
    D_P, _ = split_rows(D, pivot_rows(D))
    M = inverse_mod(D_P)
    assert np.array_equal(M @ D_P % PRIME, np.eye(len(D_P), dtype=np.int64))


def test_inverse_mod_halves():
    """det = -2 on the four-qubit chain, so the inverse holds 1/2 = (p + 1) / 2 mod p."""
    D_P = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [1, 0, 1, 0]])
    M = inverse_mod(D_P)
    assert M[0, 0] == (PRIME + 1) // 2  # the exact inverse has 1/2 there
    assert M[0, 1] == (PRIME - 1) // 2  # and -1/2 there


@pytest.mark.parametrize("case, family", [
    (steane_41row, two_qubit_family(7)[0]),
    (chain_21row, pair_family(7)[0]),
], ids=["steane_41row", "chain_21row"])
def test_sparse_straddles_equal_dense(case, family):
    measured = case()["measured"]
    S = sparse_straddles(measured, family)
    np.testing.assert_array_equal(S.toarray(), straddles(measured, family))


def test_split_rows_keeps_order():
    M = np.arange(10).reshape(5, 2)
    M_P, M_F = split_rows(M, [1, 3])
    assert M_P.tolist() == [[2, 3], [6, 7]]
    assert M_F.tolist() == [[0, 1], [4, 5], [8, 9]]  # rows 0, 2, 4, in order


def test_exact_fit_in_and_out_of_the_image():
    """X0X3 on the four-qubit chain is X0 + X3; X0X1 has no single-qubit fit."""
    D = np.array([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [1, 0, 1, 0], [0, 1, 0, 1]])
    D_P = D[:4]
    absorbed_s = np.array([1, 0, 1, 1, 1])   # X0X3
    resolved_s = np.array([0, 1, 0, 1, 1])   # X0X1
    assert _exact_fit(D_P, absorbed_s[:4], D, absorbed_s) == [1, 0, 0, 1]
    assert _exact_fit(D_P, resolved_s[:4], D, resolved_s) is None


def test_sparse_lp_nothing_resolved():
    D = np.array([[1], [0]])
    S = np.array([[1], [0]])
    assert sparse_lp(D, S, []) == (False, None)


def test_sparse_lp_raises_when_both_methods_fail(monkeypatch):
    """Interior point, then dual simplex: neither reaching a verdict is an error."""
    def no_verdict(*args, **kwargs):
        return SimpleNamespace(status=4, message="Numerical difficulties.")
    monkeypatch.setattr(qecdm.gordanLP_large, "linprog", no_verdict)

    D = np.array([[1], [0]])
    S = np.array([[1], [1]])
    with pytest.raises(RuntimeError, match="Numerical difficulties"):
        sparse_lp(D, S, [0])
