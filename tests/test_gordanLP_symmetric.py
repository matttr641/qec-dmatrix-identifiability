"""Tests for the symmetry-reduced Gordan programme.

The orbit tools are checked on small hand-made cases first: a ring of four
Z0Z1-type checks, whose rotations are a symmetry, and a chain, whose aren't.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from qldpc import codes
from sympy.abc import x, y

from qecdm.stabilizer.dmatrix import design_matrix, pauli_errors
from qecdm.stabilizer.gordanLP_large import gordan_large, sparse_straddles, verify_witness
from qecdm.stabilizer.gordanLP_symmetric import (
    bb_translations,
    exact_reduced_witness,
    gordan_symmetric,
    induced_permutation,
    orbit_matrix,
    orbits,
)
from qecdm.stabilizer.stabilizers import label_to_stab, qldpc_stabilizers, stim_stabilizers


def stabs(labels, n):
    return np.array([label_to_stab(label, n) for label in labels])


RING = stabs(["Z0Z1", "Z1Z2", "Z2Z3", "Z0Z3"], 4)  # checks round a ring of 4 qubits
ROTATIONS = [(np.arange(4) + s) % 4 for s in range(4)]  # qubit j -> j + s mod 4


# --------------------------------------------------------------------------- #
# induced_permutation
# --------------------------------------------------------------------------- #


def test_rotation_moves_each_ring_check_along_one():
    """Qubit j -> j + 1 sends Z0Z1 to Z1Z2, ..., and Z0Z3 to Z1Z0 = Z0Z1."""
    assert ROTATIONS[1].tolist() == [1, 2, 3, 0]
    assert induced_permutation(RING, ROTATIONS[1]).tolist() == [1, 2, 3, 0]


def test_swap_moves_single_qubit_errors():
    """Swapping qubits 0 and 1 swaps X0 with X1, Y0 with Y1, Z0 with Z1."""
    errors, labels = pauli_errors(3, 1)  # X0 Y0 Z0 X1 Y1 Z1 X2 Y2 Z2
    perm = induced_permutation(errors, [1, 0, 2])
    assert [labels[i] for i in perm] == ["X1", "Y1", "Z1", "X0", "Y0", "Z0", "X2", "Y2", "Z2"]


def test_identity_fixes_everything():
    errors, _ = pauli_errors(4, 2)
    assert induced_permutation(errors, np.arange(4)).tolist() == list(range(len(errors)))


def test_not_a_symmetry_raises():
    """A chain is not a ring: rotating Z2Z3 gives Z3Z0, which isn't a check."""
    chain = stabs(["Z0Z1", "Z1Z2", "Z2Z3"], 4)
    with pytest.raises(ValueError, match="does not map the list to itself"):
        induced_permutation(chain, ROTATIONS[1])


def test_not_a_permutation_raises():
    with pytest.raises(ValueError, match="not a permutation"):
        induced_permutation(RING, [0, 0, 1, 2])


def test_repeated_pauli_raises():
    with pytest.raises(ValueError, match="repeats a Pauli"):
        induced_permutation(stabs(["Z0Z1", "Z0Z1"], 2), [1, 0])


# --------------------------------------------------------------------------- #
# orbits
# --------------------------------------------------------------------------- #


def test_rotations_put_the_ring_checks_in_one_orbit():
    perms = [induced_permutation(RING, q) for q in ROTATIONS]
    assert orbits(perms).tolist() == [0, 0, 0, 0]


def test_rotations_split_single_qubit_errors_by_letter():
    """Rotating moves a Pauli between qubits but never changes its letter."""
    errors, _ = pauli_errors(4, 1)  # X0 Y0 Z0 X1 ...
    perms = [induced_permutation(errors, q) for q in ROTATIONS]
    assert orbits(perms).tolist() == [0, 1, 2] * 4  # every X has X0's label, and so on


def test_orbit_sizes_two_qubit_errors_on_the_ring():
    """Orbit-stabilizer: an orbit has 4 / (rotations fixing it) members.

    The half turn (j -> j + 2) fixes {0, 2} and {1, 3} when both qubits carry
    the same letter, so those 3 + 3 = 6 errors pair up into 3 orbits of size 2.
    Every other two-qubit error has an orbit of size 4.
    """
    errors, _ = pauli_errors(4, 2)  # 6 qubit pairs x 9 letter pairs = 54
    labels = orbits([induced_permutation(errors, q) for q in ROTATIONS])
    sizes = np.bincount(np.unique(labels, return_inverse=True)[1])
    assert sorted(sizes.tolist()) == [2] * 3 + [4] * 12
    assert sizes.sum() == 54


def test_generators_only_raise():
    """The quarter turn alone isn't a group: its labels change under it."""
    with pytest.raises(ValueError, match="not a whole group"):
        orbits([induced_permutation(RING, ROTATIONS[1])])


def test_missing_group_element_raises():
    """The identity and the quarter turn, without the other two turns."""
    with pytest.raises(ValueError, match="not a whole group"):
        orbits([induced_permutation(RING, q) for q in ROTATIONS[:2]])


# --------------------------------------------------------------------------- #
# orbit_matrix
# --------------------------------------------------------------------------- #


def test_orbit_matrix_by_hand():
    B = orbit_matrix(np.array([0, 1, 0, 3, 1, 3]))
    np.testing.assert_array_equal(B.toarray(), [[1, 0, 0],
                                                [0, 1, 0],
                                                [1, 0, 0],
                                                [0, 0, 1],
                                                [0, 1, 0],
                                                [0, 0, 1]])


def test_orbit_matrix_spreads_one_value_per_orbit():
    """lambda = B mu: each element gets its orbit's value."""
    B = orbit_matrix(np.array([0, 1, 0, 3, 1, 3]))
    assert (B @ np.array([10, 20, 30])).tolist() == [10, 20, 10, 30, 20, 30]


def test_orbit_matrix_rows_have_one_entry_and_columns_count_the_orbit():
    errors, _ = pauli_errors(4, 2)
    labels = orbits([induced_permutation(errors, q) for q in ROTATIONS])
    B = orbit_matrix(labels)
    assert B.shape == (54, 15)
    assert (B.sum(axis=1) == 1).all()  # every error in exactly one orbit
    assert sorted(np.asarray(B.sum(axis=0)).ravel().tolist()) == [2] * 3 + [4] * 12


# --------------------------------------------------------------------------- #
# bb_translations
# --------------------------------------------------------------------------- #


def bb72():
    return codes.BBCode({x: 6, y: 6}, x**3 + y + y**2, y**3 + x + x**2)


BB_CODES = [
    pytest.param(bb72, id="bb72"),
    pytest.param(lambda: codes.BBCode({x: 3, y: 5}, x + y, 1 + x * y), id="3x5"),
    pytest.param(lambda: codes.BBCode({y: 5, x: 3}, x + y, 1 + x * y), id="5x3"),  # orders listed y first
]


def test_bb72_has_36_distinct_translations():
    perms = bb_translations(bb72())
    assert len(perms) == 36
    assert len({q.tobytes() for q in perms}) == 36
    assert all(np.array_equal(np.sort(q), np.arange(72)) for q in perms)
    np.testing.assert_array_equal(perms[0], np.arange(72))  # x^0 y^0 is the identity


def test_translation_moves_one_step_along_x():
    """x moves qubit (block, a, b) to (block, a + 1, b): index + m, wrapping in a."""
    q = bb_translations(bb72())[6]  # s = 1, t = 0
    assert q[0] == 6 and q[30] == 0  # (L, 0, 0) -> (L, 1, 0); (L, 5, 0) -> (L, 0, 0)
    assert q[36] == 42               # (R, 0, 0) -> (R, 1, 0)


@pytest.mark.parametrize("make", BB_CODES)
def test_translations_map_checks_to_checks(make):
    code = make()
    checks, _, _ = qldpc_stabilizers(code)
    for q in bb_translations(code):
        induced_permutation(checks, q)  # raises if some check is sent off the list


@pytest.mark.parametrize("make", BB_CODES)
def test_translations_are_a_group_with_one_orbit_per_block(make):
    """Translations act transitively on each block: L is one orbit, R the other."""
    code = make()
    l, m = code.orders
    labels = orbits(bb_translations(code))
    assert labels.tolist() == [0] * (l * m) + [l * m] * (l * m)


def test_shifting_one_block_is_not_a_symmetry():
    """The check at h sits on hA in L and hB in R: moving L alone breaks it."""
    code = bb72()
    checks, _, _ = qldpc_stabilizers(code)
    q = np.concatenate([bb_translations(code)[6][:36], np.arange(36, 72)])
    with pytest.raises(ValueError, match="does not map the list to itself"):
        induced_permutation(checks, q)


def test_not_a_bb_layout_raises():
    with pytest.raises(ValueError, match="not a BB layout"):
        bb_translations(SimpleNamespace(orders=(2, 3), num_qudits=7))
        #simplename space makes an object with those characteristics
        #breaks since need n = 2 * l * m


@pytest.mark.slow
def test_bb72_orbit_counts():
    """355 row orbits, all of size 36; 648 error orbits: 630 of size 36, 18 of size 18."""
    code = bb72()
    checks, _, _ = qldpc_stabilizers(code)
    perms = bb_translations(code)
    _, _, measured, _ = design_matrix(checks)
    errors, _ = pauli_errors(72, 2)

    def sizes(paulis):
        labels = orbits([induced_permutation(paulis, q) for q in perms])
        return sorted(np.bincount(np.unique(labels, return_inverse=True)[1]).tolist())

    assert sizes(measured) == [36] * 355
    assert sizes(errors) == [18] * 18 + [36] * 630


# --------------------------------------------------------------------------- #
# exact_reduced_witness
# --------------------------------------------------------------------------- #


def test_exact_reduced_witness_doubles_until_it_rounds():
    """One equality mu0 = mu1 and one inequality 3 mu0 > 0, from the float mu = (0.4, 0.4).

    t = 1 rounds mu1 to 0, so mu = (0, 0) fails; t = 2 rounds 0.8 to 1, so mu = (1, 1).
    """
    D_red = np.array([[1], [-1]])  # D_red^T mu = mu0 - mu1
    S_red = np.array([[3], [0]])   # B^T s^P = (3, 0)
    assert exact_reduced_witness(D_red, S_red, np.array([0.4, 0.4])).tolist() == [1, 1]


def test_exact_reduced_witness_weights_by_the_counts():
    """B^T s^P counts rows, so <mu, B^T s^P> = 1*2 + 3*(-1) = -1 fails.

    Summing mu over the nonzero entries, as for 0/1 straddles, would give 2 - 1 = 1
    and wrongly pass. No rescaling helps, so it gives up.
    """
    D_red = np.zeros((2, 1), dtype=int)  # no equalities
    S_red = np.array([[1], [3]])
    with pytest.raises(RuntimeError, match="could not round"):
        exact_reduced_witness(D_red, S_red, np.array([2.0, -1.0]), max_doublings=5)


# --------------------------------------------------------------------------- #
# gordan_symmetric
# --------------------------------------------------------------------------- #


def bb_inputs(orders, poly_a, poly_b):
    code = codes.BBCode(orders, poly_a, poly_b)
    checks, _, _ = qldpc_stabilizers(code)
    D, _, measured, _ = design_matrix(checks)
    return D, measured, bb_translations(code)


def no_symmetry(D, measured):
    return D, measured, [np.arange(measured.shape[2])]  # the group with just the identity


def stim_surface(distance):
    generators, _, _ = stim_stabilizers("surface_code:rotated_memory_z", distance=distance, rounds=1)
    D, _, measured, _ = design_matrix(generators)
    return no_symmetry(D, measured)


def qldpc_steane():
    checks, _, _ = qldpc_stabilizers(codes.SteaneCode())
    D, _, measured, _ = design_matrix(checks)
    return no_symmetry(D, measured)


CERTIFIED_BB = (({x: 2, y: 3}, 1 + y + x * y**2, 1 + y + x * y**2),)  # D' is 54 x 18

AGREEMENT_CASES = [
    pytest.param(qldpc_steane, id="steane_identity"),
    pytest.param(lambda: stim_surface(3), id="surface_d3_identity"),
    pytest.param(lambda: bb_inputs(*CERTIFIED_BB[0]), id="bb_2x3_certified"),
    pytest.param(lambda: bb_inputs({x: 2, y: 3}, 1 + x, 1 + y), id="bb_2x3"),
    pytest.param(lambda: bb_inputs({x: 3, y: 3}, x + y, 1 + x * y), id="bb_3x3"),
]


@pytest.mark.parametrize("inputs", AGREEMENT_CASES)
def test_symmetric_route_matches_large_route(inputs):
    """Same verdict, and the same absorbed, resolved, physical and unphysical errors."""
    D, measured, perms = inputs()
    large = gordan_large(D, measured)
    symmetric = gordan_symmetric(D, measured, perms)

    assert symmetric[0] == large[0]
    assert symmetric[1:3] == large[1:3]
    assert symmetric[4:] == large[4:]
    assert (symmetric[3] is None) == (not symmetric[0])  # a witness exactly when certified


def test_both_branches_are_covered():
    """The agreement cases include a certified and an uncertified code."""
    verdicts = {gordan_symmetric(*case.values[0]())[0] for case in AGREEMENT_CASES}
    assert verdicts == {True, False}


@pytest.fixture(scope="module")
def certified_bb():
    D, measured, perms = bb_inputs(*CERTIFIED_BB[0])
    return D, measured, perms, gordan_symmetric(D, measured, perms)


def test_witness_is_an_exact_witness_on_the_full_problem(certified_bb):
    D, measured, _, (certified, _, resolved, witness, _, _) = certified_bb
    S = sparse_straddles(measured, pauli_errors(measured.shape[2], 2)[0])
    assert certified
    assert all(isinstance(v, int) for v in witness)
    assert verify_witness(D, S, resolved, witness)


def test_witness_is_constant_on_each_row_orbit(certified_bb):
    """lambda = B mu, so moving the rows by any translation leaves lambda unchanged."""
    _, measured, perms, (_, _, _, witness, _, _) = certified_bb
    for q in perms:
        assert np.array_equal(witness[induced_permutation(measured, q)], witness)


def test_not_a_symmetry_raises_before_solving():
    """Shifting block L alone is not a symmetry, so the orbits cannot be formed."""
    D, measured, perms = bb_inputs(*CERTIFIED_BB[0])
    n = measured.shape[2]
    half = np.concatenate([perms[1][: n // 2], np.arange(n // 2, n)])
    group = [np.arange(n), half]  # not a symmetry, whatever else is passed
    with pytest.raises(ValueError, match="does not map the list to itself"):
        gordan_symmetric(D, measured, group)


@pytest.mark.slow
def test_bb72_is_certified():
    """[[72,12,6]]: every two-qubit error resolved, and a witness exact on all 23004."""
    D, measured, perms = bb_inputs({x: 6, y: 6}, x**3 + y + y**2, y**3 + x + x**2)
    certified, absorbed, resolved, witness, physical, unphysical = gordan_symmetric(D, measured, perms)
    assert certified
    assert (len(absorbed), len(resolved)) == (0, 23004)
    assert physical == unphysical == []
    assert len(witness) == 12780
    assert all(isinstance(v, int) for v in witness)


def test_family_may_be_a_list():
    """family[indices] needs an array: a list of Paulis is converted first."""
    D, measured, perms = bb_inputs(*CERTIFIED_BB[0])
    family, _ = pauli_errors(measured.shape[2], 2)
    as_array = gordan_symmetric(D, measured, perms, family)
    as_list = gordan_symmetric(D, measured, perms, list(family))
    assert as_list[:3] == as_array[:3] and as_list[4:] == as_array[4:]
    assert as_list[3].tolist() == as_array[3].tolist()


def test_full_straddles_are_never_built(monkeypatch):
    """S is built for the 102 representatives, then for the 540 resolved errors in
    the final check, never for all 594 errors at once."""
    import qecdm.stabilizer.gordanLP_large
    import qecdm.stabilizer.gordanLP_symmetric

    D, measured, perms = bb_inputs(*CERTIFIED_BB[0])
    widths = []

    def recording(measured, family):
        widths.append(len(family))
        return sparse_straddles(measured, family)

    monkeypatch.setattr(qecdm.stabilizer.gordanLP_symmetric, "sparse_straddles", recording)
    monkeypatch.setattr(qecdm.stabilizer.gordanLP_large, "sparse_straddles", recording)
    assert gordan_symmetric(D, measured, perms)[0]  # certified, so the final check runs
    assert widths == [102, 540]
