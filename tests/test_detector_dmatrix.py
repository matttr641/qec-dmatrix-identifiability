"""Tests for the detector D': the graph, the parity rows, the tied counts, and that the
walk reaches rank(M)."""

import numpy as np
import pytest
from scipy import sparse
from sympy.abc import x, y

from qecdm.detector import detector_dmatrix
from qecdm.detector.circuits import bb_memory, repetition_memory, surface_memory, uniform_noise
from qecdm.detector.detector_dmatrix import (detector_design_matrix, detector_graph, signature_matrix,
                                             subset_rows)

BB = ({x: 2, y: 3}, 1 + y + x * y**2, 1 + y + x * y**2)  # [[12,4]], the small certified BB code
NOISE = uniform_noise(0.01)

CIRCUITS = {
    "rep_r2": lambda: repetition_memory(3, 2, NOISE),
    "surface_r3": lambda: surface_memory(3, 3, NOISE),
    "bb_r3": lambda: bb_memory(*BB, 3, NOISE),
}
CASES = [(name, rule) for name in CIRCUITS for rule in ("rounds", "channels")]


@pytest.fixture(scope="module")
def designs():
    """(name, rule) -> DetectorDesign, each built once and shared by every test."""
    return {(name, rule): detector_design_matrix(CIRCUITS[name](), rule) for name, rule in CASES}


# --- the pieces, by hand -------------------------------------------------------------

def test_detector_graph_by_hand():
    """Faults trip {0, 1} and {1, 2}: 0-1 and 1-2 are joined, 0-2 are not."""
    H = sparse.csc_matrix(np.array([[1, 0], [1, 1], [0, 1]]))
    assert detector_graph(H).tolist() == [[True, True, False],
                                          [True, True, True],
                                          [False, True, True]]


def test_signature_matrix_by_hand():
    assert signature_matrix([(0,), (0, 2), (1,)], 3).tolist() == [[1, 1, 0],
                                                                  [0, 0, 1],
                                                                  [0, 1, 0]]


def test_subset_rows_are_parities():
    """Row {0, 2}: pattern (0,) overlaps once -> 1; (0, 2) twice -> 0; (1,) never -> 0."""
    H_sig = signature_matrix([(0,), (0, 2), (1,)], 3)
    assert subset_rows([(0,), (0, 2)], H_sig).tolist() == [[1, 1, 0],
                                                          [1, 0, 0]]


def test_tied_faults_add_after_the_parity():
    """One parameter with a fault on D0 and one on D2: row {0, 2} gives 1 + 1 = 2, not 0."""
    H_sig = signature_matrix([(0,), (2,)], 3)
    M = np.array([[1], [1]])  # one of the parameter's faults on each pattern
    assert (subset_rows([(0, 2)], H_sig) @ M).tolist() == [[2]]


def test_repetition_idle_gives_2_on_its_pair_row(designs):
    """The same in the real circuit: data 0's idle X trips D0 in round 1 and D2 in round 2."""
    d = designs["rep_r2", "rounds"]
    column = next(k for k, group in enumerate(d.groups) if ("idle", -2, (0,), "X0") in group)
    assert d.D[d.subsets.index((0, 2)), column] == 2


# --- the walk --------------------------------------------------------------------------

@pytest.mark.parametrize("name, rule", CASES)
def test_reaches_the_target(designs, name, rule):
    d = designs[name, rule]
    assert np.linalg.matrix_rank(d.D) == d.target
    assert d.D.shape == (len(d.subsets), len(d.groups))


@pytest.mark.parametrize("name, rule", CASES)
def test_directions_are_invisible(designs, name, rule):
    """D inherits M's null space: every direction gives D v = 0."""
    d = designs[name, rule]
    for direction in d.directions:
        v = np.zeros(len(d.groups), dtype=int)
        for group, coefficient in direction:
            v[d.groups.index(group)] = coefficient
        assert not (d.D @ v).any()


@pytest.mark.parametrize("name, rule", CASES)
def test_rows_are_new_and_entries_are_counts(designs, name, rule):
    """Integer entries >= 0, each row a different detector set, and D_fixed has the same rows.
    (Tied rows may repeat: different detector sets can see the same counts per parameter,
    and are still separate measurements.)"""
    d = designs[name, rule]
    assert d.D.dtype.kind == "i" and d.D.min() >= 0
    assert len(set(d.subsets)) == len(d.subsets)
    assert d.D_fixed.shape == (len(d.D), len(d.fixed_names))


def test_rows_start_with_the_single_detectors(designs):
    """Order 1 comes first: the 6 detectors of the repetition code, then pairs."""
    subsets = designs["rep_r2", "rounds"].subsets
    assert subsets[:6] == [(0,), (1,), (2,), (3,), (4,), (5,)]
    assert {len(s) for s in subsets[6:]} == {2}


def test_max_order_stops_early():
    """One order only: the single detectors, short of the target, without raising."""
    d = detector_design_matrix(repetition_memory(3, 2, NOISE), "rounds", max_order=1)
    assert {len(s) for s in d.subsets} == {1}
    assert np.linalg.matrix_rank(d.D) < d.target


def test_raises_when_the_target_is_never_reached(monkeypatch):
    """With only single detectors available, the walk runs out below the target."""
    def singles_only(graph):
        yield [(i,) for i in range(len(graph))]
        while True:
            yield []
    monkeypatch.setattr(detector_dmatrix, "connected_subsets", singles_only)
    with pytest.raises(RuntimeError, match="at every order"):
        detector_design_matrix(repetition_memory(3, 2, NOISE), "rounds")
