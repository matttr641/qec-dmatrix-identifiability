"""Tests for the BB pipeline: bb_code_analyse's methods agree, and bb_code_split's
anchored count gives the same absorbed/resolved split as classifying every error."""

import pytest
from qldpc import codes
from sympy.abc import x, y

from qecdm.thesis_on_bb import anchored_errors, bb_code_analyse, bb_code_split

COUNTS = ["qubits", "logicals", "checks", "rows", "classes", "kappa", "max_order",
          "family", "absorbed", "resolved", "physical", "unphysical"]

SMALL = [
    pytest.param(({x: 2, y: 3}, 1 + y + x * y**2, 1 + y + x * y**2), id="2x3_certified"),
    pytest.param(({x: 2, y: 3}, 1 + x, 1 + y), id="2x3"),
    pytest.param(({x: 3, y: 3}, x + y, 1 + x * y), id="3x3"),
    pytest.param(({x: 3, y: 5}, x + y, 1 + x * y), id="3x5"),
]


@pytest.mark.parametrize("args", SMALL[:1])
def test_methods_agree(args):
    results = [bb_code_analyse(*args, method=m) for m in ("symmetric", "large", "exact")]
    for r in results[1:]:
        assert [r[c] for c in COUNTS + ["certified"]] == [results[0][c] for c in COUNTS + ["certified"]]


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="method must be"):
        bb_code_analyse({x: 2, y: 3}, 1 + x, 1 + y, method="fast")


def test_anchored_errors_count_and_origin():
    """2 origins x 11 other qubits x 9 letter pairs; each has a qubit 0 or 6 (L or R origin)."""
    errors = anchored_errors(codes.BBCode({x: 2, y: 3}, 1 + x, 1 + y))
    assert errors.shape == (2 * 11 * 9, 2, 12)
    support = errors[:, 0] | errors[:, 1]
    assert ((support[:, 0] == 1) | (support[:, 6] == 1)).all()
    assert (support.sum(axis=1) == 2).all()


@pytest.mark.parametrize("args", SMALL)
def test_split_matches_full_pipeline(args):
    """The anchored count equals classifying all errors, on certified and uncertified codes."""
    full, split = bb_code_analyse(*args), bb_code_split(*args)
    assert [split[c] for c in COUNTS] == [full[c] for c in COUNTS]


@pytest.mark.slow
def test_split_bb144():
    """[[144,12,12]]: 648 absorbed (all physical), as bb_code_analyse finds from all 92,664."""
    r = bb_code_split({x: 12, y: 6}, x**3 + y + y**2, y**3 + x + x**2)
    assert (r["family"], r["absorbed"], r["resolved"]) == (92664, 648, 92016)
    assert (r["physical"], r["unphysical"]) == (648, 0)
