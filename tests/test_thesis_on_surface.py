"""Tests for the surface-code pipeline.

surface_code_analyse only chains tested functions together, so these lock in
the chain itself: the right functions, in the right order, with the right
arguments (for example the measured set, not the generators, going to gordan).
"""

import pytest

from qecdm.thesis_on_surface import surface_code_analyse


@pytest.fixture(scope="module")
def d3():
    """The d = 3 rotated surface code, analysed once for the module."""
    return surface_code_analyse(3)


def test_d3_known_values(d3):
    """The values recorded when the pipeline was first run on d = 3."""
    expected = {
        "qubits": 9, "generators": 8, "rows": 50, "classes": 23, "kappa": 27,
        "max_order": 3, "family": 324, "absorbed": 78, "resolved": 246,
        "physical": 78, "unphysical": 0, "certified": False,
    }
    assert {key: d3[key] for key in expected} == expected


def test_d3_counts_add_up(d3):
    """Every family member is absorbed or resolved, and every absorbed one is
    physical or unphysical, with nothing counted twice.
    """
    assert d3["absorbed"] + d3["resolved"] == d3["family"]
    assert d3["physical"] + d3["unphysical"] == d3["absorbed"]
    assert sorted(d3["absorbed_indices"] + d3["resolved_indices"]) == list(range(d3["family"]))
