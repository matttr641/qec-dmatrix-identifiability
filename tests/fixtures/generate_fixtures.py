"""Save the legacy outputs for each thesis case, as the answer key for the regression tests.

Run once, from the repo root:
    python tests/fixtures/generate_fixtures.py

Check the printed shapes and counts against the thesis, then commit the .npz files.
Rerun only when you deliberately change what is stored.
"""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
REPO = HERE.parent.parent

# This runs as a plain script, not under pytest, so add the folders pytest would add.
sys.path[:0] = [str(REPO / "legacy" / "src"), str(REPO / "tests")]

from cases import chain_11row, chain_21row, steane_41row
from qecnoise.Dmatrixsinglequbit import pair_family, two_qubit_family
from qecnoise.gordanLP import gordan


# Each case, with the error family its regression test uses.
CASES = [
    (chain_11row, pair_family(7)[0]),
    (chain_21row, pair_family(7)[0]),
    (steane_41row, two_qubit_family(7)[0]),
]


if __name__ == "__main__":
    for case, family in CASES:
        result = case()
        _, absorbed, _, _ = gordan(result["D"], result["measured"], family)

        path = HERE / f"{case.__name__}.npz"
        np.savez(
            path,
            D=result["D"],
            measured=result["measured"],
            absorbed=np.array(absorbed, dtype=int),
        )
        print(f"{path.name}: D {result['D'].shape}, "
              f"measured {result['measured'].shape}, {len(absorbed)} absorbed")
