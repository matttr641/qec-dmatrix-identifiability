"""The thesis test cases, built once and shared by the tests and the fixture generator.

Each function returns a dict with:
    "D"         the design matrix D' (rows = measured stabilizers, columns = syndrome classes)
    "merged"    the single-qubit errors each column of D' stands for
    "measured"  the measured stabilizers, shape (rows, 2, n)
"""

from qecnoise.Dmatrixsinglequbit import bit_flip_chain, design_matrix


def chain_11row():
    """7-qubit bit-flip chain. Algorithm 1 stops when rank(D') = |C*|, giving 11 rows."""
    D, merged, measured = design_matrix(7, 6, bit_flip_chain(7))
    return {"D": D, "merged": merged, "measured": measured}


def chain_21row():
    """7-qubit bit-flip chain, walked to every subset size: the complete weight-2 set (21 rows)."""
    D, merged, measured = design_matrix(7, 6, bit_flip_chain(7), max_order=6)
    return {"D": D, "merged": merged, "measured": measured}


def steane_generators():
    """The 6 Steane [[7,1,3]] generators: the 3 Hamming parity checks as X-type, then as Z-type."""
    hamming = [
        [0, 0, 0, 1, 1, 1, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [1, 0, 1, 0, 1, 0, 1],
    ]
    none = [0] * 7
    x_type = [[row, none] for row in hamming]
    z_type = [[none, row] for row in hamming]
    return x_type + z_type


def steane_41row():
    """Steane code. Algorithm 1 stops when rank(D') = |C*|, giving 41 rows."""
    D, merged, measured = design_matrix(7, 6, steane_generators())
    return {"D": D, "merged": merged, "measured": measured}
