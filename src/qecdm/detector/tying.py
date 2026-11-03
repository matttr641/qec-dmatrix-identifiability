"""The tying matrix T: which faults share one rate.

T is (faults x parameters) with a single 1 in each row, so D'_tied = D'_raw @ T
adds up the columns of the faults that share a parameter. This is valid
because log<(-1)^row> is linear in theta = log(1 - 2p), one theta per fault.

A tying rule is a key function: faults with equal keys share a parameter.
- "rounds": stationary in time. A fault is tied with the same fault (same
  channel, position in the round, qubits and Pauli) in every other round. All
  resets of a qubit share one rate; prep and readout happen once per qubit.
- "channels": one rate per channel (idle, gate_1q, gate_2q, measure, reset,
  prep, readout), the same for every qubit, gate and round.

Faults in a `fixed` channel (by default the data preparation, which is
calibrated separately) are kept out of T and go to T_fixed, so a fit can move
their known contribution to the other side: y - D'_fixed theta_cal.
"""

import numpy as np
from scipy import sparse

from qecdm.detector.faults import CHANNELS


def key_rounds(fault):
    """Fault -> tuple: the per-round key. Ties a fault with its copies in every round."""
    if fault.channel in ("reset", "prep", "readout"):
        return (fault.channel, fault.qubits, fault.pauli)  # the same for every round
    return (fault.channel, fault.position, fault.qubits, fault.pauli)


def key_channels(fault):
    """Fault -> str: the per-channel key. Ties every fault of a channel."""
    return fault.channel


RULES = {"rounds": key_rounds, "channels": key_channels}


def _indicator(keys, rows, n):
    """The (n x len(names)) 0/1 matrix with a 1 at (row, its key's column), and the names."""
    names = list(dict.fromkeys(keys))  # each key once, in order of first appearance
    column = {key: j for j, key in enumerate(names)}
    T = sparse.csc_matrix((np.ones(len(rows), dtype=np.int64), (rows, [column[k] for k in keys])),
                          shape=(n, len(names)))
    return T, names


def tying_matrix(faults, rule="rounds", fixed=("prep",)):
    """list of Fault -> (T, names, T_fixed, fixed_names).

    `rule` is "rounds", "channels", or any function Fault -> hashable key.
    T is (faults x parameters): row j has a 1 in the column of fault j's key,
    and names[k] is column k's key. Faults in a `fixed` channel have an
    all-zero row in T and are tied, by the same rule, in T_fixed instead.
    """
    key = RULES[rule] if isinstance(rule, str) else rule
    unknown = set(fixed) - set(CHANNELS)
    if unknown:
        raise ValueError(f"unknown fixed channels {sorted(unknown)}: use {CHANNELS}")

    free = [j for j, f in enumerate(faults) if f.channel not in fixed]
    known = [j for j, f in enumerate(faults) if f.channel in fixed]
    T, names = _indicator([key(faults[j]) for j in free], free, len(faults))
    T_fixed, fixed_names = _indicator([key(faults[j]) for j in known], known, len(faults))
    return T, names, T_fixed, fixed_names
