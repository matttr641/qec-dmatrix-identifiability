"""Estimating the rates from detector data, and the true values to check them against.

The model is log<(-1)^row> = D theta + D_fixed theta_fixed, with one
theta = log q per fault, where q = 1 - 2p and p is the probability of that one
fault (one Pauli of one channel). stim turns a channel of rate r into
independent faults exactly: a flip is one fault with p = r; DEPOLARIZE1(r) is 3
independent Paulis with q^2 = 1 - 4r/3; DEPOLARIZE2(r) is 15 with
q^8 = 1 - 16r/15.
"""

import numpy as np

from qecdm.detector.tying import RULES

NOISE_KEY = {"prep": "reset"}  # the data preparation uses the reset rate; every other channel its own

#Making our true value for our theta = sum(log(1-2p_i)) for parameters i in theta, eg. just idle or measure+reset

def fault_probability(fault, noise):
    """Fault, noise dict -> float: p, the probability of this one fault."""
    r = noise[NOISE_KEY.get(fault.channel, fault.channel)]
    if fault.channel in ("idle", "gate_1q"):  # DEPOLARIZE1: one of 3 Paulis
        return (1 - np.sqrt(1 - 4 * r / 3)) / 2
    if fault.channel == "gate_2q":            # DEPOLARIZE2: one of 15 Paulis
        return (1 - (1 - 16 * r / 15) ** (1 / 8)) / 2
    return r                                  # a single flip


def true_thetas(faults, noise, design, rule="rounds"):
    """list of Fault, noise dict, DetectorDesign -> (theta, theta_fixed): the true values.

    theta[k] is the sum of theta over the parameters merged into column k of
    design.D (all the data can see of them); theta_fixed[k] is that of
    design.fixed_names[k]. `rule` must be the rule the design was built with.
    """
    key = RULES[rule] if isinstance(rule, str) else rule
    theta = {}
    for fault in faults:
        value = np.log(1 - 2 * fault_probability(fault, noise))
        if not np.isclose(theta.setdefault(key(fault), value), value):
            raise ValueError(f"parameter {key(fault)} ties faults with different rates")
    return (np.array([sum(theta[name] for name in group) for group in design.groups]),
            np.array([theta[name] for name in design.fixed_names]))

#What is the exact data vector we should get

def exact_expectations(circuit, subsets):
    """stim circuit, list of detector tuples -> array: the exact <(-1)^row> of each subset.

    stim's detector error model lists independent errors, each with a
    probability p and the detectors it flips. An error flips a subset's parity
    when it flips an odd number of its detectors, and then multiplies
    <(-1)^row> by 1 - 2p; otherwise it leaves it alone.
    """
    sets = [set(subset) for subset in subsets]
    log_y = np.zeros(len(subsets))
    for error in circuit.detector_error_model().flattened():
        if error.type != "error":  # skip detector and observable declarations
            continue
        flipped = {t.val for t in error.targets_copy() if t.is_relative_detector_id()}
        theta = np.log(1 - 2 * error.args_copy()[0])
        log_y += [len(flipped & s) % 2 * theta for s in sets]
    return np.exp(log_y)

#Sampling the data vector

def sample_detectors(circuit, shots, seed=None):
    """stim circuit, int -> (shots x detectors) bool array: True where a detector fired."""
    return circuit.compile_detector_sampler(seed=seed).sample(shots)


def row_expectations(samples, subsets):
    """(shots x detectors) array, list of detector tuples -> array: the sampled <(-1)^row>.

    A shot gives a row +1 if an even number of its detectors fired and -1 if
    odd, so the average over the shots is 1 - 2 * (the fraction with odd parity).
    """
    y = np.zeros(len(subsets))
    for i, subset in enumerate(subsets):
        odd = samples[:, list(subset)].sum(axis=1) % 2 #sum over columns of detectors in subset for all shots 
        y[i] = 1 - 2 * odd.mean() #mean = pr(subset of detectors flipped), since 0=not flipped, 1=flipped
    return y

#Fitting the data vector and then getting the rates

def fit_thetas(design, y, theta_fixed):
    """DetectorDesign, row expectations, known thetas -> theta.

    Least squares on log y - D_fixed theta_fixed = D theta. Where D has a null
    space (see design.directions) this is the smallest theta that fits, so only
    D theta is determined.
    """
    if (y <= 0).any():
        raise ValueError("a row has y <= 0, so log y is undefined: use more shots or less noise")
    D_inverse = np.linalg.pinv(design.D, rcond=max(design.D.shape) * np.finfo(float).eps)
    return D_inverse @ (np.log(y) - design.D_fixed @ theta_fixed)
#Converting the thetas back to rates

DEPOLARIZING = {"idle": (2, 4 / 3), "gate_1q": (2, 4 / 3), "gate_2q": (8, 16 / 15)}  # (a, b): q^a = 1 - b r


def rates_from_thetas(design, theta):
    """DetectorDesign, theta per column of design.D -> array: the rate of each column.

    One parameter covering all of a channel's Paulis: the rate r of its channel.
    One parameter of a single Pauli or flip: its own probability p. Several merged
    parameters: only their sum of thetas is known, so the probability that an odd
    number of their faults fire. NaN for a column in one of design.directions,
    whose own theta is not determined by the data.
    """
    undetermined = [group for direction in design.directions for group, _ in direction]
    rates = np.full(len(design.groups), np.nan)
    for k, group in enumerate(design.groups):
        if group in undetermined:
            continue
        name = group[0]
        channel = name if isinstance(name, str) else name[0]  # "channels" keys are the channel itself
        one_pauli = not isinstance(name, str) and isinstance(name[-1], str)  # a "rounds" key ends in its Pauli
        a, b = DEPOLARIZING.get(channel, (1, 2)) if len(group) == 1 and not one_pauli else (1, 2)  # p = (1 - q) / 2
        rates[k] = (1 - np.exp(a * theta[k])) / b
    return rates
