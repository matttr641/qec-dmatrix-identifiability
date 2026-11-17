"""Examples for the paper, all on the 3-qubit bit-flip code.

The toy circuit: data qubits 0, 1, 2 and ancillas 3, 4 measuring Z0Z1 and Z1Z2.
Each round: idle noise on the data (DEPOLARIZE1), two CX layers, then a flip
before each ancilla measurement. Z on the data trips nothing and X, Y trip the
same detectors, so each data qubit has one bit flip d_k per round, with
probability 2r/3; m_j is the flip of ancilla j's measurement. No detectors on
the final readout, as in the paper.

Paper detector k + 1 is detector k here (0, 1 = round 1; 2, 3 = round 2; ...).

Run: python paper/examples.py
"""

import numpy as np
import stim

from qecdm.detector.circuits import surface_memory
from qecdm.detector.detector_dmatrix import detector_design_matrix
from qecdm.detector.estimate import (exact_expectations, fault_probability, fit_thetas, rates_from_thetas,
                                     row_expectations, sample_detectors, true_thetas)
from qecdm.detector.faults import circuit_faults
from qecdm.detector.tying import merge_parameters, signature_counts, tying_matrix
from qecdm.detector.windows import (drifting_circuit, long_run_rates, track_rates, true_window_thetas,
                                    window_design)

NOISE = {"idle": 0.045, "measure": 0.02}  # bit flip 2r/3 = 0.03, measurement flip 0.02
UNTIED = lambda fault: (fault.offset, fault.qubits, fault.pauli)
np.set_printoptions(linewidth=150)


def toy_circuit(rounds, noise=NOISE):
    """int -> stim.Circuit: the bit-flip memory experiment, `rounds` rounds."""
    c = stim.Circuit("R 0 1 2 3 4\nTICK")
    for t in range(rounds):
        c.append("DEPOLARIZE1", [0, 1, 2], noise["idle"])
        c.append("TICK")
        c.append("CX", [0, 3, 1, 4])
        c.append("TICK")
        c.append("CX", [1, 3, 2, 4])
        c.append("TICK")
        c.append("X_ERROR", [3, 4], noise["measure"])
        c.append("MR", [3, 4])
        for j in (2, 1):  # D_{t,0} then D_{t,1}: this round's outcome XOR the last round's
            c.append("DETECTOR", [stim.target_rec(-j)] + ([stim.target_rec(-j - 2)] if t else []))
        c.append("TICK")
    c.append("M", [0, 1, 2])
    return c


def drift(rounds):
    """The drift used everywhere: the idle rate ramps from 0.03 to 0.09; measurement stays at 0.02."""
    return lambda t: {"idle": 0.03 + 0.06 * (t - 1) / max(rounds - 1, 1), "measure": 0.02}


# --- helpers for printing -----------------------------------------------------------------

def short(fault, with_round):
    """Fault -> "d0", "m1", or with the round "d0^2"."""
    name = f"d{fault.qubits[0]}" if fault.channel == "idle" else f"m{fault.qubits[0] - 3}"
    return f"{name}^{fault.round}" if with_round else name


def column_names(design, faults, key, with_round):
    """One name per column of design.D: its faults, X and Y together, joined by +."""
    names = []
    for group in design.groups:
        members = {short(f, with_round) for f in faults if key(f) in group}
        names.append("+".join(sorted(members)))
    return names


def show(title, matrix, rows, columns):
    """Print a matrix with its rows labelled by detector sets, numbered from 1 as in the paper."""
    print(f"\n{title}")
    print(" " * 10 + " ".join(f"{c:>9}" for c in columns))
    for row, values in zip(rows, matrix):
        label = "{" + ",".join(str(int(d) + 1) for d in row) + "}"
        print(f"{label:>10}" + " ".join(f"{v:>9}" for v in values))


def rank_walk(design):
    """Rank of D after each order of rows."""
    orders = sorted({len(s) for s in design.subsets})
    return {k: np.linalg.matrix_rank(design.D[[i for i, s in enumerate(design.subsets) if len(s) <= k]])
            for k in orders}


def table(columns, **values):
    """One line per column of D, one column per named array."""
    print(f"{'':>12}" + "".join(f"{name:>10}" for name in values))
    for k, column in enumerate(columns):
        print(f"{column:>12}" + "".join(f"{vals[k]:>10.4f}" for vals in values.values()))


def exact_and_sampled(circuit, design, theta_true, shots, seed=1):
    """Fit exact y and sampled y; return both thetas."""
    exact = fit_thetas(design, exact_expectations(circuit, design.subsets), np.zeros(0))
    y = row_expectations(sample_detectors(circuit, shots, seed), design.subsets)
    return exact, fit_thetas(design, y, np.zeros(0))


def drifting_truth(design, faults, key, noise_at):
    """The theta of each column when every fault has its own round's rate (the sum over its faults)."""
    theta = {}
    for f in faults:
        theta.setdefault(key(f), np.log(1 - 2 * fault_probability(f, noise_at(f.round))))
    return np.array([sum(theta[name] for name in group) for group in design.groups])


# --- the examples ---------------------------------------------------------------------------

def untied(rounds=2, shots=200_000, drifting=False):
    """Section 2: one column per signature; with drift every round's rate is recovered."""
    print(f"\n=== Untied, {rounds} rounds, {'drift' if drifting else 'no drift'} ===")
    base = toy_circuit(rounds)
    circuit = drifting_circuit(base, drift(rounds)) if drifting else base
    _, faults = circuit_faults(base)
    design = detector_design_matrix(base, rule=UNTIED, fixed=())
    names = column_names(design, faults, UNTIED, with_round=True)
    print(f"faults {len(faults)} (X and Y counted separately), columns (signatures) {design.D.shape[1]}, "
          f"rows {design.D.shape[0]}, target {design.target}, rank by order {rank_walk(design)}")
    if not drifting:
        show("D (untied)", design.D, design.subsets, names)

    truth = (drifting_truth(design, faults, UNTIED, drift(rounds)) if drifting
             else true_thetas(faults, NOISE, design, UNTIED)[0])
    exact, sampled = exact_and_sampled(circuit, design, truth, shots)
    print(f"\nexact fit - truth: max |theta error| = {np.abs(exact - truth).max():.1e}")
    print(f"rates per column, sampled from {shots} shots:")
    result = {"names": names, "truth": rates_from_thetas(design, truth), "sampled": rates_from_thetas(design, sampled)}
    table(names, truth=result["truth"], sampled=result["sampled"])
    return result


def tied(rounds=2, shots=200_000, drifting=False):
    """Section 3: "rounds" tying, T and M; with drift the fit is a weighted average and misfits."""
    print(f"\n=== Tied (rounds), {rounds} rounds, {'drift' if drifting else 'no drift'} ===")
    base = toy_circuit(rounds)
    circuit = drifting_circuit(base, drift(rounds)) if drifting else base
    H, faults = circuit_faults(base)
    design = detector_design_matrix(base, rule="rounds", fixed=())
    names = column_names(design, faults, lambda f: (f.channel, f.position, f.qubits, f.pauli), with_round=False)
    print(f"faults {len(faults)}, parameters {design.D.shape[1]}, rows {design.D.shape[0]}, "
          f"target {design.target}, rank by order {rank_walk(design)}")
    if not drifting:
        T, keys, _, _ = tying_matrix(faults, "rounds", ())
        M, signatures = signature_counts(H, T)
        M, _ = merge_parameters(M, keys)
        show("M (signatures x parameters)", M, signatures, names)
        show("D = D_sig M", design.D, design.subsets, names)

    if drifting:
        y = exact_expectations(circuit, design.subsets)
        theta = fit_thetas(design, y, np.zeros(0))
        residual = np.log(y) - design.D @ theta
        print(f"\nexact data, tied fit: max |residual| = {np.abs(residual).max():.1e} (nonzero: the model is wrong)")
        flips = [2 * drift(rounds)(t)["idle"] / 3 for t in range(1, rounds + 1)]
        print("true bit-flip rate per round:", np.round(flips, 4))
        table(names, fitted=rates_from_thetas(design, theta))
        return {"names": names, "fitted": rates_from_thetas(design, theta), "per_round": np.array(flips)}
    truth = true_thetas(faults, NOISE, design, "rounds")[0]
    exact, sampled = exact_and_sampled(circuit, design, truth, shots)
    print(f"\nexact fit - truth: max |theta error| = {np.abs(exact - truth).max():.1e}")
    result = {"names": names, "truth": rates_from_thetas(design, truth), "sampled": rates_from_thetas(design, sampled)}
    table(names, truth=result["truth"], sampled=result["sampled"])
    return result


# --- windows: the toy, and the rotated surface code ----------------------------------------

SURFACE_NOISE = {"idle": 0.003, "gate_1q": 0.005, "gate_2q": 0.005, "measure": 0.01, "reset": 0.005,
                 "readout": 0.02}


def surface_drift(rounds):
    """Surface drift: the gate rates (1q and 2q together) ramp from 0.003 to 0.009; the rest stay put."""
    def noise_at(t):
        gate = 0.003 + 0.006 * (t - 1) / max(rounds - 1, 1)
        return {**SURFACE_NOISE, "gate_1q": gate, "gate_2q": gate}
    return noise_at


# code -> (circuit builder, noise, drift, tying rule, the column to follow)
CODES = {
    "toy": (toy_circuit, NOISE, drift, "rounds", "d0"),
    "surface3": (lambda r: surface_memory(3, r, SURFACE_NOISE), SURFACE_NOISE, surface_drift, "channels", "gate_2q"),
    "surface5": (lambda r: surface_memory(5, r, SURFACE_NOISE), SURFACE_NOISE, surface_drift, "channels", "gate_2q"),
}


def window(code, rounds):
    """The window D for rounds 3-4 of a long circuit, its column names, and the setup for `code`."""
    build, noise, drift_of, rule, follow = CODES[code]
    base = build(rounds)
    design, _ = window_design(base, 3, 2, rule)
    if code == "toy":
        _, faults = circuit_faults(base)
        names = column_names(design, faults, lambda f: (f.channel, f.position, f.qubits, f.pauli), with_round=False)
    else:
        names = ["+".join(group) for group in design.groups]
    return base, design, names, noise, drift_of(rounds), follow


def multi_shot(code="toy", rounds=60, shots=20_000, stride=3, drifting=False):
    """3i: fit each window position separately, averaging over shots."""
    print(f"\n=== {code}: windows, multi-shot (3i), {rounds} rounds, {'drift' if drifting else 'no drift'} ===")
    base, design, names, noise, drift_at, follow = window(code, rounds)
    print(f"window D {design.D.shape}, target {design.target}, rank by order {rank_walk(design)}")
    noise_at = drift_at if drifting else (lambda t: noise)
    circuit = drifting_circuit(base, noise_at) if drifting else base
    starts, _, rates = track_rates(design, circuit, size=2, shots=shots, stride=stride, seed=1)
    truth = np.array([rates_from_thetas(design, th)
                      for th in true_window_thetas(design, base, noise_at, 2, starts)])
    result = {"names": names, "starts": np.array(starts), "fitted": rates, "truth": truth, "follow": follow}
    if not drifting:
        print(f"{len(starts)} independent positions (stride {stride}), {shots} shots each")
        table(names, truth=truth[0], mean=rates.mean(axis=0), std=rates.std(axis=0))
        return result
    print(f"{follow} at each window position: fitted vs true (window's weighted average)")
    k = names.index(follow)
    print(f"{'start':>6} {'fitted':>8} {'true':>8}")
    for i in range(0, len(starts), max(len(starts) // 8, 1)):
        print(f"{starts[i]:>6} {rates[i, k]:>8.4f} {truth[i, k]:>8.4f}")
    return result


def long_run(code="toy", rounds=200, shots=1000, drifting=False, lasts=(5, 20, None)):
    """3ii: average the rows over the shifted windows of each run, then fit once."""
    print(f"\n=== {code}: windows, one long run (3ii), {rounds} rounds, {shots} shots, "
          f"{'drift' if drifting else 'no drift'} ===")
    base, design, names, noise, drift_at, follow = window(code, rounds)
    noise_at = drift_at if drifting else (lambda t: noise)
    circuit = drifting_circuit(base, noise_at) if drifting else base
    stride = 3
    every = list(range(2, rounds, stride))  # the starts long_run_rates uses
    results = []
    for last in lasts:
        _, rates = long_run_rates(design, circuit, size=2, last=last, stride=stride, shots=shots, seed=1)
        print(f"\nlast = {last or 'all'} windows")
        if drifting:
            starts = every[-last:] if last else every
            true_avg = rates_from_thetas(design, true_window_thetas(design, base, noise_at, 2, starts, average=True))
            now = rates_from_thetas(design, true_window_thetas(design, base, noise_at, 2, starts[-1:])[0])
            table(names, fitted=rates, true_avg=true_avg, true_now=now)
            results.append({"last": last, "fitted": rates, "true_avg": true_avg, "true_now": now})
        else:
            truth = rates_from_thetas(design, true_window_thetas(design, base, noise_at, 2, [3])[0])
            table(names, truth=truth, fitted=rates, error=np.abs(rates - truth))
            results.append({"last": last, "fitted": rates, "truth": truth})
    return {"names": names, "windows": len(every), "runs": results, "follow": follow}


if __name__ == "__main__":
    untied()
    untied(rounds=6, drifting=True)
    tied()
    tied(rounds=6, drifting=True)
    multi_shot()
    multi_shot(drifting=True)
    long_run()
    long_run(drifting=True)
    for code in ("surface3", "surface5"):
        multi_shot(code)
        multi_shot(code, drifting=True)
        long_run(code, rounds=100)
        long_run(code, rounds=100, drifting=True)
