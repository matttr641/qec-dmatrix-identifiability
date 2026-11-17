# qec-dmatrix-surface

Learning the noise of a QEC circuit from the parity statistics of its detectors.

- **Stabilizer version** (`src/qecdm/stabilizer/`): the thesis method. Rows are products of
  stabilizers, columns are syndrome classes; Gordan LPs certify which errors are absorbed.
- **Detector version** (`src/qecdm/detector/`): the circuit-level method. Rows are parities
  of connected sets of detectors, columns are tied fault rates; sliding windows and absorption.

Tested with Python 3.13, stim 1.16.0, pymatching 2.4.0, qldpc 0.4.1.

## Install

```
pip install -e ".[dev]"
```

## Detector pipeline

| step | module |
|---|---|
| noisy memory circuit (repetition, surface, BB) | `circuits.py` |
| faults and H (detectors x faults) | `faults.py` |
| tying T, signature counts M, target rank M | `tying.py` |
| design matrix D = D_sig M | `detector_dmatrix.py` |
| sample y, fit theta, convert to rates | `estimate.py` |
| windows, drift, long runs | `windows.py` |
| absorption of idle crosstalk | `absorption.py` |

## Quick start

```python
from qecdm.detector.circuits import surface_memory, uniform_noise
from qecdm.detector.windows import window_design, long_run_rates

noise = uniform_noise(0.005, readout=0.02)
circuit = surface_memory(3, rounds=200, noise=noise)
design, _ = window_design(circuit, start=3, size=2, rule="channels")
theta, rates = long_run_rates(design, circuit, size=2, shots=200, seed=1)
for group, rate in zip(design.groups, rates):
    print(group, round(rate, 4))   # about 0.005 each; measure + reset merged, about 0.01
```

## Results

```
python -m qecdm.stabilizer.thesis_on_surface
python -m qecdm.stabilizer.thesis_on_bb
python -m qecdm.detector.applied_absorption   # about 18 min, mostly BB72
```

## Tests

```
python -m pytest              # everything
python -m pytest -m "not slow"
```

## Conventions

- r: channel rate, p: fault probability, q = 1 - 2p, theta = log q.
- Tying rules: `rounds` (one rate per fault, the same in every round), `gates` (one
  depolarizing rate per gate), `channels` (one rate per channel).

## Known limitations

- The window size is chosen by hand (2 rounds in every case tested).
- `gates` leaves unlearnable directions; absorption raises on it.
- qldpc circuits (Steane, BB) only have detectors on the Z checks.
- No error bars yet.

## Legacy

`legacy/` holds the original code (`qecnoise@b70f114`), which reproduces the thesis results
for the 7-qubit bit-flip chain and the Steane code; the regression tests check against it.
