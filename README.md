# __PyTDLM__ <img src="docs/assets/logo.png" align="right" alt="" width="138" />

# Systematic comparison of trip distribution laws and models in Python

<!-- badges: start -->
[![Test Build and Distribution](https://github.com/RTDLM/PyTDLM/actions/workflows/test-builds.yml/badge.svg)](https://github.com/RTDLM/PyTDLM/actions/workflows/test-builds.yml)
[![PyPI Version](https://img.shields.io/pypi/v/pytdlm.svg)](https://pypi.org/project/pytdlm)
[![PyPI Downloads](https://img.shields.io/pypi/dm/pytdlm.svg)](https://pypistats.org/packages/pytdlm)
[![Conda Version](https://img.shields.io/conda/v/conda-forge/pytdlm.svg)](https://anaconda.org/conda-forge/pytdlm)
[![Conda Downloads](https://img.shields.io/conda/d/conda-forge/pytdlm)](https://anaconda.org/conda-forge/pytdlm)

<!-- badges: end -->

## Description

The main purpose of these packages is to provide a rigorous framework for 
fairly comparing trip distribution laws and models, as described in 
[Lenormand *et al.* (2016)](https://doi.org/10.1016/j.jtrangeo.2015.12.008). 
This general framework relies on a two-step approach to generate mobility flows, 
separating the trip distribution law, gravity or intervening opportunities, from 
the modeling approach used to derive flows from this law. 

To make this framework more accessible, we developed both an 
[R package](https://rtdlm.github.io/TDLM/) 
and a [Python package](https://rtdlm.github.io/PyTDLM/), which replace the 
original [Java scripts](https://github.com/maximelenormand/Trip-distribution-laws-and-models) 
and extend their functionality.

[PyTDLM](https://rtdlm.github.io/PyTDLM/) is a Python port of the 
[TDLM R package](https://github.com/RTDLM/TDLM), with 
numpy-based implementations and parallel processing support for multiple 
exponent values. 

## Installation

### Using conda
```bash
conda install -c conda-forge pytdlm
```

### Using pip
```bash
pip install PyTDLM
```

### From source
```bash
git clone https://github.com/PyTDLM/TDLM.git
cd PyTDLM
pip install -e .
```

## Quick start

```python
import numpy as np
from TDLM import tdlm

# Prepare your data
mi = np.array([100, 200, 150])  # Origin masses
mj = np.array([80, 180, 120])   # Destination masses
dij = np.array([[0, 10, 15],    # Distance matrix
                [10, 0, 8], 
                [15, 8, 0]])
Oi = np.array([50, 80, 60])     # Out-trips
Dj = np.array([40, 90, 50])     # In-trips
Tij_observed = np.array([[0, 25, 25],  # Observed trip matrix
                         [30, 0, 50],
                         [35, 35, 0]])

# Run simulation
exponent = np.arange(0.1, 1.01, 0.01)
results = tdlm.run_law_model(
    law='NGravExp',
    mass_origin=mi,
    mass_destination=mj, 
    distance=dij,
    exponent=exponent,
    model='DCM',
    out_trips=Oi,
    in_trips=Dj,
    repli=100
)

# Calculate goodness-of-fit
gof_results = tdlm.gof(sim=results, obs=Tij_observed, distance=dij)

# Print results for a given exponent
print(gof_results[0.1].to_markdown(index=False))
```

## Documentation

For detailed documentation and examples, visit: 
[https://rtdlm.github.io/PyTDLM/](https://rtdlm.github.io/PyTDLM/)

## Citation

If you use this library in your research, please cite: [Reference to come].

```bibtex
@software{PyTDLM,
  author = {Perrier, R., Gargiulo, G., Jayet, C. and Lenormand, M.},
  title = {PyTDLM: Systematic comparison of trip distribution laws and models in Python},
  year = {2025},
  note = {Reference forthcoming}
}
```
