# TDLM: Systematic comparison of trip distribution laws and models

A Python port of the TDLM R package, with numpy-based implementations and parallel processing support for multiple exponent values.

## Overview

TDLM provides implementations of several trip distribution models commonly used in transportation planning and spatial analysis:

### Available Laws
- **GravExp**: Gravity model with exponential distance decay
- **NGravExp**: Normalized gravity model with exponential decay
- **GravPow**: Gravity model with power distance decay
- **NGravPow**: Normalized gravity model with power decay
- **Schneider**: Schneider's intervening opportunities model
- **Rad**: Radiation model
- **RadExt**: Extended radiation model
- **Rand**: Random model (baseline)

### Available Models
- **UM**: Unconstrained Model
- **PCM**: Production Constrained Model
- **ACM**: Attraction Constrained Model
- **DCM**: Doubly Constrained Model

## Installation

### Using conda (recommended)
```bash
conda install -c conda-forge tdlm
```

### Using pip
```bash
pip install TDLM
```

### From source
```bash
git clone https://github.com/PyTDLM/TDLM.git
cd TDLM
pip install -e .
```

## Quick Start

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
```

## Documentation

For detailed documentation and examples, visit: [https://pytdlm.github.io/](https://pytdlm.github.io/)

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this library in your research, please cite:

```bibtex
@software{lenormand2024tdlm,
  title={TDLM: Trip Distribution Law Models},
  author={Lenormand, Maxime},
  year={2024},
  url={https://github.com/PyTDLM/TDLM}
}
```
