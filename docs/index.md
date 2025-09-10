---
hide:
  - navigation
---

# PyTDLM: Systematic comparison of trip distribution laws and models in Python

## Description

A Python port of the [TDLM R package](https://github.com/RTDLM/TDLM), with 
numpy-based implementations and parallel processing support for multiple 
exponent values.

[PyTDLM](https://rtdlm.github.io/PyTDLM/) provides implementations of several 
trip distribution models commonly used in transportation planning and 
spatial analysis:

### Available laws
- **GravExp**: Gravity model with exponential distance decay
- **NGravExp**: Normalized gravity model with exponential decay
- **GravPow**: Gravity model with power distance decay
- **NGravPow**: Normalized gravity model with power decay
- **Schneider**: Schneider's intervening opportunities model
- **Rad**: Radiation model
- **RadExt**: Extended radiation model
- **Rand**: Random model (baseline)

### Available models
- **UM**: Unconstrained Model
- **PCM**: Production Constrained Model
- **ACM**: Attraction Constrained Model
- **DCM**: Doubly Constrained Model

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


