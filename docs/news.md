---
hide:
  - navigation
---

# What's new

## v0.2.2 - Mar 24, 2026
- Modify DCM : now uses PCM instead of UM
- Fix handling of zeroes in the marginals in DMC
- Add `eps` parameter to the `_DCM()` function: allows to chose the value used to replace zero values in the marginals of the DCM

## v0.2.1 - Nov 21, 2025
- Align parameter name in `extract_opportunity` with R package
- Fix `gof` to support results with probabilities
- Make arrays reconstructed from shared buffer read-only

## v0.2.0 - Oct 19, 2025
- Align with R package : add functions `run_law` and `run_model`, implement `average` option
- Refactor multiprocessing with `shared_memory`
- Add function `run_law_model_gof`
- Add function `run_optimization`
- Implement verbosity control

## v0.1.0 - Jun 19, 2025
Initial public release of **PyTDLM**


