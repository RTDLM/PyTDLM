---
hide:
  - navigation
---

# Quick start

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


