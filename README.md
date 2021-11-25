# GW

Python-based implementation of the GW methods implemented in NWChem (see [https://arxiv.org/abs/2107.10423](https://arxiv.org/abs/2107.10423)). Originally based on Alex Kunitsa's [GW-approximation notebook](https://github.com/aakunitsa/GW-approximation).

**Dependencies**

- Python 3
- Numpy
- SciPy
- Psi4 


**Contents**

- cdgw.py contains the Contour-Deformation GW approach
- sdgw.py contains the Spectral-Decomposition GW approach


**Tests**

The outputs in the test directory were obtained with
```
cdgw.py n2.xyz --noqpa -1 --nvqpa 10 --ieta 0.01 --evgw --xcfun pbe
```
and
```
sdgw.py n2.xyz --noqpa -1 --nvqpa 10 --ieta 0.01 --evgw --xcfun pbe
```
