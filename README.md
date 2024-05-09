# simpeg_gpu_solvers
Testing GPU solvers in SimPEG using CuPy and PyTorch

Typically, MKL Pardiso solver (via Pydiso) is the preffered solver for running simulations in SimPEG. 

Large speedups are possible for certain matrix operations using GPUs (https://cupy.dev). Whether solving $ **Ax = b** $ type problems involving matrix factorization of **A** and matrix-vector products encountered in geophysics are faster using GPUs. Here, LU solvers from CuPy and PyTorch are used to solve on GPUs compared against Pardiso and default SimPEG solver which uses SciPy.

#### Results:
For a problem **AX=b** with **A** matrix of size (28564, 28564) and _b_ of size 108
- Time to run simulation with MKL Pardiso solver: 2.631314992904663 seconds
- Time to run simulation with Pytorch LU solver: 16.436639070510864 seconds
- Time to run simulation with SimPEG (SciPy LU) solver: 11.663026332855225 seconds
- Time to run simulation with CuPy LU solver: 1068.3546755313873 seconds

Pardiso outperforms alternatives here. While matrix-vector products are faster on GPUs, much of the slowdown for PyTorch can be attributed to extremely slow matrix factorization using GPUs. As such, PyTorch solver can be useful in cases where matrix **A** does not change and large vector _b_ (large number of sources).

CuPy accounts for the slow matrix factorization on GPU by purportedly doing LU factorization on CPU using SciPy's LU routines and doing the matrix-vector products on GPUs. However, it was significantly slower in the tests. I attribute the slow performance of CuPy here to the fact that the CuPy solver operations are not yet implemented for sparse matrices and thus it factorizes a dense matrix while Pardiso and SimPEG solvers use sparse CSR matrices. 


## TODO
- Add install instructions