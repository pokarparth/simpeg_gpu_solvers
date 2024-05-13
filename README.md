# Testing GPU solvers in SimPEG using CuPy and PyTorch

Typically, MKL Pardiso solver (via Pydiso) is the preffered solver for running simulations in SimPEG. 

Large speedups are possible for certain matrix operations using GPUs (https://cupy.dev). I test whether solving $ **Ax = b** $ type problems involving matrix factorization of **A** and matrix-vector products encountered in geophysics are faster using GPUs. Here, LU solvers from CuPy and PyTorch are used to solve on GPUs compared against Pardiso and default SimPEG solver which uses SciPy.

#### Results:
For a problem **AX=b** with **A** matrix of size (28564, 28564) and _b_ of size 108
- Time to run simulation with MKL Pardiso solver: 2.631314992904663 seconds
- Time to run simulation with Pytorch LU solver: 16.436639070510864 seconds
- Time to run simulation with SimPEG (SciPy LU) solver: 11.663026332855225 seconds
- Time to run simulation with CuPy LU solver: 1068.3546755313873 seconds

Pardiso outperforms alternatives here. While matrix-vector products are faster on GPUs, much of the slowdown for PyTorch can be attributed to extremely slow matrix factorization using GPUs. As such, PyTorch solver can be useful in cases where matrix **A** does not change and large vector _b_ (large number of sources).

CuPy accounts for the slow matrix factorization on GPU by purportedly doing LU factorization on CPU using SciPy's LU routines and doing the matrix-vector products on GPUs. However, it was significantly slower in the tests. I attribute the slower performance of CuPy and Pytorch here to the fact that the CuPy and Pytorch solver operations are not yet implemented for sparse matrices [1,2] and thus it factorizes a dense matrix. On the other hand, Pardiso and SimPEG solvers use sparse CSR matrices. 

### Installation instructions
To run the project, follow these installation instructions:

1. Create a conda environment using the provided environment.yml file:

    ```
    conda env create -f environment.yml
    ```

2. Copy files `solver_utils_cupy.py` and `solver_utils_pytorch.py` from `simpeg_utils` folder and paste it in the installed SimPEG package by browsing to `<your_conda_install_path>\envs\solvers-test\Lib\site-packages\SimPEG\utils`.

3. Copy the file `matrix_utils_cupy.py` from `discretize_edits` folder and paste it in the installed Discretize package by browsing to `<your_conda_install_path>\envs\solvers-test\Lib\site-packages\discretize\utils`

4. Browse to `<your_conda_install_path>\envs\solvers-test\Lib\site-packages\cupyx` and replace `_init_.py` in `<your_conda_install_path>\envs\cupy\Lib\site-packages\cupyx\scipy\sparse` with `_init_.py` in folder `scipy-sparse-init`

5. Replace `_init_.py` in `<your_conda_install_path>\envs\cupy\Lib\site-packages\cupyx\scipy\sparse\linalg` with `_init_.py` in folder `scipy-sparse-init/scipy-sparse-linalg-init`

### References
[1] https://github.com/pytorch/pytorch/issues/69538
[2] https://discuss.pytorch.org/t/solving-ax-b-for-sparse-tensors-preferably-with-backward/102331
