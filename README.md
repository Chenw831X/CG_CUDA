# CG_CUDA
Conjugate Gradient Method to solve linear system using CUDA

Conjugate gradient method(CG) is a common iterative method used to solve sparse linear systems. The theoretical knowledge of CG can be referred to [An Introduction to the Conjugate Gradient Method Without the Agonizing PainMarch 1994](https://dl.acm.org/doi/book/10.5555/865018)

CG_CUDA is implemented through CUDA to achieve GPU parallelism, which can speed up the solution. CG_CUDA needs to call the CUBLAS library and CUSPARSE library in CUDA.

CG_CUDA can be compiled with the following command:  
```
nvcc CG.cu -lcublas -lcusparse -o CG
```
