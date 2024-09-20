# Physics-informed kernel learning 

This project aims at illustrating the results of the paper _Physics-informed kernel learning_ (Nathan Doumèche, Francis Bach, Gérard Biau, and Claire Boyer). The paper is available at https://hal.science/hal-04701052. 

## Finite-element method implementation of the physics-informed kernel
The file _1-FEM_kernel_approximation.ipynb_ is a Jupyter notebook with the code necessary to compute the physics-informed kernel 
(see also _Physics-informed machine learning as a kernel method_, Nathan Doumèche, Francis Bach, Gérard Biau, and Claire Boyer, COLT 2024) by solving the kernel equation (3). 
This is the file used to plot Figure 1.

## Physics-informed kernel learning
The file _2-PIK package and effective dimension.ipynb_ is a Jupyter notebook with the code necessary to run the physics-informed kernel learning (PIKL) method relying on a Fourier approximation. 

The first section of the notebook contains the package usefull to implement the PIKL estimator and to estimate its effective dimension. It is designed to run both on CPU and on GPU, and 
for linear PDEs with constant coefficients in dimension $d=1$ and $d=2$. Let $m \in \mathbb N$ and consider the approximation of the kernel with Fourier modes of frequencies indexed by 
$\{-m,...,m\}^d$. This algorithm requires storing a $(2m+1)^{d}\times(2m+1)^{d}$-matrix, which can be computed with $(2m+1)^{2d}\times n$ operations, where $n$ is the sample complexity. 
The PIKL estimator is then computed through inverting this matrix, as evidenced by equation (7), which is done in $O((2m+1)^{2d})$ thanks to a LU solving algorithm (torch.linalg.solve).

The second section of this notebook computes the effective dimension of different PDEs on different domains. The lower the effective dimension $N(\lambda_n, \mu_n)$, 
the quickest the convergence rate (see equation (8)). This is where Figures 5, 6, 7, 9, 10, 11, and 12 are plotted.

The thrid section quantifies the performance of the PIKL algorithm. It is first evaluated in hybrid modelling with the heat equation and compared both to a Sobolev kernel method (without physics), 
and to a kernel method strongly enforcing the heat equation. This is where Figure 4 is plotted. 
Then, the PIKL is compared to a least-square linear regression in the space of the PDE solution for the harmonic oscillator ODE. This is where Figures 2 and 3 are plotted.

The fourth section implements sanity checks. For example, it checks that, when $n < m$ and when the regularizing parameters are set to zero, the kernel estimator indeed interpolates (the empirical risk is zero).

## Benchmarking PIKL

The comparison with PINNs and traditional PDE solvers on the wave equation is carried out in the _3-PIKs_experiment_Wave_equation.ipynb_ Jupyter notebook. 
This is where Figures 8, 13, and 14 are plotted, and Tables 2, 3, and 4 are computed.

The comparison with PINNs on the advection equation is carried out in the _Advection equation_ folder. This is where Tables 1 is computed.
