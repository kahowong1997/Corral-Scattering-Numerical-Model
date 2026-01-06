# Corral-Scattering-Numerical-Model
Numerical framework for computing Green's Functions under renormalization from scattering potentials. Optimized for Magnet-Superconductor Hybrid (MSH) systems using Python (Numba).

# Project Overview
This framework provides a suite of high-performance tools designed to obtain the topological Hamiltonian for MSH systems. The core of the project focuses on the Renormalization of the zero energy Green's Functions due to corral scattering potentials, allowing for the precise identification of topological phases and Majorana signatures in real space.

The framework is built on analytical foundations, using Cauchy's residue theorem to bypass the computational bottlenecks of standard numerical integration.

# Framework Modules

## 1. Analytical Kernel (`MSH_Analytical_Kernel.py`)
This is the foundational math engine of the framework. It provides a suite of high-performance tools to solve the underlying complex analysis required for Green's Function construction.

* **Characteristic Polynomial Solver:** Computes coefficients $c_n$ for the Hamiltonian's characteristic equation.
* **Companion Matrix Pole-Finding:** Identifies complex poles $z_i$ via an eigenvalue approach, ensuring numerical stability compared to standard root-finding.
* **Residue Components:** Analytically evaluates the denominator $P'(z)$ and the adjugate matrix $\text{Adj}(H(z))$ to bypass computationally expensive numerical integration.

---

## 2. Quasi-1D Module (`MSH_Quasi_1D.py`)
This module utilizes the analytical kernel to solve for systems with longitudinal translational symmetry (infinite strips) and finite transverse width.

* **Hybrid Propagator Construction:** Directly computes the $(k_x, Y)$ Green's Function. It utilizes the Hermitian relation $G(k_x, -Y) = [G(k_x^*, Y)]^\dagger$ to resolve the propagator across the entire spatial domain while maintaining validity in the analytical $Y > 0$ half-plane.
* **Two-Site T-Matrix Renormalization:** Implements a scattering formalism for corral boundaries (top and bottom). It features a **Dual-Regime Solver** that switches inversion methods based on the scattering potential strength ($V$) to avoid precision loss in the strong-coupling limit.
* **Effective Hamiltonian Mapping:** Extracts the renormalized quasiparticle spectrum through self-energy mapping ($H_{\text{eff}} = -G^{-1} + J\tau_0\sigma_z$). This allows for the identification of topological phases and Majorana signatures in the hybrid representation.

---

## 🛠 Performance & Numerical Optimizations
* **Numba JIT Acceleration:** The entire pipeline is JIT-compiled to achieve C-level execution speeds for large-scale parameter sweeps.
* **Memory Contiguity Management:** Explicit use of `np.ascontiguousarray` for all matrix inversions ensures the `@` (matmul) operator utilizes optimal BLAS/LAPACK pathways without memory-copy overhead.

---

## 📐 Mathematical Definition
The engine transforms the momentum-space Hamiltonian into the hybrid $(k_x, Y)$ representation:

$$G(k_x, Y) = \frac{\sqrt{3}}{4\pi} \int_{-\frac{2\pi}{\sqrt{3}}}^{\frac{2\pi}{\sqrt{3}}} G(k_x, k_y) e^{ik_y Y} dk_y$$

By identifying the poles $z_i$ of the characteristic polynomial $P(z)$ where $z = e^{ik_ya_y}$, the integral is computed via the residue sum:

$$G(k_x, Y) = \sum_{z_i < 1} \text{Res}\left[-\frac{\text{Adj}(H(z))}{P(z)} z^{n_y-1}, z_i\right]$$
