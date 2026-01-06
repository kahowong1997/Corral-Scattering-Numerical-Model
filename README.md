# Corral-Scattering-Numerical-Model
Numerical framework for computing Green's Functions under renormalization from scattering potentials. Optimized for Magnet-Superconductor Hybrid (MSH) systems using Python (Numba).

# Project Overview
This framework provides a suite of high-performance tools designed to obtain the topological Hamiltonian for MSH systems. The core of the project focuses on the Renormalization of the zero energy Green's Functions due to corral scattering potentials, allowing for the precise identification of topological phases and Majorana signatures in real space.

The framework is built on analytical foundations, using Cauchy's residue theorem to bypass the computational bottlenecks of standard numerical integration. The core of this framework is its ability to analytically solve for the hybrid Green's function $G(k_x, Y)$ for Quasi-1D system with horizontal corral and efficiently compute $G(X,Y)$ for 2D system in real space with arbitrary corral, allowing the definition of an effective topological Hamiltonian for a MSH chain with magnetic adatoms deposited on a superconducting substrate renormalized by the presence of quantum corral.

## Theoretical Foundation: The Rashba Superconducting Substrate

The framework is optimized for a Rashba superconductor on a triangular lattice. The base Hamiltonian is defined in the Nambu basis $\Psi_{\mathbf{k}} = (c_{\mathbf{k}\uparrow}, c_{\mathbf{k}\downarrow}, c_{-\mathbf{k}\downarrow}^\dagger, -c_{-\mathbf{k}\uparrow}^\dagger)^T$:

$$H_0(\mathbf{k}) = \xi_{\mathbf{k}}\tau_z\sigma_0 + \tau_{z}(\vec{\alpha}(\mathbf{k})\times\vec{\sigma})^{z}+ \Delta\tau_x\sigma_0$$

Where:
* **Kinetic Term:** $\xi_{\mathbf{k}}=-2t\cos{k_x}-4t\cos{\frac{k_x}{2}}\cos{\frac{\sqrt{3}}{2}k_y}$ accounts for the tight-binding hopping on the triangular lattice.
* **Rashba SOC:** The spin-orbit coupling induced by the substrate is defined by the vector field $\vec{\alpha}(\mathbf{k}) = (\alpha^x_{\mathbf{k}}, \alpha^y_{\mathbf{k}})$, where:
  - $$\alpha^x_{\mathbf{k}} = 2\alpha \left( \sin k_x + \sin \frac{k_x}{2} \cos \frac{\sqrt{3} k_y}{2} \right)$$
  - $$\alpha^y_{\mathbf{k}} = 2\sqrt{3}\alpha \cos \frac{k_x}{2} \sin \frac{\sqrt{3} k_y}{2}$$

* **Superconducting Gap:** $\Delta$ is the $s$-wave superconducting order parameter.

### **Renormalization & Exchange Coupling**
The framework follows a two-stage process to arrive at the final topological state:
1. **Dressing the Substrate:** The bare Green's function of the superconductor is renormalized by the corral scattering potential ($V$) via the T-matrix formalism. This creates a "Dressed" propagator $G_{\text{dressed}}$ that accounts for the corral geometry.
2. **Ferromagnetic Adatoms ($J$):** The magnetic exchange coupling is then incorporated into the inverse of the dressed propagator. This represents the magnetic moments interacting with a pre-renormalized superconducting background, leading to an effective topological Hamiltonian:

$$H_{\text{eff}}(k_x) = -[G_{\text{dressed}}(k_x,\omega=0)]^{-1} + J\tau_0\sigma_z$$

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
* **Effective Hamiltonian Mapping:** Extracts the renormalized quasiparticle spectrum through self-energy mapping ($H_{\text{eff}} = -G^{-1} + J\tau_0\sigma_z$). This allows for the identification of topological phases and Majorana signatures using Pfaffians and winding number in the hybrid representation.

---

## 3. 2D Real-Space Module (`MSH_2D_Library.py`)
Optimized for the simulation of massive and arbitrary corral geometries on substrate with periodic boundary condition.

* **Symmetry-Folded Library Construction:** Utilizes the Hermitian relation of the Green's function to halve the integration search space. Unique spatial displacements are sorted and filtered to achieve $O(1)$ lookup times during global matrix assembly.
* **Eigen-Basis T-Matrix Solver:** Decomposes the scattering environment into decoupled chiral sectors. This enables analytical T-matrix generation via eigenvector scaling, bypassing the numerical instability of large-scale matrix inversions near resonance.
* **Real-Space Topological Diagnostic:** Computes the localized effective Hamiltonian ($H_{\text{top}}$) of a "dressed" substrate, providing the backend for computation of local winding number density for topological characterization and Majorana Bound State (MBS) wavefunction visualization.

---
## 📈 Results & Verification

This framework is currently being utilized for a forthcoming publication on topological phase transitions in MSH systems. 

**Verification and Numerical Integrity**
The model is continuously validated through a dual-testing suite to ensure both mathematical precision and physical consistency.

### **1. Kernel Validation (`verify_hybrid_kernel.py`)**
This test verifies the foundational **Analytical Residue Engine** against a brute-force numerical integration.
* **Methodology**: We compare the hybrid Green’s function $G(k_x, Y)$ generated via Cauchy’s Residue Theorem against a high-resolution Simpson-rule integration of the momentum-space propagator $[- H_0(\mathbf{k})]^{-1}$.
* **Accuracy**: The analytical approach maintains an error margin within $10^{-14}$, reaching the limit of double-precision complex math.
* **Significance**: This ensures that the companion-matrix pole solver and adjugate evaluations are exact before any renormalization is applied.



### **2. Topological Hamiltonian 1D Limit (`verify_Topo_Ham_1D.py`)**
This test verifies the physical validity of the **T-matrix renormalization** and the **Quasi-1D geometry**.
* **Methodology**: By setting the corral width to two lattice spacings ($W_c = \sqrt{3}$) and the scattering potential to the hard-wall limit ($V \to \infty$), we "pinch" the quasi-1D strip into a strictly 1D wire.
* **Physical Identity**: The resulting numerical $4 \times 4$ matrix is compared directly against the analytical 1D Shiba-chain Hamiltonian. 
* **Significance**: Passing this test confirms that the Dyson equation correctly integrates out the substrate degrees of freedom to recover known 1D topological physics.

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
