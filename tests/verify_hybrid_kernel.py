import sys
import os
import numpy as np
import numba
from scipy.integrate import simpson
from collections import namedtuple

# Adds the parent directory to sys.path so Python can find your modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from MSH_Quasi_1D import GF_hybrid

pauli=np.array([[[1, 0], [0, 1]],
                [[0, 1], [1, 0]],
                [[0, -1j], [1j, 0]],
                [[1, 0], [0, -1]]],dtype=np.complex128)

txs0 = np.kron(pauli[1], pauli[0])
tzs0 = np.kron(pauli[3], pauli[0])
tzsx = np.kron(pauli[3], pauli[1])
tzsy = np.kron(pauli[3], pauli[2])

@numba.njit
def Bulk_Hamiltonian_bare(k, params):
    """(kx,ky)-space Hamiltonian with spinor (up_particle, down_particle, down_hole, up_hole)"""
    H = np.zeros((4, 4), dtype=np.complex128)
    kx, ky = k
    t, mu, alpha, Delta = params.t, params.mu, params.alpha, params.Delta
    
    xi_k = -2*t*np.cos(kx) - 4*t*np.cos(kx/2)*np.cos(np.sqrt(3)*ky/2) - mu
    onsite = xi_k * tzs0 + Delta * txs0 
    
    alpha_x = 2*alpha*(np.sin(kx) + np.cos(np.sqrt(3)*ky/2)*np.sin(kx/2))
    alpha_y = 2*alpha*np.sqrt(3)*np.sin(np.sqrt(3)*ky/2)*np.cos(kx/2)
    
    H += onsite + alpha_x * tzsy - alpha_y * tzsx
    return H

def numerical_GF(kx, Y, params, steps=1000):
    """Brute-force numerical integration of the bare Green's Function."""
    ky_limit = 2 * np.pi / np.sqrt(3)
    ky_vec = np.linspace(-ky_limit, ky_limit, steps)
    
    # Pre-allocate integrand array
    g_integrand = np.zeros((steps, 4, 4), dtype=np.complex128)
    
    for i, ky in enumerate(ky_vec):
        k = np.array([kx, ky], dtype=np.float64)
        H0 = Bulk_Hamiltonian_bare(k, params)
        G0 = -np.linalg.inv(H0) 
        phase = np.exp(1j * ky * Y)
        g_integrand[i] = G0 * phase
      
    G_integrated = np.zeros((4, 4), dtype=np.complex128)
    for row in range(4):
        for col in range(4):
            G_integrated[row, col] = simpson(g_integrand[:, row, col], ky_vec)
                
    normalization = np.sqrt(3) / (4 * np.pi)
    return G_integrated * normalization

Params = namedtuple('Params', ['t', 'mu', 'alpha', 'Delta'])

def test_hybrid_kernel_consistency():
    params = Params(t=1.0, mu=-3.5, alpha=0.21, Delta=0.36)
    kx_test = 1.0 # Standard test with real kx
    Y_test = 3 * np.sqrt(3) 
    
    print(f"Verifying GF_hybrid for kx={kx_test}, Y={Y_test}...")

    # This uses the Residue Theorem internally
    G_analytical = GF_hybrid(kx_test, Y_test, params)

    # This uses the Bulk_Hamiltonian_bare and inv() 
    G_numerical = numerical_GF(kx_test, Y_test, params, steps=10000)

    # ERROR ANALYSIS
    max_err = np.max(np.abs(G_analytical - G_numerical))
    print(f"Max Absolute Error: {max_err:.2e}")
    
    # Assert that the analytical engine is exact to within a high tolerance
    assert max_err < 1e-10, f"Kernel mismatch found! Error: {max_err}"
    print("✅ Verification Successful: Analytical Engine matches Numerical Integral.")

if __name__ == "__main__":
    test_hybrid_kernel_consistency()
