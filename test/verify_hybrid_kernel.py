import numpy as np
from scipy.integrate import simpson
from MSH_Quasi_1D import GF_hybrid

@numba.njit
def Bulk_Hamiltonian_bare(k,params):
    """(kx,ky)-space Hamiltonian with spinor (up_particle, down_particle, down_hole, up_hole)"""
    
    H = np.zeros((4,4),dtype=np.complex128)
    
    kx, ky = k
    t,mu,alpha,Delta = params.t, params.mu, params.alpha, params.Delta
    
    xi_k = -2*t*np.cos(kx)-4*t*np.cos(kx/2)*np.cos(np.sqrt(3)*ky/2)-mu
    onsite=xi_k*tzs0 + Delta*txs0 
    
    alpha_x = 2*alpha*(np.sin(kx)+np.cos(np.sqrt(3)*ky/2)*np.sin(kx/2))
    alpha_y = 2*alpha*np.sqrt(3)*np.sin(np.sqrt(3)*ky/2)*np.cos(kx/2)
    
    H += onsite + alpha_x*tzsy - alpha_y*tzsx
    return H
      
def numerical_GF(kx, Y, params, steps=1000):
    """Brute-force numerical integration of the bare Green's Function."""
    ky_vec = np.linspace(-2*np.pi/np.sqrt(3), 2*np.pi/np.sqrt(3), steps)
    g_vals = []
    
    for ky in ky_vec:
        k = np.array([kx, ky], dtype=np.float64)
        H0 = Bulk_Hamiltonian_bare(k, params)
        G0 = -np.linalg.inv(H0) # invertible for superconductor with hard s-wave gap
        phase = np.exp(1j * ky * Y)
        g_integrand[i] = G0 * phase
      
    G_integrated = np.zeros((4, 4), dtype=np.complex128)
      for row in range(4):
          for col in range(4):
              G_integrated[row, col] = simpson(g_integrand[:, row, col], ky_vec)
            
    normalization = np.sqrt(3) / (4 * np.pi)
    return G_integrated * normalization

class MockParams:
    def __init__(self):
        self.t = 1.0
        self.mu = -3.5
        self.alpha = 0.21
        self.Delta = 0.36
      
def test_hybrid_kernel_consistency():
    params = MockParams()
    kx_test = 1+2j # kx is real, but equation is accurate for complex kx
    Y_test = 3*np.sqrt(3)  # Test at a non-zero displacement
    
    print(f"Verifying GF_hybrid for kx={kx}, Y={Y}...")

    # This uses the Residue Theorem internally
    G_analytical = GF_hybrid(kx, Y, params)

    # This uses the Bulk_Hamiltonian_bare and inv() 
    G_numerical = numerical_GF(kx, Y, params, steps=10000)

    # ERROR ANALYSIS
    max_err = np.max(np.abs(G_analytical - G_numerical))
    
    print(f"Max Absolute Error: {max_err:.2e}")
    
    # Assert that the analytical engine is exact to within a high tolerance
    assert max_err < 1e-10, f"Kernel mismatch found! Error: {max_err}"
    print("✅ Verification Successful: Analytical Engine matches Numerical Integral.")

if __name__ == "__main__":
    test_hybrid_kernel_consistency()
