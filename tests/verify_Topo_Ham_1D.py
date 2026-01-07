import numpy as np
from corral_framework import Topo_Ham_1D
from collections import namedtuple

pauli=np.array([[[1, 0], [0, 1]],
                [[0, 1], [1, 0]],
                [[0, -1j], [1j, 0]],
                [[1, 0], [0, -1]]],dtype=np.complex128)

t0sz = np.kron(pauli[0], pauli[3])
txs0 = np.kron(pauli[1], pauli[0])
tzs0 = np.kron(pauli[3], pauli[0])
tzsy = np.kron(pauli[3], pauli[2])

Params = namedtuple('Params', ['t', 'mu', 'alpha', 'Delta', 'W_c', 'V', 'J'])

def pure_1D_hamiltonian(kx, t, mu, alpha, Delta, J):
    """Analytical 1D Hamiltonian for a single row of adatoms."""
  
    H = np.zeros((4,4),dtype=np.complex128)
    
    xi_k = -2*t*np.cos(kx)-mu 
    onsite = xi_k*tzs0 + Delta*txs0 + J*t0sz
    alpha_x = 2*alpha*np.sin(kx)
    
    H += onsite + alpha_x*tzsy 
    return H


def test_1D_matrix():
    params = Params(t=1.0, mu=-3.5, alpha=0.2, Delta=0.3, W_c=np.sqrt(3), V=1e6, J=0.5)
    
    kx_test = 1.347 
    V_values = [1e6, 1e9, 1e12]
  
    for V in V_values:
        # 1. Setup Quasi-1D params in the 'pinched' limit
        params = Params(t=1.0, mu=-3.5, alpha=0.2, Delta=0.3, W_c=np.sqrt(3), V=V, J=0.5)
        
        # 2. Compute Numerical Matrix
        H_num = Topo_Ham_1D(kx_test, params)
        
        # 3. Compute your Analytical Matrix
        # Note: We use the same 'bare' params to check 1/V convergence
        H_analytical = pure_1D_hamiltonian(kx_test, params.t, params.mu, params.alpha, params.Delta, params.J)
        
        # 4. Analyze Error
        max_diff = np.max(np.abs(H_num - H_analytical))
        print(f"{V:<12.0e} | {max_diff:<15.2e} | {1/V:.0e}")

        # Assertion: Error should be roughly proportional to 1/V
        # We allow a small factor (e.g., 10) for pre-factors in the T-matrix expansion
        assert max_diff < (10 / V), f"Error at V={V} is too high: {max_diff}"

    print("\n✅ Verification Successful: Error scales linearly with 1/V.")
  
if __name__ == "__main__":
    test_1D_matrix()
