import numpy as np
from MSH_Quasi_1D import Topo_Ham_1D
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
    kx = 1.347 
    
    numerical_gap = []
    theoretical_gap = []


    H_quasi1D = Topo_Ham_1D(kx, params)
    H_pure1D = pure_1D_hamiltonian(kx, params.t, params.mu, params.alpha, params.Delta, params.J)
  
    # Check difference
    diff = np.abs(H_quasi1D - H_pure1D)
    max_err = np.max(diff) 
    
    # Check if the error is consistent with the 1/V scaling
    print(f"Observed Error: {max_err:.2e} | Expected scaling (1/V): {1/params.V:.2e}")
    
    assert max_err < (10/params.V), f"Error exceeds 1/V scaling! Diff: {max_err}"
    print("✅ Theoretical Limit: PASSED (Error consistent with 1/V)")

if __name__ == "__main__":
    test_1D_matrix()
