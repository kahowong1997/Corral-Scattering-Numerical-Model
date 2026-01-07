import numpy as np
from corral_framework.MSH_2D_Library import build_GF_library, apply_GF_library_vectorized
from corral_framework.MSH_Quasi_1D import GF_hybrid
from collections import namedtuple

Params = namedtuple('Params', ['t', 'mu', 'alpha', 'Delta'])

def test_2D_vectorized_integration():
    params = Params(t=1.0, mu=-3.42, alpha=0.36, Delta=0.17)
    steps = 512
    library_data = build_GF_library(params, steps=steps)
    
    # 1. Define multiple displacements (Array Input)
    # Testing different sectors: positive Y, negative Y, and zero displacement
    X_arr = np.array([0.0, 3.5, -6.0])
    Y_arr = np.array([0.0, 7*np.sqrt(3)/2, -4*np.sqrt(3)/2])
    
    # 2. Run Vectorized Library Call
    print(f"Running vectorized 2D library for {len(X_arr)} points...")
    G_vectorized = apply_GF_library_vectorized(X_arr, Y_arr, library_data)
    
    # 3. Verify each point against the Analytical Integral
    k_vec = np.linspace(0, 2*np.pi, steps)
    dk = k_vec[1] - k_vec[0]
    
    for n in range(len(X_arr)):
        X, Y = X_arr[n], Y_arr[n]
        G_ref_sum = np.zeros((4, 4), dtype=np.complex128)
        
        for i in range(steps):
            kx = k_vec[i]
            weight = 0.5 if (i == 0 or i == steps - 1) else 1.0
            G_kx = GF_hybrid(kx, Y, params)
            G_ref_sum += weight * G_kx * np.exp(1j * kx * X)
            
        G_reference = G_ref_sum * (dk / (2 * np.pi))
        
        # Check point-wise difference
        diff = np.max(np.abs(G_vectorized[n] - G_reference))
        print(f"Point {n} (X={X:.2f}, Y={Y:.2f}) Error: {diff:.2e}")
        
        assert diff < 1e-10, f"Vectorization error at point {n}!"

    print("\n✅ 2D Vectorization Verified: Array inputs match independent integrals.")

if __name__ == "__main__":
    test_2D_vectorized_integration()
