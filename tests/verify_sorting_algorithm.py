import numpy as np
import time
from collections import namedtuple
from MSH_2D_Library import build_GF_library, build_GF_matrix_final, apply_GF_library_vectorized

Params = namedtuple('Params', ['t', 'mu', 'alpha', 'Delta'])

def run_fair_benchmark():
    params = Params(t=1.0, mu=-3.5, alpha=0.24, Delta=0.36)
    
    steps = 256 
    print(f"Pre-computing GF Library (steps={steps})...")
    library_data = build_GF_library(params, steps=steps)
    
    N_sites = 30 # 900 total site-pairs
    
    # --- Force repeated differences ---
    while True:
        pos = np.zeros((N_sites, 2))
        for i in range(N_sites):
            n1, n2 = np.random.randint(-3, 3), np.random.randint(-3, 3)
            pos[i] = [n1 + 0.5*n2, (np.sqrt(3)/2)*n2]
        
        diffs = []
        for i in range(N_sites):
            for j in range(N_sites):
                dx = round(pos[i,0] - pos[j,0], 8)
                dy = round(pos[i,1] - pos[j,1], 8)
                diffs.append((dx, dy))
        
        unique_vectors = len(set(diffs))
        if unique_vectors < (N_sites**2):
            print(f"Lattice Ready: {unique_vectors} unique vectors for {N_sites**2} pairs.")
            break

    # --- WARMUP (JIT Compilation) ---
    # We run both once so Numba compilation time isn't included in the benchmark
    _ = build_GF_matrix_final(pos, pos, library_data)
    _ = apply_GF_library_vectorized(np.array([1.0]), np.array([0.0]), library_data)

    # --- METHOD A: Optimized Folding ---
    # This uses the argsort/unique/symmetry logic
    start_opt = time.perf_counter()
    G_optimized = build_GF_matrix_final(pos, pos, library_data)
    t_opt = time.perf_counter() - start_opt

    # --- METHOD B: Vectorized Brute-Force ---
    # 1. Pre-calculate the displacement arrays (Standard Python/NumPy)
    dx_all = np.zeros(N_sites**2)
    dy_all = np.zeros(N_sites**2)
    k = 0
    for i in range(N_sites):
        for j in range(N_sites):
            dx_all[k] = pos[i,0] - pos[j,0]
            dy_all[k] = pos[i,1] - pos[j,1]
            k += 1
            
    # 2. Timing the vectorized execution + assembly
    start_brute = time.perf_counter()
    # Call the core engine ONCE for all 900 displacements
    G_blocks_all = apply_GF_library_vectorized(dx_all, dy_all, library_data)
    
    # 3. Assemble into the large matrix
    G_brute = np.zeros((N_sites * 4, N_sites * 4), dtype=np.complex128)
    for n in range(N_sites**2):
        r, c = n // N_sites, n % N_sites
        G_brute[4*r:4*r+4, 4*c:4*c+4] = G_blocks_all[n]
    t_brute = time.perf_counter() - start_brute

    # --- FINAL ANALYSIS ---
    max_diff = np.max(np.abs(G_optimized - G_brute))
    speedup = t_brute / t_opt if t_opt > 0 else 0
    
    print("\n" + "="*45)
    print(f"FAIR BENCHMARK RESULTS (N={N_sites}, Steps={steps})")
    print("="*45)
    print(f"Unique Vectors:          {unique_vectors}")
    print(f"Redundant Vectors saved: {N_sites**2 - unique_vectors}")
    print(f"Vectorized Brute-Force:  {t_brute:.6f} s")
    print(f"Optimized Folding:       {t_opt:.6f} s")
    print(f"Speedup Factor:     {speedup:.2f}x")
    print(f"Numerical Error:         {max_diff:.2e}")
    print("="*45)
    
    if max_diff < 1e-14:
        print("✅ VERIFICATION SUCCESSFUL")
    else:
        print("❌ VERIFICATION FAILED: Results deviate")

if __name__ == "__main__":
    run_fair_benchmark()
