"""
MSH 2D Real-Space Module: Library Generation & Spatial Propagators
Author: Ka Ho Wong
Date: Jan 2026

This module implements the 2D real-space workflow for MSH systems. It utilizes 
the analytical kernel to pre-compute a high-fidelity library of Green's function 
components, enabling the construction of O(N) spatial propagators G(X, Y).

The module is optimized for arbitrary corral geometries where longitudinal 
translational symmetry is broken, allowing for the calculation of the 
renormalized Local Density of States (LDOS) across finite 2D lattices.
"""

import numba
import numpy as np
import scipy.linalg

from .MSH_Analytical_Kernel import (
    coefficients, denominator, get_adjugate, solve_poles
)

txs0 = np.array([[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]], dtype=np.complex128)
tzs0 = np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,-1]], dtype=np.complex128)


@numba.njit(cache=True)
def build_GF_library(params, steps=256):
    """
    Pre-computes the analytical poles and residues over a 1D longitudinal momentum grid.
    
    This library serves as the backbone for all real-space constructions. By pre-solving 
    the characteristic equations and residue components across the first Brillouin Zone, 
    the framework avoids redundant complex-root finding during spatial integration.
    
    Parameters:
        params (NamedTuple): System parameters (t, mu, alpha, Delta).
        steps (int): Number of kx points for the integration grid (default=256).
        
    Returns:
        tuple: (k_vec, roots_q, res_q, roots_qd, res_qd, dk) containing pre-calculated
               analytical components for both chiral sectors.
    """
  
    t, mu, alpha, Delta = params.t, params.mu, params.alpha, params.Delta
    
    k_start = 0.0
    k_end = 2 * np.pi
    dk = (k_end - k_start) / (steps - 1)
    k_vec = np.linspace(k_start, k_end, steps)

    roots_q  = np.zeros((steps, 2), dtype=np.complex128)
    roots_qd = np.zeros((steps, 2), dtype=np.complex128)
    res_q  = np.zeros((steps, 2, 2, 2), dtype=np.complex128)
    res_qd = np.zeros((steps, 2, 2, 2), dtype=np.complex128)

    for i in range(steps):
        kx = k_vec[i]
        
        # --- Block 1: Chiral q ---
        c4, c3, c2, c1, c0 = coefficients(kx, t, mu, alpha, Delta)
        z_in = solve_poles(c4, c3, c2, c1, c0)
        
        denom_0 = denominator(z_in[0], c4, c3, c2, c1)
        denom_1 = denominator(z_in[1], c4, c3, c2, c1)
        
        Res1 = (1.0 / denom_0) * get_adjugate(z_in[0], kx, t, mu, alpha, Delta, False)
        Res2 = (1.0 / denom_1) * get_adjugate(z_in[1], kx, t, mu, alpha, Delta, False)
        
        roots_q[i, 0] = z_in[0]
        roots_q[i, 1] = z_in[1]
        
        res_q[i, 0] = Res1
        res_q[i, 1] = Res2
        
        # --- Block 2: Chiral dagger (Delta -> -Delta) ---
        c4_d, c3_d, c2_d, c1_d, c0_d = coefficients(kx, t, mu, alpha, -Delta)
        z_in_d = solve_poles(c4_d, c3_d, c2_d, c1_d, c0_d)
        
        denom_d0 = denominator(z_in_d[0], c4_d, c3_d, c2_d, c1_d)
        denom_d1 = denominator(z_in_d[1], c4_d, c3_d, c2_d, c1_d)
        
        Res1_d = (1.0 / denom_d0) * get_adjugate(z_in_d[0], kx, t, mu, alpha, Delta, True)
        Res2_d = (1.0 / denom_d1) * get_adjugate(z_in_d[1], kx, t, mu, alpha, Delta, True)
        
        roots_qd[i, 0] = z_in_d[0]
        roots_qd[i, 1] = z_in_d[1]
        
        res_qd[i, 0] = Res1_d
        res_qd[i, 1] = Res2_d

    return k_vec, roots_q, res_q, roots_qd, res_qd, dk


@numba.njit(cache=True)
def apply_GF_library_vectorized(X_arr, Y_arr, library_data, halfplane=True):
    """
    Performs vectorized Fourier-transform integration to generate real-space G(X, Y).
    
    This function utilizes Numba-optimized scalar accumulators to evaluate the 
    Residue Theorem sum. It employs the Euler form for phase-decay coupling 
    to minimize trigonometric overhead and memory allocations within the hot loop.
    
    Parameters:
        X_arr, Y_arr (ndarray): Arrays of spatial displacements.
        library_data (tuple): Data returned from build_GF_library.
        halfplane (bool): If True, invokes G(-X, -Y) = [G(X, Y)]^dagger symmetry.
        
    Returns:
        ndarray: A stack of (n_vecs, 4, 4) real-space Green's Function matrices.
    """
  
    k_vec, roots_q, res_q, roots_qd, res_qd, dk = library_data
    steps = len(k_vec)
    n_vecs = len(X_arr)
    
    norm_factor = dk / (2 * np.pi)
    
    r1 = np.zeros(steps); theta1 = np.zeros(steps)
    r2 = np.zeros(steps); theta2 = np.zeros(steps)
    r1d = np.zeros(steps); theta1d = np.zeros(steps)
    r2d = np.zeros(steps); theta2d = np.zeros(steps)
    
    for i in range(steps):
        r1[i] = np.abs(roots_q[i, 0]); theta1[i] = np.angle(roots_q[i, 0])
        r2[i] = np.abs(roots_q[i, 1]); theta2[i] = np.angle(roots_q[i, 1])
        r1d[i] = np.abs(roots_qd[i, 0]); theta1d[i] = np.angle(roots_qd[i, 0])
        r2d[i] = np.abs(roots_qd[i, 1]); theta2d[i] = np.angle(roots_qd[i, 1])

    G_results = np.zeros((n_vecs, 4, 4), dtype=np.complex128)
    
    # U matrix construction
    f = 1.0 / np.sqrt(2.0)
    U = f * (np.eye(4, dtype=np.complex128) + 1j * txs0)
    U_adj = U.conj().T

    for n in range(n_vecs):
        orig_X = X_arr[n]
        orig_Y = Y_arr[n]
        
        if orig_Y < 0:
            X = -orig_X  
            Y = -orig_Y  
            needs_conjugation = True
        else:
            X = orig_X
            Y = orig_Y
            needs_conjugation = False
        
        ny = np.round(Y * (2.0 / np.sqrt(3.0)))
        
        # Scalar accumulators to avoid array overhead
        s_q00=0j; s_q01=0j; s_q10=0j; s_q11=0j
        s_qd00=0j; s_qd01=0j; s_qd10=0j; s_qd11=0j
        
        for i in range(steps):
            kx = k_vec[i]
            weight = 0.5 if (i == 0 or i == steps - 1) else 1.0
            
            # Combine Phase and Decay into one Euler expression
            # -1.0 * weight * r^ny * e^(i * (kx*X + ny*th))
            w_neg = -1.0 * weight
            
            arg1 = kx * X + ny * theta1[i]
            power_z1 = w_neg * (r1[i]**ny) * (np.cos(arg1) + 1j * np.sin(arg1))
            
            arg2 = kx * X + ny * theta2[i]
            power_z2 = w_neg * (r2[i]**ny) * (np.cos(arg2) + 1j * np.sin(arg2))
            
            arg1d = kx * X + ny * theta1d[i]
            power_z1d = w_neg * (r1d[i]**ny) * (np.cos(arg1d) + 1j * np.sin(arg1d))
            
            arg2d = kx * X + ny * theta2d[i]
            power_z2d = w_neg * (r2d[i]**ny) * (np.cos(arg2d) + 1j * np.sin(arg2d))
            
            Res1, Res2 = res_q[i, 0], res_q[i, 1]
            Res1d, Res2d = res_qd[i, 0], res_qd[i, 1]
            
            s_q00 += Res1[0,0]*power_z1 + Res2[0,0]*power_z2
            s_q01 += Res1[0,1]*power_z1 + Res2[0,1]*power_z2
            s_q10 += Res1[1,0]*power_z1 + Res2[1,0]*power_z2
            s_q11 += Res1[1,1]*power_z1 + Res2[1,1]*power_z2

            s_qd00 += Res1d[0,0]*power_z1d + Res2d[0,0]*power_z2d
            s_qd01 += Res1d[0,1]*power_z1d + Res2d[0,1]*power_z2d
            s_qd10 += Res1d[1,0]*power_z1d + Res2d[1,0]*power_z2d
            s_qd11 += Res1d[1,1]*power_z1d + Res2d[1,1]*power_z2d

        # Assemble temporary 4x4 block
        temp_block = np.zeros((4, 4), dtype=np.complex128)
        temp_block[0, 2] = s_qd00; temp_block[0, 3] = s_qd01
        temp_block[1, 2] = s_qd10; temp_block[1, 3] = s_qd11
        temp_block[2, 0] = s_q00;  temp_block[2, 1] = s_q01
        temp_block[3, 0] = s_q10;  temp_block[3, 1] = s_q11
        
        rotated = U @ temp_block @ U_adj
        rotated *= norm_factor
        
        if needs_conjugation:
            G_results[n] = rotated.conj().T
        else:
            G_results[n] = rotated
            

    return G_results


@numba.njit(cache=True)
def build_GF_matrix_final(pos1, pos2, library_data):
    """
    Assembles a global Green's Function matrix for arbitrary lattice configurations.
    
    Implements a 'Symmetry Folding' algorithm that sorts and filters unique spatial 
    displacements between site pairs. This reduces the number of heavy integration 
    calls from O(N^2) to O(Unique_Displacements), drastically accelerating the 
    construction of large-scale Green's function matrices.
    
    Parameters:
        pos1 (ndarray): Source site coordinates (N, 2).
        pos2 (ndarray): Sink site coordinates (M, 2).
        library_data (tuple): Pre-computed analytical kernel data.
        
    Returns:
        ndarray: Reshaped (N*4, M*4) global Green's Function matrix.
    """
  
    N = pos1.shape[0]
    M = pos2.shape[0]
    n_pairs = N * M
    
    diffs_c = np.empty(n_pairs, dtype=np.complex128)
    orig_rows = np.empty(n_pairs, dtype=np.int32)
    orig_cols = np.empty(n_pairs, dtype=np.int32)
    
    flipped = np.empty(n_pairs, dtype=np.bool_)

    k = 0
    for i in range(N):
        x1, y1 = pos1[i, 0], pos1[i, 1]
        for j in range(M):
            x2, y2 = pos2[j, 0], pos2[j, 1]
            
            dx = x1 - x2
            dy = y1 - y2
            
            if dy < -1e-9 or (abs(dy) < 1e-9 and dx < -1e-9):
                dx_key = -dx
                dy_key = -dy
                flipped[k] = True
            else:
                dx_key = dx
                dy_key = dy
                flipped[k] = False
            
            dx_rnd = round(dx_key, 8)
            dy_rnd = round(dy_key, 8)
            
            diffs_c[k] = dx_rnd + 1j * dy_rnd
            orig_rows[k] = i
            orig_cols[k] = j
            k += 1

    sort_idx = np.argsort(diffs_c)
    
    unique_count = 1
    for k in range(1, n_pairs):
        if diffs_c[sort_idx[k]] != diffs_c[sort_idx[k-1]]:
            unique_count += 1
            
    unique_X = np.empty(unique_count, dtype=np.float64)
    unique_Y = np.empty(unique_count, dtype=np.float64)
    
    curr_u = 0
    idx_0 = sort_idx[0]
    
    # Reconstruct EXACT folded values
    r0, c0 = orig_rows[idx_0], orig_cols[idx_0]
    dx0 = pos1[r0, 0] - pos2[c0, 0]
    dy0 = pos1[r0, 1] - pos2[c0, 1]
    if flipped[idx_0]: dx0, dy0 = -dx0, -dy0
    
    unique_X[0] = dx0
    unique_Y[0] = dy0
    
    for k in range(1, n_pairs):
        curr_idx = sort_idx[k]
        prev_idx = sort_idx[k-1]
        
        if diffs_c[curr_idx] != diffs_c[prev_idx]:
            curr_u += 1
            r, c = orig_rows[curr_idx], orig_cols[curr_idx]
            dx = pos1[r, 0] - pos2[c, 0]
            dy = pos1[r, 1] - pos2[c, 1]
            if flipped[curr_idx]: dx, dy = -dx, -dy
            unique_X[curr_u] = dx
            unique_Y[curr_u] = dy

    unique_GFs = apply_GF_library_vectorized(unique_X, unique_Y, library_data, halfplane=True)
    G_out = np.zeros((N, 4, M, 4), dtype=np.complex128)
    
    curr_ptr = 0
    curr_u = 0
    
    while curr_ptr < n_pairs:
        current_val = diffs_c[sort_idx[curr_ptr]]
        G_base = unique_GFs[curr_u]
        G_conj = G_base.conj().T
        
        while curr_ptr < n_pairs:
            idx = sort_idx[curr_ptr]
            if diffs_c[idx] != current_val:
                curr_u += 1
                break 
            
            r = orig_rows[idx]
            c = orig_cols[idx]
            
            if flipped[idx]:
                G_out[r, :, c, :] = G_conj
            else:
                G_out[r, :, c, :] = G_base
            
            curr_ptr += 1

    return G_out.reshape(N * 4, M * 4)

@numba.njit(cache=True)
def get_sector_blocks_fast(G0_corral):
    """
    Decomposes the Green's function matrix restricted to the corral subspace into decoupled chiral sector blocks.
    
    Utilizes the BDI symmetry of the Nambu-basis transformation to split the 
    (4*N_corral x 4*N_corral) matrix into two (2*N_corral x 2*N_corral) sectors. This 
    halves the complexity of the subsequent eigenvalue problem.
    """
  
    dim = G0_corral.shape[0]
    N_corral = dim // 4
    dim_half = 2 * N_corral
    
    factor = 1.0 / np.sqrt(2.0)
    U_loc = factor * (np.eye(4, dtype=np.complex128) + 1j * txs0)
    U_H = U_loc.conj().T
    h_v = -tzs0

    M_A = np.zeros((dim_half, dim_half), dtype=np.complex128)
    M_B = np.zeros((dim_half, dim_half), dtype=np.complex128)

    for i in range(N_corral):
        for j in range(N_corral):
     
            block = G0_corral[4*i:4*i+4, 4*j:4*j+4].copy()
            
            M_rot = U_H @ block @ h_v @ U_loc
            
            M_A[2*i:2*i+2, 2*j:2*j+2] = M_rot[0:2, 0:2]
            M_B[2*i:2*i+2, 2*j:2*j+2] = M_rot[2:4, 2:4]
            
    return M_A, M_B

@numba.njit(cache=True)
def reconstruct_basis_fast(L_A, L_B, R_A, R_B):
    """
    Rebuilds the full Nambu basis from individual chiral sector eigenvectors.
    
    Performs high-speed block-wise matrix reconstruction using pre-sliced 
    transformation matrices (U_loc) to ensure the final T-matrix 
    is back to the correct Nambu basis.
    """
  
    dim_half = L_A.shape[0]
    N_corral = dim_half // 2
    dim = 2 * dim_half
    
    factor = 1.0 / np.sqrt(2.0)
    U_loc = factor * (np.eye(4, dtype=np.complex128) + 1j * txs0)
    U_H = U_loc.conj().T 
    
    U_A_slice = np.ascontiguousarray(U_loc[:, 0:2])
    U_B_slice = np.ascontiguousarray(U_loc[:, 2:4])
    UH_A_slice = np.ascontiguousarray(U_H[0:2, :])
    UH_B_slice = np.ascontiguousarray(U_H[2:4, :])
    
    L_final = np.zeros((dim, dim), dtype=np.complex128)
    R_final = np.zeros((dim, dim), dtype=np.complex128)
    
    for i in range(N_corral):
        L_block_A = L_A[2*i:2*i+2, :].copy() 
        L_block_B = L_B[2*i:2*i+2, :].copy()
        L_final[4*i:4*i+4, :dim_half] = U_A_slice @ L_block_A
        L_final[4*i:4*i+4, dim_half:] = U_B_slice @ L_block_B
        
        R_block_A = R_A[:, 2*i:2*i+2].copy()
        R_block_B = R_B[:, 2*i:2*i+2].copy()
        R_final[:dim_half, 4*i:4*i+4] = R_block_A @ UH_A_slice
        R_final[dim_half:, 4*i:4*i+4] = R_block_B @ UH_B_slice
        
    return L_final, R_final

def build_corral_basis(G0_corral):
    """
    Computes the Eigen-basis decomposition of the corral Green's function.
    
    This is a hybrid function combining Numba-accelerated block handling with 
    Scipy's LAPACK-optimized 'eig' and 'inv' routines. By solving the system 
    in the eigen-basis, the T-matrix can be calculated analytically for any 
    scattering potential strength V.
    """
  
    M_A, M_B = get_sector_blocks_fast(G0_corral)
    
    w_A, L_A = scipy.linalg.eig(M_A)
    w_B, L_B = scipy.linalg.eig(M_B)
    
    R_A = np.linalg.inv(L_A)
    R_B = np.linalg.inv(L_B)
    
    w = np.concatenate((w_A, w_B))
    
    L_final, R_final = reconstruct_basis_fast(L_A, L_B, R_A, R_B)

    return w, L_final, R_final

@numba.njit(cache=True)
def build_T_matrix(V, basis_data):
    """
    Constructs the renormalized T-matrix via the analytical Basis-Weighting method.
    
    Instead of performing a full matrix inversion (I-GV)^-1, this function 
    scales the pre-computed eigenvectors by the scattering weight (V / (1 - V*w)). 
    This allows for instantaneous T-matrix generation during parameter sweeps 
    over varying potential strengths (V).
    """
  
    w, L, R = basis_data
    
    d = (V / (1.0 - V * w)).astype(np.complex128)
    L_weighted = L * d  # Vectorized broadcasting: scales each column j by d[j]
    
    dim = L.shape[0]
    N_corral = dim // 4
    TZS0_neg = -tzs0  # Global operator
    
    # 2. Vectorized Block-Left-Action: H @ (L_weighted)
    # Instead of nested loops, we reshape to 4-row blocks and use a batch dot product
    # L_mod: (N_corral, 4, dim)
    L_reshaped = L_weighted.reshape((N_corral, 4, dim))
    L_mod_reshaped = np.empty_like(L_reshaped)
    
    for k in range(N_corral):
        L_mod_reshaped[k] = TZS0_neg @ L_reshaped[k]
        
    # 3. Final projection back to Nambu space
    # Re-flatten the reshaped blocks back to (dim, dim)
    return L_mod_reshaped.reshape((dim, dim)) @ R

@numba.njit(cache=True)
def Topo_Ham_2D(J, G0_chain, G0_chain_corral, T_matrix):
    """
    Generates the final Effective Topological Hamiltonian for the MSH system.
    
    Renormalizes the substrate's bare propagator with the corral's T-matrix 
    via the Dyson equation. Finally, incorporates the ferromagnetic exchange 
    coupling (J) for the MSH chain.
    
    Returns:
        ndarray: The (4N_chain x 4N_chain) Effective Topological Hamiltonian matrix, 
        N_chain is the number of sites on the chain.
    """
  
    is_T_zero = True
    if T_matrix.shape[0] > 0:
        if np.any(T_matrix):
            is_T_zero = False

    if is_T_zero:
        G_chain = G0_chain
    else:
        G_chain = G0_chain + G0_chain_corral @ T_matrix @ G0_chain_corral.conj().T        
    
    H_top = -np.linalg.inv(G_chain)
    
    if J != 0:
        dim = H_top.shape[0]
        for i in range(dim):
            H_top[i, i] += ((-1)**i) * J
            
    return H_top
