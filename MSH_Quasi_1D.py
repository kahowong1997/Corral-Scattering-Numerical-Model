"""
MSH Quasi-1D Module: Hybrid Propagators & T-Matrix Renormalization
Author: Ka Ho Wong
Date: Jan 2026

This module implements the Quasi-1D (strip) workflow for MSH systems. It utilizes 
the analytical kernel to construct the hybrid Green's Function G(kx, Y) and 
applies a T-matrix scattering formalism to compute renormalized topological bands.
"""

import numba
import numpy as np
from MSH_Analytical_Kernel import (
    coefficients, denominator, get_adjugate, solve_poles, txs0, tzs0
)

@numba.njit(cache=True)
def GF_hybrid(kx, Y, params):
    """
    Constructs the hybrid Green's Function in (kx, Y) via the Cauchy Residue Theorem.
    
    Note: The analytical residue summation is derived strictly for the Y > 0 
    half-plane. To compute the Green's Function for Y < 0, we invoke the 
    Hermitian relation G(kx, -Y) = [G(kx*, Y)]^dagger.
    """
    t, mu, alpha, Delta = params.t, params.mu, params.alpha, params.Delta
    ny = abs(int(np.round(Y * 2 / np.sqrt(3))))
    
    # Mapping Y < 0 to the analytical domain Y > 0 via Hermitian symmetry
    if Y < 0:
        kx = kx.conjugate()  
        
    # --- Block 1: Chiral q ---
    c4, c3, c2, c1, c0 = coefficients(kx, t, mu, alpha, Delta) 
    z_in = solve_poles(c4, c3, c2, c1, c0)
    
    Res1 = (1.0/denominator(z_in[0], c4, c3, c2, c1)) * get_adjugate(z_in[0], kx, t, mu, alpha, Delta, False)
    Res2 = (1.0/denominator(z_in[1], c4, c3, c2, c1)) * get_adjugate(z_in[1], kx, t, mu, alpha, Delta, False)
    G_chiral_q = -Res1 * (z_in[0]**ny) - Res2 * (z_in[1]**ny)
    
    # --- Block 2: Chiral dagger (Delta -> -Delta) ---
    c4_d, c3_d, c2_d, c1_d, c0_d = coefficients(kx, t, mu, alpha, -Delta)
    z_in_d = solve_poles(c4_d, c3_d, c2_d, c1_d, c0_d)
    
    Res1_d = (1.0/denominator(z_in_d[0], c4_d, c3_d, c2_d, c1_d)) * get_adjugate(z_in_d[0], kx, t, mu, alpha, Delta, True)
    Res2_d = (1.0/denominator(z_in_d[1], c4_d, c3_d, c2_d, c1_d)) * get_adjugate(z_in_d[1], kx, t, mu, alpha, Delta, True)
    G_chiral_q_dagger = -Res1_d * (z_in_d[0]**ny) - Res2_d * (z_in_d[1]**ny)
    
    # --- Final Construction ---
    G_block = np.zeros((4, 4), dtype=np.complex128)
    G_block[0:2, 2:4] = G_chiral_q_dagger
    G_block[2:4, 0:2] = G_chiral_q
    
    U = (1.0/np.sqrt(2.0)) * (np.eye(4, dtype=np.complex128) + 1j * txs0)
    U_adj = np.ascontiguousarray(U.conj().T)
    G_mid = np.ascontiguousarray(U @ G_block)
    G = G_mid @ U_adj 

    if Y < 0:
        G = G.conj().T
    
    return G

@numba.njit(cache=True)
def Dressed_GF(kx, params):
    """
    Computes the renormalized Green's Function at the corral center (y=0).
    """
    W_c, V = params.W_c, params.V

    # 1. Construct 8x8 Corral Propagator
    G_local = GF_hybrid(kx, 0, params)
    G_scat_down = GF_hybrid(kx, W_c, params)
    G_scat_up = GF_hybrid(kx, -W_c, params)
    
    G_corral = np.zeros((8,8), dtype=np.complex128)
    G_corral[0:4,0:4] = G_local
    G_corral[0:4,4:] = G_scat_up
    G_corral[4:,0:4] = G_scat_down
    G_corral[4:,4:] = G_local

    # 2. Define Scattering Potential H_V
    H_V = np.zeros((8,8), dtype=np.complex128)
    H_V[:4,:4] = -V * tzs0
    H_V[4:,4:] = -V * tzs0

    # 3. T-Matrix Calculation with Stability Switch
    if V <= 10.0:
        inv_part = np.ascontiguousarray(np.linalg.inv(np.eye(8, dtype=np.complex128) - G_corral @ H_V))
        T = H_V @ inv_part
    else:
        H_V_inv = np.zeros((8,8), dtype=np.complex128)
        H_V_inv[:4,:4] = -(1.0/V) * tzs0
        H_V_inv[4:,4:] = -(1.0/V) * tzs0
        T = np.ascontiguousarray(np.linalg.inv(H_V_inv - G_corral))
  
    # 4. Dyson Equation for the Center Point
    G_down = np.ascontiguousarray(GF_hybrid(kx, W_c/2.0, params))
    G_up = np.ascontiguousarray(GF_hybrid(kx, -W_c/2.0, params))

    T00 = np.ascontiguousarray(T[:4, :4])
    T01 = np.ascontiguousarray(T[:4, 4:])
    T10 = np.ascontiguousarray(T[4:, :4])
    T11 = np.ascontiguousarray(T[4:, 4:])

    G_dressed = G_local + G_down @ T00 @ G_up + G_down @ T01 @ G_down + \
                G_up @ T10 @ G_up + G_up @ T11 @ G_down
    
    return G_dressed

@numba.njit(cache=True)
def Topo_Ham_1D(kx, params):
    """
    Constructs the 1D Effective Topological Hamiltonian.
    """
    G_dressed = Dressed_GF(kx, params)
    G_inv = np.ascontiguousarray(np.linalg.inv(G_dressed))
    
    t0sz = np.array([
        [1.0,  0.0,  0.0,  0.0],
        [0.0, -1.0,  0.0,  0.0],
        [0.0,  0.0,  1.0,  0.0],
        [0.0,  0.0,  0.0, -1.0]
    ], dtype=np.complex128)
    
    H_mag = -G_inv + params.J * t0sz
    return H_mag
