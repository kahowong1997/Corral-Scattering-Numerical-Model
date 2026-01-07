"""
MSH Analytical Kernel: Residue Theorem & Pole-Solving Engine
Author: Ka Ho Wong
Date: Jan 2026

This module provides the core high-performance analytical framework for Magnet-Superconductor 
Hybrid (MSH) systems.
"""

import numba
import numpy as np

@numba.njit(cache=True)
def coefficients(kx, t, mu, alpha, Delta):
    """
    Calculates the analytic coefficients for the MSH characteristic polynomial P(z).
    """
    A = -2.0 * t * np.cos(kx / 2.0)
    B = -2.0 * t * np.cos(kx) - mu - 1j * Delta
    
    # Pre-calculating trig terms for clarity and speed
    sin_kx2 = np.sin(kx / 2.0)
    cos_kx2 = np.cos(kx / 2.0)
    sin_kx = np.sin(kx)
    
    c4 = -(A**2 - alpha**2 * sin_kx2**2 + 3.0 * alpha**2 * cos_kx2**2)
    c3 = -(2.0 * A * B - 4.0 * alpha**2 * sin_kx2 * sin_kx)
    c2 = -((2.0 * A**2 + B**2) - (2.0 * alpha**2 * sin_kx2**2 + 4.0 * alpha**2 * sin_kx**2) - 6.0 * alpha**2 * cos_kx2**2)
    
    # Symmetry of the polynomial for the MSH triangular lattice
    return c4, c3, c2, c3, c4

@numba.njit(cache=True)
def denominator(z, c4, c3, c2, c1):
    """
    Calculates the derivative of the characteristic polynomial P'(z).
    """
    return 4.0 * c4 * z**3 + 3.0 * c3 * z**2 + 2.0 * c2 * z + c1

@numba.njit(cache=True)
def get_adjugate(z, kx, t, mu, alpha, Delta, is_dagger):
    """
    Calculates the adjugate matrix of the Chiral Hamiltonian at a specific pole z.
    """
    D_sign = 1.0 if not is_dagger else -1.0
    
    cos_kx = np.cos(kx)
    cos_kx2 = np.cos(kx / 2.0)
    sin_kx = np.sin(kx)
    sin_kx2 = np.sin(kx / 2.0)
    z2_plus_1 = z**2 + 1.0
    
    M11 = (-2.0 * t * cos_kx - mu - 1j * D_sign * Delta) * z - 2.0 * t * cos_kx2 * (z2_plus_1)
    Mx = 2.0 * alpha * (sin_kx * z + sin_kx2 * (z2_plus_1) / 2.0)
    My = 2.0 * alpha * np.sqrt(3.0) * (cos_kx2 * (z**2 - 1.0) / (2j))
    
    res = np.empty((2, 2), dtype=np.complex128)
    if not is_dagger:
        res[0, 0] = 1j * M11
        res[0, 1] = -Mx + 1j * My
        res[1, 0] = Mx + 1j * My
        res[1, 1] = 1j * M11
    else:
        res[0, 0] = -1j * M11
        res[0, 1] = Mx - 1j * My
        res[1, 0] = -Mx - 1j * My
        res[1, 1] = -1j * M11
    return res

@numba.njit(cache=True)
def solve_poles(c4, c3, c2, c1, c0):
    """
    Solves for the zeros of the characteristic polynomial P(z) using a companion matrix.
    """
    # 4-space indentation strictly maintained here
    coeffs = np.array([c3/c4, c2/c4, c1/c4, c0/c4], dtype=np.complex128)
    companion = np.zeros((4, 4), dtype=np.complex128)

    for i in range(3):
        companion[i+1, i] = 1.0
    for i in range(4):
        companion[0, i] = -coeffs[i]
    
    all_poles = np.linalg.eigvals(companion)
    
    in_poles = np.zeros(2, dtype=np.complex128)
    count = 0
    for p in all_poles:
        if np.abs(p) < 1.0 - 1e-10:
            if count < 2:
                in_poles[count] = p
                count += 1
    return in_poles
