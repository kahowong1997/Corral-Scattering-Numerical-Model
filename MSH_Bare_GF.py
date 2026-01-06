"""
MSH Green's Function Engine: Analytical Pole-Solving
Author: Ka Ho Wong
Date: Jan 2026

This module provides a high-performance framework for computing the 
Green's Function G(kx, Y) of Magnet-Superconductor Hybrid (MSH) 
systems using an analytical residue theorem approach. 
"""

import numba
import numpy as np

@numba.njit(cache=True)
def coefficients(kx, t, mu, alpha, Delta):
	"""
	Calculates the analytic coefficients for the MSH Chiral Hamiltonian characteristic polynomial P(z).
	
	Parameters:
			kx (float): Momentum in the x-direction.
			t, mu, alpha, Delta (float): Tight-binding parameters in Hamiltonian.
			
	Returns:
			tuple: (c4, c3, c2, c1, c0) coefficients used for pole-solving.
	"""
	
	A = -2*t*np.cos(kx/2)
	B = -2*t*np.cos(kx) - mu - 1j*Delta
	
	c4 = -(A**2 - alpha**2*np.sin(kx/2)**2 + 3*alpha**2*np.cos(kx/2)**2)
	
	c3 = -(2*A*B - 4*alpha**2*np.sin(kx/2)*np.sin(kx))
	
	c2 = -((2*A**2 + B**2) - (2*alpha**2*np.sin(kx/2)**2 + 4*alpha**2*np.sin(kx)**2) - 6*alpha**2*np.cos(kx/2)**2)
	
	return c4,c3,c2,c3,c4

@numba.njit(cache=True)
def denominator(z, c4, c3, c2, c1):
	"""
	Calculates the derivative of the characteristic polynomial P'(z).
	
	In the analytical derivation for zero energy Green's function, 
	this value serves as the denominator of the residue at pole z, 
	following the Residue Theorem for simple poles: Res(-H^-1, z) = Adj(H(z)) / P'(z).
	
	Parameters:
	z (complex): The specific pole (root) of the characteristic polynomial.
	c4, c3, c2, c1 (complex): Coefficients of the polynomial P(z), 
				where P(z) = c4*z^4 + c3*z^3 + c2*z^2 + c1*z + c0.
	
	Returns:
	complex: The value of the derivative 4*c4*z^3 + 3*c3*z^2 + 2*c2*z + c1.
	"""
	
	return 4*c4*z**3 + 3*c3*z**2 + 2*c2*z + c1

@numba.njit(cache=True)
def get_adjugate(z, kx, t, mu, alpha, Delta, is_dagger):
	"""
	Calculates the adjugate matrix of the Chiral Hamiltonian at a specific pole z.
	
	This function is used to analytically solve for the residues of the Green's 
    Function. The 'is_dagger' flag determines whether to compute the block 
    corresponding to Delta or -Delta, which represents the two off-diagonal 
    sectors in the chiral (BDI) basis.
	
	Parameters:
	z (complex): The specific pole (root) of the characteristic polynomial.
	is_dagger (bool): If True, computes the adjugate for the q-dagger sector (Delta -> -Delta) in the chiral basis.
	
	Returns:
	np.ndarray: 2x2 matrix representing the numerator of the residue in the chiral sector.
	"""
	
	D_sign = 1.0 if not is_dagger else -1.0
	
	# Pre-calculate common terms
	cos_kx = np.cos(kx)
	cos_kx2 = np.cos(kx/2)
	sin_kx = np.sin(kx)
	sin_kx2 = np.sin(kx/2)
	z2_plus_1 = z**2 + 1
	
	M11 = (-2*t*cos_kx - mu - 1j * D_sign * Delta)*z - 2*t*cos_kx2*(z2_plus_1)
	Mx = 2*alpha*(sin_kx*z + sin_kx2*(z2_plus_1)/2)
	My = 2*alpha*np.sqrt(3)*(cos_kx2*(z**2 - 1)/(2j))
	
	res = np.empty((2, 2), dtype=np.complex128)
	if not is_dagger:
		res[0, 0] = 1j*M11;   res[0, 1] = -Mx + 1j*My
		res[1, 0] = Mx + 1j*My; res[1, 1] = 1j*M11
	else:
		res[0, 0] = -1j*M11;  res[0, 1] = Mx - 1j*My
		res[1, 0] = -Mx - 1j*My; res[1, 1] = -1j*M11
	return res

@numba.njit(cache=True)
def solve_poles(c4, c3, c2, c1, c0):
	"""
	Solves for the zeros of the characteristic polynomial P(z) using a companion matrix.
	
	This method provides higher numerical stability than standard root-finding 
	algorithms for quartic polynomials. It specifically filters for poles 
	inside the unit circle (|z| < 1), which represent the physical 
	evanescent modes in the real-space Green's function.

	Parameters:
			c4, c3, c2, c1, c0 (complex): Polynomial coefficients.

	Returns:
			np.ndarray: Array of complex poles inside the unit circle.
	"""
	
    coeffs = np.array([c3/c4, c2/c4, c1/c4, c0/c4], dtype=np.complex128)
    companion = np.zeros((4, 4), dtype=np.complex128)

    # Fill companion matrix
    for i in range(3):
        companion[i+1, i] = 1.0
    for i in range(4):
        companion[0, i] = -coeffs[i]
    
    all_poles = np.linalg.eigvals(companion)
    
    # Filter poles inside unit circle
    # Expect exactly 2 in our physics model
    in_poles = np.zeros(2, dtype=np.complex128)
    count = 0
    for p in all_poles:
        if np.abs(p) < 1.0 - 1e-10: # Small epsilon for numerical safety
            if count < 2:
                in_poles[count] = p
                count += 1
    return in_poles

@numba.njit(cache=True)
def GF_general_numba(kx, Y, params):
	"""
    Constructs the hybrid Green's Function in (kx, Y) via the Cauchy Residue Theorem.
    
	Note: The analytical residue summation is derived strictly for the Y > 0 
	half-plane. To compute the Green's Function for Y < 0, we invoke the 
	Hermitian relation G(kx, -Y) = [G(kx*, Y)]^dagger, mapping the calculation 
	back to the valid analytical domain.

    Parameters:
        kx (complex): Momentum in the x-direction.
        Y (float): Real-space distance in the y-direction.
        params (NamedTuple): Physical parameters of the system.

    Returns:
        np.ndarray: The 4x4 Nambu-space Green's Function matrix.
    """
	
    t, mu, alpha, Delta = params.t, params.mu, params.alpha, params.Delta
    ny = abs(int(Y * 2 / np.sqrt(3)))
	
	# Mapping Y < 0 to the analytical domain Y > 0 via Hermitian symmetry
    if Y < 0:
        kx = kx.conjugate()  
        
    # --- Block 1: Chiral q ---
    c4, c3, c2, c1, c0 = coefficients(kx, t, mu, alpha, Delta) 
    z_in = solve_poles(c4, c3, c2, c1, c0)
    
    Res1 = (1/denominator(z_in[0], c4, c3, c2, c1)) * get_adjugate(z_in[0], kx, t, mu, alpha, Delta, False)
    Res2 = (1/denominator(z_in[1], c4, c3, c2, c1)) * get_adjugate(z_in[1], kx, t, mu, alpha, Delta, False)
    G_chiral_q = -Res1 * (z_in[0]**ny) - Res2 * (z_in[1]**ny)
    
    # --- Block 2: Chiral dagger (Delta -> -Delta) ---
    c4_d, c3_d, c2_d, c1_d, c0_d = coefficients(kx, t, mu, alpha, -Delta)
    z_in_d = solve_poles(c4_d, c3_d, c2_d, c1_d, c0_d)
    
    Res1_d = (1/denominator(z_in_d[0], c4_d, c3_d, c2_d, c1_d)) * get_adjugate(z_in_d[0], kx, t, mu, alpha, Delta, True)
    Res2_d = (1/denominator(z_in_d[1], c4_d, c3_d, c2_d, c1_d)) * get_adjugate(z_in_d[1], kx, t, mu, alpha, Delta, True)
    G_chiral_q_dagger = -Res1_d * (z_in_d[0]**ny) - Res2_d * (z_in_d[1]**ny)
    
    # --- Final Construction ---
    # Manual block construction
    G_block = np.zeros((4, 4), dtype=np.complex128)
    G_block[0:2, 2:4] = G_chiral_q_dagger
    G_block[2:4, 0:2] = G_chiral_q
    
    U = (1.0/np.sqrt(2.0)) * (np.eye(4, dtype=np.complex128) + 1j * txs0)
    U_adj = np.ascontiguousarray(U.conj().T)
    G_mid = np.ascontiguousarray(U @ G_block)
    G = G_mid @ U_adj 

	# Apply the adjoint operator as required by the Hermitian relation for Y < 0
    if Y < 0:
        G = G.conj().T
    
    return G
