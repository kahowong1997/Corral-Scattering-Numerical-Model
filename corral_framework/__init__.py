"""
Corral Framework
----------------
A numerical simulation framework for computing the renormalized Green's function 
and effective topological Hamiltonians for Magnetic-Superconductor Hybrid system 
confined in a quantum corral, with either quasi-1D or arbitrary 2D geometry.
"""

from .MSH_Quasi_1D import GF_hybrid, Topo_Ham_1D
from .MSH_2D_Library import (
    build_GF_library, 
    apply_GF_library_vectorized,
    build_GF_matrix_final, 
    build_corral_basis, 
    build_T_matrix, 
    Topo_Ham_2D
)

__version__ = "0.1.0"

__all__ = [
    "GF_hybrid",
    "Topo_Ham_1D",
    "build_GF_library",
    "apply_GF_library_vectorized",
    "build_GF_matrix_final",
    "build_corral_basis",
    "build_T_matrix",
    "Topo_Ham_2D"
]
