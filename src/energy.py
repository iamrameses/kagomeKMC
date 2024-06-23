# src/energy.py

import cupy as cp
from math import sqrt, pi
import numpy as np
import scipy as sp

from .constants import A0, HT, HV


# Wrapper for Scipy's Struve function of order 0 
scipy_H0 = lambda x: sp.special.struve(0, x)

# Wrapper for Scipy's Neumann function of order 0 (a.k.a. Bessel function of the second kind of order 0)
scipy_Y0 = lambda x: sp.special.y0(x)

def kappa_h(eps_r=4.22, mu=0.100, v_f=1.49e6):
    """Computes the inverse screening radius in Hartree atomic units
    
    Parameters
    ----------
    eps_r : float
        The relative permittivity of the medium
    mu : float
        The reduced mass of the impurity and the medium (in eV)
    v_f : float
        The Fermi velocity of the medium (in m/s)

    Returns
    -------
    kappa_h : float
        The inverse screening radius in Hartree atomic units
        
    """
    return (4 / eps_r) * (mu / HT) * ((HV**2) / (v_f**2))

def total_impurity_energies(r, Zval=1.0, epsr=4.22):
    """Computes the pairwise interaction energies via the total impurity potential
    
    Parameters
    ----------
    r : array-like
        The pairwise distances between molecules in the system
    Zval : float
        The valency of the impurity
    epsr : float
        The relative permittivity of the medium

    Returns
    -------
    Vth : array-like
        Total impurity potential energies (in Hartree atomic units) between all pairs of molecules
        
    """
    # Check if the input is a scalar or a 1D, 2D, or 3D array
    r_ndims = None if isinstance(r, float) else len(r.shape)
    if r_ndims == 3:
        cp.einsum('iji->ij', r)[...] = 0.0
    # Convert the distances into Hartree atomic units
    r = r / A0
    # Compute the total impurity potential
    khr = kappa_h(eps_r=epsr) * r
    khr_cpu = khr.get()
    Va = cp.where(r!=0.0, 1./r, cp.zeros_like(r)) * (Zval / epsr)
    Vb = cp.where(khr!=0.0, cp.array(scipy_H0(khr_cpu) - scipy_Y0(khr_cpu)), cp.zeros_like(khr))
    Vc = Va * (1.0 - (0.5 * pi * khr * Vb))
    # Reduce the dimensions of the array depending on dimensions of the input
    Vth = Vc if (r_ndims == 1) or (r_ndims is None) else cp.sum(Vc, axis=r_ndims-1)
    return Vth


class TriangularLatticeEnergies: 
    """Class to define the energy of a particle on a triangular lattice."""
    def __init__(self, lattice_constant=4.0, amplitude=1.0):
        self.lattice_constant = lattice_constant
        self.amplitude = amplitude * 11.605  # In terms of meV converted to Kelvin
    
    def U(self, x, y, angle=0., shift=[0., 0.]):
        """Evaluates potential at x, y coord lists
        x : list of x pos for [nparticles, nsteps]
        y : list of y pos for [nparticles, nsteps]
        angle : angle in radians to rotate lattice
        shift in energy is for numerical stability, since constant shift in energy
        only multiplies all rates by a common scale factor
        """
        x2 = (x+shift[0])*cp.cos(angle) - (y+shift[1])*cp.sin(angle)
        y2 = (x+shift[0])*cp.sin(angle) + (y+shift[1])*cp.cos(angle)
        a, A = self.lattice_constant, self.amplitude
        theta_x, theta_y = 2 * pi * x2 / a, 2 * pi * y2 / (a * sqrt(3))
        sigma = 2 * cp.cos(theta_x) * cp.cos(theta_y) + cp.cos(2 * (theta_y))
        return (A * 2 / 9 * ((3 - sigma)))