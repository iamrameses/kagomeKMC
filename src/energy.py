# src/energy.py

import cupy as cp
from math import sqrt, pi
import numpy as np
import scipy as sp

from .constants import A0, KB, HT, HV


# Wrapper for Scipy's Struve function of order 0 
_scipy_H0 = lambda x: sp.special.struve(0, x)

# Wrapper for Scipy's Neumann function of order 0 (a.k.a. Bessel function of the second kind of order 0)
_scipy_Y0 = lambda x: sp.special.y0(x)

def ideal_boxwidths(tl_lc=4, tl_range=[2, 78], kl_lc=0.246, kl_range=[32, 1254], show_topn=10):
    """Computes the ideal box widths for the triangular and Kagome lattices

    Parameters
    ----------
    tl_lc : float
        The lattice constant of the triangular lattice (in nm)
    tl_range : list
        The range of box widths for the triangular lattice (in number of lattice constants)
    kl_lc : float
        The lattice constant of the Kagome lattice (in nm)
    kl_range : list
        The range of box widths for the Kagome lattice (in number of lattice constants)
    show_topn : int
        The number of top box widths to display

    Returns
    -------
    None
    
    """
    tle_x = tl_lc * np.arange(tl_range[0], tl_range[1])
    tle_cols = np.array(tle_x//tl_lc, dtype=int)
    tle_rows = np.array(tle_cols/np.sqrt(3), dtype=int)
    tle_y = tl_lc * np.sqrt(3) * tle_rows
    print(f"tle_x (first and last value): {tle_x[0]}, {tle_x[-1]}")
    kl_x = kl_lc * np.arange(kl_range[0], kl_range[1])
    kl_cols = np.array(kl_x//kl_lc, dtype=int)
    kl_rows = np.array(np.round(kl_cols/np.sqrt(3)), dtype=int)
    kl_y = kl_lc * np.sqrt(3) * kl_rows
    print(f"\nkl_x (first and last value): {kl_x[0]}, {kl_x[-1]}\n")
    min_ids = np.zeros(len(kl_x), dtype=int)
    min_vals = np.zeros(len(kl_x))
    min_diffs = np.zeros(len(kl_x))
    print("Ideal box widths (sorted by L2-norm error):")
    for i, (j, k) in enumerate(zip(kl_x, kl_y)):
        xdiffs = np.abs(tle_x-j)**2
        ydiffs = np.abs(tle_y-k)**2
        diffs = np.sqrt(xdiffs + ydiffs)
        min_ids[i] = np.argmin(diffs)
        min_vals[i] = j
        min_diffs[i] = diffs[min_ids[i]]
    sort_ids = np.argsort(min_diffs)
    min_vals = min_vals[sort_ids]
    min_diffs = min_diffs[sort_ids]
    for i in range(show_topn):
        print(f"#{i+1}: {min_vals[i]:.3f}nm | L2-norm error: {min_diffs[i]:.4f}")

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
    Vb = cp.where(khr!=0.0, cp.array(_scipy_H0(khr_cpu) - _scipy_Y0(khr_cpu)), cp.zeros_like(khr))
    Vc = Va * (1.0 - (0.5 * pi * khr * Vb))
    # Reduce the dimensions of the array depending on dimensions of the input
    Vth = Vc if (r_ndims == 1) or (r_ndims is None) else cp.sum(Vc, axis=r_ndims-1)
    return Vth

def transition_rates(m_energies, nn_energies, temperature):
    """Computes the transition rates between the microstates of the system

    Parameters
    ----------
    m_energies : array-like
        The energies of the molecules in the system 
        (in Hartree atomic units)
    nn_energies : array-like
        The energies of the nearest neighbors of the molecules in the system 
        (in Hartree atomic units)
    temperature : float
        The temperature of the system 
        (in Kelvin)

    Returns
    -------
    rates : array-like
        The transition rates between the molecules and their nearest neighbors sites
        (in dimensionless units)
    """
    m_ndims = len(m_energies.shape)
    nn_ndims = len(nn_energies.shape)
    if m_ndims != (nn_ndims - 1):
        raise ValueError('m_energies and nn_energies must be 1D and 2D arrays respectively.')
    factor = HT / (2 * KB * temperature)
    energy_differences = m_energies[:,None] - nn_energies
    return cp.exp(factor * energy_differences)

class TriangularLatticeEnergies: 
    """Class to define the energy of a particle on a triangular lattice.
    
    Parameters
    ----------
    lattice_constant : float
        The lattice constant of the triangular lattice
    amplitude : float
        The amplitude of the potential energy

    Methods
    -------
    U : array-like
        Evaluates the potential energy at the given x, y coordinates
    """
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
        theta_x, theta_y = 2 * pi * x2 / a, 2 * pi * y2 / (a * np.sqrt(3))
        sigma = 2 * cp.cos(theta_x) * cp.cos(theta_y) + cp.cos(2 * (theta_y))
        return (A * 2 / 9 * ((3 - sigma)))