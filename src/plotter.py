# src/plotter.py

import cupy as cp
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np

def _graphene_lines(row, col, sqrt3, lc, lbp, colors, lw, ls, alpha, zorder):
    """Return the bond lines of a graphene lattice unit cell as a LineCollection object."""
    ## Define primitive unit vectors that can define every site on the lattice
    B1 = np.array([0.25, sqrt3/12.]) * lc
    B2 = np.array([0.25, -sqrt3/12.]) * lc
    B3 = np.array([0., sqrt3/6.]) * lc
    ## Define the base unit cell of 6 lattice points
    base_cell = np.array([
        [0., 0.], 
        [0.5, 0.], 
        [0.75, 0.25*sqrt3],
        [0., 0.5*sqrt3], 
        [0.5, 0.5*sqrt3], 
        [0.25, 0.75*sqrt3]
    ]) * lc
    ## Shift cluster cell location according to the given row/column
    x_shift = row * lc
    y_shift = (col * sqrt3) * lc
    cell = base_cell + np.array([x_shift, y_shift])
    ## Generate respective graphene bond lines for plots
    bonds = np.array([
        [cell[0] - 0.5*B2, cell[0] + B2],
        [cell[0] + B2, cell[0] + B2 - 0.25*B3],
        [cell[1] - B1, cell[1] + B1], 
        [cell[1] + B1, cell[1] + B1 + 0.5*B2],
        [cell[2] - B3, cell[2] + B3],
        [cell[3] - 0.5*B1, cell[3] + B1],
        [cell[4] - B2, cell[4] + B2],
        [cell[4] + B2, cell[4] + B2 + 0.5*B1],
        [cell[5] - B3, cell[5] + 0.75*B3]
    ])
    return LineCollection(bonds + lbp, colors=colors, lw=lw, ls=ls, alpha=alpha, zorder=zorder)

def plot_graphenebonds(ax, L, colors='b', lw=0.5, ls='--', alpha=0.25, zorder=2):
    """Plot the graphene lattice lines on a given axis."""
    lbp = L._lbp.get()
    for row in range(L._n_rows):
        for col in range(L._n_columns):
            ax.add_collection(_graphene_lines(col, row, L._sqrt3, L._lc, lbp, colors, lw, ls, alpha, zorder))
    return ax

def plot_kagomesites(ax, L, s=35, c='k', ec='k', alpha=0.3, zorder=3):
    """Plot the Kagome lattice sites on a given axis."""
    lxy = L.get_latticesites.get()
    ax.scatter(lxy[:,0], lxy[:,1], s=s, c=c, ec=ec, alpha=alpha, zorder=zorder)
    return ax

def plot_latticeenergies(ax, LE, boxsize, angle=0., shift=[0., 0.], nsamples=1000, cmap='viridis', alpha=0.2, zorder=1):
    xtri = cp.linspace(0.0, boxsize[0], nsamples)
    ytri = cp.linspace(0.0, boxsize[1], nsamples)
    Xtri, Ytri = cp.meshgrid(xtri, ytri)
    Utri = LE.U(Xtri, Ytri, angle, shift)
    ax.contourf(Xtri.get(), Ytri.get(), Utri.get(), cmap='viridis', alpha=0.2, zorder=zorder)
    return ax