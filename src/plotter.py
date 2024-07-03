# src/plotter.py

import cupy as cp
import freud
import matplotlib as mpl
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.patches import Polygon
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

def figsize_xscale(L):
    return (L._n_columns*L._lc) / (L._n_rows*L._lc*np.sqrt(3))

def figsize_yscale(L):
    return (L._n_rows*L._lc*np.sqrt(3)) / (L._n_columns*L._lc)

def plot_graphenebonds(ax, L, colors='b', lw=0.5, ls='--', alpha=0.25, zorder=2):
    """Plot the graphene lattice lines on a given axis."""
    lbp = L._lbp.get()
    for row in range(L._n_rows):
        for col in range(L._n_columns):
            ax.add_collection(_graphene_lines(col, row, L._sqrt3, L._lc, lbp, colors, lw, ls, alpha, zorder))
    return ax

def plot_disclinations(fig, ax, L, mxy):
    """Plot the disclinations on a given axis."""
    # Define the colormap and normalization for the number of neighbors
    cmap = ListedColormap(np.vstack([
        mpl.cm.get_cmap('PRGn_r', 10)(np.arange(10))[np.array([3,2,1,0])],
        [[0.,0.,1.,1.], [0.9,0.9,0.9,0.9], [1.,0.,0.,1.]],
        mpl.cm.get_cmap('PuOr_r', 10)(np.arange(10))[np.array([9,8,7,6])]
    ]))
    cnorm = Normalize(vmin=1-0.5, vmax=11+0.5)
    # Compute the Voronoi tessellation and the hexatic order parameter
    xdim, ydim = L.boxsize
    box = freud.box.Box(Lx=xdim, Ly=ydim, Lz=0, is2D=True)
    points = np.hstack((mxy-L.center_xy, np.zeros((mxy.shape[0], 1))))
    vor = freud.locality.Voronoi()
    psi6 = freud.order.Hexatic(k=6, weighted=False)
    vor.compute(system=(box, points))
    psi6.compute(system=(box, points), neighbors=vor.nlist)
    psi6_k = psi6.particle_order
    psi6_avg = np.mean(np.abs(psi6_k))
    # psi6_phase = np.abs(np.angle(psi6.particle_order))
    nsides = np.array([polytope.shape[0] for polytope in vor.polytopes])
    print(f"Global bond orientational order: {psi6_avg}")
    # Plot the Voronoi tessellation and the hexatic order parameter
    patches = []
    for polytope in vor.polytopes:
        poly = Polygon(polytope[:,:2]+L.center_xy, closed=True, facecolor='r')
        patches.append(poly)
    collection = PatchCollection(patches, edgecolors='k', lw=0.3, cmap=cmap, norm=cnorm, alpha=0.6)
    collection.set_array(nsides)
    dax = ax.add_collection(collection)
    # Plot the molecules
    ax.scatter(mxy[:,0], mxy[:,1], s=1.5, c='k', zorder=2)
    # Plot the colorbar
    cbar = fig.colorbar(dax, ax=ax, ticks=np.arange(1, 12), shrink=0.75)
    cbar.set_label(label='Number of Disclinations', labelpad=10., rotation=270, fontsize=12)
    return fig, ax

def plot_kagomesites(ax, L, s=35, c='k', ec='k', alpha=0.3, zorder=3):
    """Plot the Kagome lattice sites on a given axis."""
    lxy = L.get_latticesites().get()
    ax.scatter(lxy[:,0], lxy[:,1], s=s, c=c, ec=ec, alpha=alpha, zorder=zorder)
    return ax

def plot_latticeenergies(ax, LE, boxsize, nsamples=1000, cmap='viridis', alpha=0.2, zorder=1):
    """Plot the lattice energies on a given axis."""
    xtri = cp.linspace(0.0, boxsize[0], nsamples)
    ytri = cp.linspace(0.0, boxsize[1], nsamples)
    Xtri, Ytri = cp.meshgrid(xtri, ytri)
    Utri = LE.U(Xtri, Ytri)
    ax.contourf(Xtri.get(), Ytri.get(), Utri.get(), cmap='viridis', alpha=0.2, zorder=zorder)
    return ax

def plot_molecules(ax, mnnxy, show_nn=False, s=100, mc='tab:red', nnc='yellow', ec='k', lw=2.5, alpha=1.0, zorder=4):
    """Plot the molecules on a given axis."""
    # Highlight the site corresponding to the selected ID in red
    ax.scatter(mnnxy[:,0,0], mnnxy[:,0,1], s=s, c=mc, ec=ec, lw=lw, alpha=alpha, zorder=zorder)
    # Highlight the nearest-neighbors of selected ID in yellow
    if show_nn: ax.scatter(mnnxy[:,1:,0], mnnxy[:,1:,1], s=s, c=nnc, ec=ec, lw=lw, alpha=alpha, zorder=zorder)
    return ax