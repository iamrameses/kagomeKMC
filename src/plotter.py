# src/plotter.py

import cupy as cp
import matplotlib as mpl
import matplotlib.animation as animation
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np

from .analysis import get_disclinations, get_global_orientational_order
from .lattice import KagomeLattice
from .energy import TriangularLatticeEnergies


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
    try:
        cmap = ListedColormap(np.vstack([
            mpl.cm.get_cmap('PRGn_r', 10)(np.arange(10))[np.array([3,2,1,0])],
            [[0.,0.,1.,1.], [0.9,0.9,0.9,0.9], [1.,0.,0.,1.]],
            mpl.cm.get_cmap('PuOr_r', 10)(np.arange(10))[np.array([9,8,7,6])]
        ]))
    except:
        cmap = ListedColormap(np.vstack([
            mpl.pyplot.get_cmap('PRGn_r', 10)(np.arange(10))[np.array([3,2,1,0])],
            [[0.,0.,1.,1.], [0.9,0.9,0.9,0.9], [1.,0.,0.,1.]],
            mpl.pyplot.get_cmap('PuOr_r', 10)(np.arange(10))[np.array([9,8,7,6])]
        ]))
    cnorm = Normalize(vmin=1-0.5, vmax=11+0.5)
    patches, nsides, psi6_avg = get_disclinations(L, mxy)
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

def animate_simulation(data, save_filepath=None, frameskip=2, interval=33, repeat=True, figsize=(28, 10)):
    """Animate basic information from the simulation data.""" 
    # Define the lattice and lattice energies objects
    lattice = KagomeLattice(**data['lattice_params'])
    if data['energy_params']['lattice'] == 'triangular':
        tle = TriangularLatticeEnergies(**data['energy_params']['lattice_params'])
    else:
        tle = None
    # Get the molecule positions and global orientational order (if not already calculated)
    mxy = lattice.get_latticesites().get()
    mxy = mxy[data['ids']]
    if 'globalboops' not in data:
        data['globalboops'] = get_global_orientational_order(lattice, mxy)
    # Define the colormap and normalization for the number of neighbors
    try:
        cmap = ListedColormap(np.vstack([
            mpl.cm.get_cmap('PRGn_r', 10)(np.arange(10))[np.array([3,2,1,0])],
            [[0.,0.,1.,1.], [0.9,0.9,0.9,0.9], [1.,0.,0.,1.]],
            mpl.cm.get_cmap('PuOr_r', 10)(np.arange(10))[np.array([9,8,7,6])]
        ]))
    except:
        cmap = ListedColormap(np.vstack([
            mpl.pyplot.get_cmap('PRGn_r', 10)(np.arange(10))[np.array([3,2,1,0])],
            [[0.,0.,1.,1.], [0.9,0.9,0.9,0.9], [1.,0.,0.,1.]],
            mpl.pyplot.get_cmap('PuOr_r', 10)(np.arange(10))[np.array([9,8,7,6])]
        ]))
    cnorm = Normalize(vmin=1-0.5, vmax=11+0.5)
    # Create the initial patch collection for the disclinations
    patches, nsides, psi6_avg = get_disclinations(lattice, mxy[0,:,:])
    collection = PatchCollection(patches, edgecolors='k', lw=0.3, cmap=cmap, norm=cnorm, alpha=0.6)
    collection.set_array(nsides)
    # Create the figure and axes
    fig = plt.figure(figsize=figsize, layout='constrained', dpi=72)
    gs = GridSpec(4, 5, figure=fig)
    axs = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[2, 0]),
        fig.add_subplot(gs[3, 0]),
        fig.add_subplot(gs[:, 1:3]),
        fig.add_subplot(gs[:, 3:5]),
    ]
    fig.canvas.resizable = False
    # Plot the initial data
    line1, = axs[0].plot([], [])
    line2, = axs[1].plot([], [])
    line3, = axs[2].plot([], [])
    line4, = axs[3].plot([], [])
    line5, = axs[4].plot([], [], 'o', color='tab:red', lw=1.0, ms=5, mec='k')
    col6 = axs[5].add_collection(collection)
    line6, = axs[5].plot(mxy[0,:,0], mxy[0,:,1], '.', color='k', ms=2.5)
    axs[0].set_title('Time intervals (s)')
    axs[0].set_yscale('log')
    axs[1].set_title('Temperature (K)')
    axs[2].set_title('Total energy (Hartrees)')
    axs[2].set_yscale('log')
    axs[3].set_title('Global orientational order')
    axs[3].set_xlabel('Time (s)')
    axs[3].set_xscale('log')
    axs[4].set_title('Molecule positions')
    axs[4].set_xlabel('x (nm)', fontsize=14)
    axs[4].set_ylabel('y (nm)', fontsize=14)
    if tle is not None:
        axs[4] = plot_latticeenergies(axs[4], tle, lattice.boxsize, nsamples=1000, cmap='viridis', alpha=0.5, zorder=1)
    axs[4].set(xlim=lattice.xlim, ylim=lattice.ylim)
    axs[5].set_title('Voronoi tessellation and disclinations')
    axs[5].set_xlabel('x (nm)', fontsize=14)
    axs[5].set_ylabel('y (nm)', fontsize=14)
    axs[5].set(xlim=lattice.xlim, ylim=lattice.ylim)
    # Define the frame generator and update plot functions
    def frame_generator(data, lattice, mxy, frameskip):
        nframes = data['times'].shape[0]
        frame = 0
        while frame < nframes:
            times = data['times'][1:frame+1]
            intervals = data['deltatimes'][1:frame+1]
            temps = data['temperatures'][1:frame+1]
            energies = data['totalenergies'][1:frame+1]
            rxy = mxy[frame+1,:,:]
            patches, nsides, _ = get_disclinations(lattice, rxy)
            gboops = data['globalboops'][1:frame+1] 
            frame += frameskip
            yield times, intervals, temps, energies, gboops, rxy, patches, nsides
    # Define the update plot function
    def update_plot(data):
        times, intervals, temps, energies, gboops, rxy, patches, nsides = data
        line1.set_data(times, intervals)
        line2.set_data(times, temps)
        line3.set_data(times, energies)
        line4.set_data(times, gboops)
        line5.set_data(rxy[:,0], rxy[:,1])
        collection.set_paths(patches)
        collection.set_array(nsides)
        line6.set_data(rxy[:,0], rxy[:,1])
        for i in range(4):
            axs[i].relim()
            axs[i].autoscale_view()
        return line1, line2, line3, line4, line5, collection, line6,
    # Create the animation object and display the plot
    ani = animation.FuncAnimation(fig, update_plot, frames=frame_generator(data, lattice, mxy, frameskip), blit=True, interval=interval, repeat=repeat, save_count=mxy.shape[0])
    if save_filepath is not None:
        writergif = animation.PillowWriter(fps=5)
        writergif.setup(fig, save_filepath, dpi=72)
        ani.save(save_filepath, writer=writergif, dpi='figure') 
    plt.show()
    return ani