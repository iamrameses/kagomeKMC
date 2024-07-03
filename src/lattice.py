# src/lattice.py

import cupy as cp
from math import sqrt
from numba import cuda


class KagomeLattice(object):
    """Class to define the Kagome lattice for the simulation.

    Parameters
    ----------
    boxwidth : float
        Width of the simulation box in nm.
    lattice_constant : float, optional
        Lattice constant of the Kagome lattice in nm.
    energy_barrier : float, optional
        Diffusion energy barrier of lattice in Kelvin.
    debye_freq : float, optional
        Debye frequency of the lattice in Hz.
    transition_type : str, optional
        Type of nearest-neighbor transitions to consider.
    intdtype : str, optional
        Data type for integer values.
    floatdtype : str, optional
        Data type for floating-point values.

    Attributes
    ----------
    boxsize : tuple
        Dimensions of the simulation box in nm.
    center_xy : tuple
        XY-coordinates of the center of the simulation box in nm.
    debye_frequency : float
        Debye frequency of the lattice in Hz.
    energy_barrier : float
        Energy barrier for the simulation in Kelvin.
    lattice_constant : float
        Lattice constant of the Kagome lattice in nm.
    n_unitcells : tuple
        Number of unit cells (columns, rows) that make up the lattice.
    xlim : tuple
        x-axis limits of the simulation box in nm.
    ylim : tuple
        y-axis limits of the simulation box in nm.

    Methods
    -------
    generate_randomids : CuPy array
        Generates random indices for placing molecules on the lattice.
    get_latticesites : CuPy array
        Returns the XY-coordinates of the lattice sites.
    get_sitennids : CuPy array
        Returns the nearest-neighbor indices of the lattice sites.
    
    """
    def __init__(self, boxwidth=102.336, lattice_constant=0.246, energy_barrier=275.0, debye_freq=10e10, transition_type='t', intdtype='int32', floatdtype='float64'):
        self._bw = float(boxwidth)
        self._lc = float(lattice_constant)
        self._eb = float(energy_barrier)
        self._freq = float(debye_freq)
        self._idtype = intdtype
        self._fdtype = floatdtype
        self._sqrt3 = sqrt(3)
        # Number of columns of unit cells
        self._n_columns = int(self._bw // self._lc)
        # Number of rows of unit cells
        self._n_rows = int(cp.round(self._n_columns / self._sqrt3))
        # Total number of sites on the lattice
        self._n_totalsites = int(6 * self._n_columns * self._n_rows) 
        # Lattice padding required for periodic boundary conditions
        self._lbp = cp.array([1., self._sqrt3], dtype=self._fdtype) * 0.125 * self._lc
        # Lattice dimensions (not including padding for periodic boundary conditions)
        self._dims = cp.array([self._lc * self._n_columns, self._lc * self._sqrt3 * self._n_rows], self._fdtype)
        self._center_xy = self._dims / 2.
        # XY-coordinates of the 6 lattice sites that define the base unit cell
        self._unitcell = cp.array([
            [0.5, 0.],
            [0., 0.],
            [0.75, 0.25*self._sqrt3], 
            [0., sqrt(0.75)],
            [0.5, sqrt(0.75)],
            [0.25, 0.75*self._sqrt3]
        ], dtype=self._fdtype) * self._lc + self._lbp
        # Define the mapping for each of the 6 unit cell sites to its 4 nearest-neighbor transitions
        if transition_type == 't':  # long translation-only
            self.__nns = cp.array([
                [[1,0,0], [1,0,3], [-1,0,0], [0,-1,3]],
                [[1,0,1], [-1,0,4], [-1,0,1], [0,-1,4]],
                [[1,0,5], [0,0,5], [0,-1,5], [1,-1,5]],
                [[1,0,3], [0,1,0], [-1,0,3], [-1,0,0]],
                [[1,0,4], [0,1,1], [-1,0,4], [1,0,1]],
                [[0,1,2], [-1,1,2], [-1,0,2], [0,0,2]]
            ], dtype=self._idtype)
        elif transition_type == 'tr':  # short translation + 60-degree rotation
            self.__nns = cp.array([
                [[1,0,1], [0,0,2], [0,0,1], [0,-1,5]],
                [[0,0,0], [-1,0,2], [-1,0,0], [0,-1,5]],
                [[1,0,3], [0,0,4], [0,0,0], [1,0,1]],
                [[0,0,4], [0,0,5], [-1,0,4], [-1,0,2]],
                [[1,0,3], [0,0,5], [0,0,3], [0,0,2]],
                [[0,1,0], [0,1,1], [0,0,3], [0,0,4]]
            ], dtype=self._idtype)
        elif transition_type == 'tr2':  # long translation + two 60-degree rotations
            self.__nns = cp.array([
                [[0,0,4], [-1,0,2], [0,-1,4], [1,-1,5]],
                [[0,0,2], [0,0,3], [-1,-1,5], [0,-1,3]],
                [[1,0,4], [0,0,3], [0,0,1], [1,0,0]],
                [[0,1,1], [-1,0,5], [0,0,1], [0,0,2]],
                [[1,0,5], [0,1,0], [-1,0,2], [0,0,0]],
                [[1,1,1], [-1,1,0], [-1,0,4], [1,0,3]]
            ], dtype=self._idtype)
        else:
            raise ValueError('Invalid transition type. Choose from: t, tr, tr2')

    @property
    def boxsize(self): 
        return self._dims.get()
    
    @property
    def center_xy(self):
        return self._center_xy.get()
    
    @property
    def debye_frequency(self):
        return self._freq

    @property
    def energy_barrier(self): 
        return self._eb

    @property
    def lattice_constant(self): 
        return self._lc

    @property
    def n_unitcells(self): 
        return (self._n_columns, self._n_rows)

    @property
    def xlim(self): 
        return (0., self._dims.get()[0])

    @property
    def ylim(self): 
        return (0., self._dims.get()[1])
    
    @staticmethod
    @cuda.jit
    def __site_nnids_gpu(output, site_numbers, nns_arr, n_columns, n_rows): 
        i = cuda.grid(1)
        if i < site_numbers.shape[0]:
            n_unitcells = int(site_numbers[i] // 6)
            col_num = int(n_unitcells // n_rows) 
            row_num = int(n_unitcells % n_rows)  
            unit_cell_num = int(site_numbers[i] % 6)
            nns = nns_arr[unit_cell_num]
            output[i,0] = site_numbers[i]
            cuda.syncthreads()
            for j in range(nns.shape[0]):
                a = (col_num + nns[j,0]) % n_columns
                b = (row_num + nns[j,1]) % n_rows
                c = nns[j,2]
                output[i,j+1] = int(((a * n_rows) + b) * 6 + c)

    @staticmethod
    @cuda.jit
    def __sites_1D(output, site_numbers, unit_cell, n_rows, lc, sqrt3):
        i = cuda.grid(1)
        if i < site_numbers.shape[0]:
            n_unitcells = int(site_numbers[i] // 6)  
            col_num = int(n_unitcells // n_rows)  
            row_num = int(n_unitcells % n_rows)  
            unit_cell_num = int(site_numbers[i] % 6)
            output[i,0] = unit_cell[unit_cell_num][0] + col_num * lc
            output[i,1] = unit_cell[unit_cell_num][1] + row_num * sqrt3 * lc

    @staticmethod
    @cuda.jit
    def __sites_2D(output, site_numbers, unit_cell, n_rows, lc, sqrt3):
        i, j = cuda.grid(2)
        if (i < site_numbers.shape[0]) and (j < site_numbers.shape[1]):
            n_unitcells = int(site_numbers[i,j] // 6)  
            col_num = int(n_unitcells // n_rows)  
            row_num = int(n_unitcells % n_rows)  
            unit_cell_num = int(site_numbers[i,j] % 6)
            output[i,j,0] = unit_cell[unit_cell_num][0] + col_num * lc
            output[i,j,1] = unit_cell[unit_cell_num][1] + row_num * sqrt3 * lc
    
    def generate_randomids(self, n_molecules, sites, alt_site_ids=None, n_attempts=100, threshold=5, seed=None):
        """Generates random indices for placing molecules on the lattice.

        Parameters
        ----------
        n_molecules : int
            Number of molecules to place on the lattice.
        sites : CuPy array
            XY-coordinates of the lattice sites.
        site_ids : CuPy array, optional
            Array of possible indices where a molecule can be placed.
            If None, all sites are considered.
        n_attempts : int, optional
            Number of attempts to generate random indices.
        threshold : float, optional
            Minimum distance between molecules in units of lattice constant.
        seed : int, optional
            Seed for random number generator.

        Returns
        -------
        CuPy array
            Random indices of the lattice sites where molecules are placed.
        """  
        # Initialize random number generator with or without a defined seed
        cp.random.seed(seed=seed)
        lc_threshold = threshold * self._lc
        attempts_left = n_attempts
        while attempts_left > 0:
            try:
                # Array of possible indices where a molecule can be placed
                if alt_site_ids is None: 
                    site_ids = cp.arange(self._n_totalsites, dtype=self._idtype)
                else:
                    site_ids = alt_site_ids
                # Randomize the order that these indices appear in the array
                cp.random.shuffle(site_ids)
                # Pre-allocate the array to hold the coordinate indices of each molecule
                molecule_ids = cp.empty(n_molecules, dtype=self._idtype)
                for i in range(n_molecules):
                    # Assign molecule to the first coordinate index in site_ids
                    molecule_ids[i] = site_ids[0]
                    # Calculate the distances between this molecule and every other site in site_ids
                    dr = cp.empty((site_ids.shape[0], 2), dtype=self._fdtype)
                    dr = cp.abs(cp.subtract(sites[site_ids], sites[molecule_ids[i]], out=dr), out=dr)
                    dr = cp.minimum(dr, self._dims - dr, out=dr)
                    dists = cp.linalg.norm(dr, axis=1)
                    # Generate mask array of sites in site_ids whose distance from this molecule is below threshold
                    outside_threshold = cp.where(dists >= lc_threshold)[0]
                    # Remove those sites (that are too close to the assigned molecule) from site_ids
                    site_ids = site_ids[outside_threshold]
                print(f"Generated {n_molecules} molecule indices in {n_attempts - attempts_left + 1} attempts.")
                break
            except:
                attempts_left -= 1
                pass
        if attempts_left == 0:
            raise ValueError(f'Failed to generate random indices after ({n_attempts}) maximum attempts.')
        return molecule_ids 
    
    def get_latticesites(self, site_numbers=None, tpb=32): 
        """Returns the XY-coordinates of the lattice sites.

        Parameters
        ----------
        site_numbers : CuPy array, optional
            Array of site indices for which to return the XY-coordinates.
            If None, returns all sites.
        tpb : int, optional
            Number of threads per block.

        Returns
        -------
        CuPy array
            XY-coordinates of the sites in the lattice. 
            If site_numbers are not specified, returns all sites.
        """
        if site_numbers is None: 
            site_numbers = cp.arange(self._n_totalsites, dtype=self._idtype)
        try:
            n_dimensions = len(site_numbers.shape)
        except:
            raise ValueError('"site_numbers" must be an CuPy array.')
        bpg = lambda x, threads: (x + threads - 1) // threads
        tpb2 = site_numbers.shape[1] if n_dimensions > 1 else 1
        tpb3 = site_numbers.shape[2] if n_dimensions > 2 else 1
        block = (tpb, tpb2, tpb3)
        grid = (bpg(site_numbers.shape[0], block[0]), bpg(tpb2, block[1]), tpb3)
        sites = cp.empty((*site_numbers.shape, 2), dtype=self._fdtype)
        if n_dimensions == 1: 
            self.__sites_1D[grid, block](sites, site_numbers, self._unitcell, self._n_rows, self._lc, self._sqrt3)
        elif n_dimensions == 2:
            self.__sites_2D[grid, block](sites, site_numbers, self._unitcell, self._n_rows, self._lc, self._sqrt3)
        else:
            raise ValueError('This function only supports 1D and 2D arrays.')
        return sites
    
    def get_sitennids(self, site_numbers, tpb=32):
        """Returns the nearest-neighbor indices of the lattice sites.

        Site nearest-neighbors are defined by the transition type given during initialization of lattice object.
            't'   - long translation-only 
            'tr'  - short translation + 60-degree rotation
            'tr2' - long translation + two 60-degree rotations

        Parameters
        ----------
        site_numbers : CuPy array
            Array of site indices for which to return the nearest-neighbor indices.
        tpb : int, optional
            Number of threads per block.

        Returns
        -------
        CuPy array
            Nearest-neighbor indices of the sites in the lattice.
        """
        bpg = lambda x, threads: (x + threads - 1) // threads
        block = (tpb, 1, 1)
        grid = (bpg(site_numbers.shape[0], block[0]), bpg(1, block[1]), 1)
        output = cp.zeros((site_numbers.shape[0], 5), dtype=self._idtype)
        self.__site_nnids_gpu[grid, block](output, site_numbers, self.__nns, self._n_columns, self._n_rows)
        return output