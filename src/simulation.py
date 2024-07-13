# src/simulation.py

import cupy as cp
import freud
import numpy as np
import time

from .analysis import get_global_orientational_order
from .energy import TriangularLatticeEnergies, interaction_energy_function, transition_rates
from .pairwise_distance import pairwise_distance3
from .temperature import temperature_function


def simulate(lattice, energy_params, temp_params, molecule_params, duration, frames_per_kelvin=30, nwarmupsteps=1e5, progress_freq=2.5e5, max_dt=10, seed=None): 
    """Run a kinetic Monte Carlo simulation on a given Kagome lattice.

    Parameters
    ----------
    lattice : KagomeLattice
        The Kagome lattice object to simulate on.
    energy_params : dict
        Dictionary containing the energy parameters for the simulation.
    temp_params : dict
        Dictionary containing the temperature parameters for the simulation.
    molecule_params : dict
        Dictionary containing the molecule parameters for the simulation.
    duration : float
        Target duration of the simulation in seconds.
    frames_per_kelvin : int, optional
        Number of frames to sample per integer temperature change in Kelvin, by default 60.
    nwarmupsteps : int, optional
        Number of warm-up steps to take before the simulation starts, by default 50000.
    max_dt : int, optional
        Maximum time step interval threshold in seconds, by default 10. 
        Simulations will stop if the time interval exceeds this threshold.
    seed : int, optional
        Seed for reproducibility, by default None.

    Returns
    -------
    results : dict
        Dictionary containing the simulation results.

    """
    print('Setting up the simulation...')
    # Set the seed for reproducibility
    cp.random.seed(seed) 

    # Set up the lattice energies
    if energy_params['lattice'] == 'triangular':
        lat_energies = TriangularLatticeEnergies(**energy_params['lattice_params'])
    else:
        lat_energies = None
    
    # Set up the interaction energies
    int_energies = interaction_energy_function(
        lattice, 
        method=energy_params['interaction'], 
        energy_params=energy_params['interaction_params']
    )

    # Set up the temperature function for the simulation
    temp, Ti, Tf = temperature_function(duration, **temp_params)
    
    # Calculate the number of frames to sample during the simulation
    nframes = int((Ti - Tf) * frames_per_kelvin)

    # Get the xy-coordinate lookup for all lattice sites
    lxy = lattice.get_latticesites()
    
    # Get the nearest neighbor indices for all lattice sites
    lnnids = lattice.get_sitennids(cp.arange(lxy.shape[0], dtype=lattice._idtype))

    # Generate random initial positions for the molecules
    nmolecules = molecule_params['n_molecules']
    threshold = molecule_params['threshold']
    mids = lattice.generate_randomids(n_molecules=nmolecules, sites=lxy, n_attempts=1000, threshold=threshold, seed=seed)  

    # Initialize freud library to compute global bond-orientational order parameter
    xdim, ydim = lattice.boxsize
    box = freud.box.Box(Lx=xdim, Ly=ydim, Lz=0, is2D=True)
    vor = freud.locality.Voronoi()
    psi6 = freud.order.Hexatic(k=6, weighted=False)

    # Initialize the arrays to store the simulation data
    times = cp.zeros((nframes+1), dtype=lattice._fdtype)
    ids = cp.zeros((nframes+1, nmolecules), dtype=lattice._idtype)
    deltatimes = cp.zeros((nframes+1), dtype=lattice._fdtype)
    temperatures = cp.zeros((nframes+1), dtype=lattice._fdtype)
    totalenergies = cp.zeros((nframes+1), dtype=lattice._fdtype)
    globalboops = np.zeros((nframes+1), dtype=lattice._fdtype)

    # Store the initial positions of the molecules
    ids[0,:] = mids 

    # Generate random numbers for warmup steps
    nwarmupsteps = int(nwarmupsteps)
    rands = cp.random.uniform(0., 0.9999999, size=(nwarmupsteps, 2))

    # Get the temperature for the warm-up steps
    temp_i = float(temp(0))

    # Timing the warm-up steps
    start_gpu = cp.cuda.Event()
    end_gpu = cp.cuda.Event()
    start_gpu.record()
    start_cpu = time.perf_counter()
    
    # Warm up steps
    print(f'Starting simulation warm-up steps at {temp_i:.2f} K...') 
    textpad = ' ' * 100
    update_interval = int(0.05*nwarmupsteps)
    for i in range(nwarmupsteps):
        # Get the xy-coordinates of the molecules and their nearest neighbors
        mnnxy = lxy[lnnids[ids[0]]]
        # Compute the pairwise distances between all nearest neighbors sites and molecules
        mnndists = pairwise_distance3(mnnxy, lattice._dims)
        # Compute the interaction energies for all pairwise distances in mnndists
        mnnenergies = int_energies(mnndists)
        # Ensure that the diagonal elements are zero (self-interaction energies)
        cp.einsum('iji->ij', mnnenergies)[...] = 0.0
        # Sum up the interaction energy contributions from each molecule
        mnnenergies = cp.sum(mnnenergies, axis=2)
        # Add the energy from the lattice if it exists
        if lat_energies is not None:
            mnnenergies = cp.add(mnnenergies, lat_energies.U(mnnxy[:,:,0], mnnxy[:,:,1]), out=mnnenergies)
        # Compute the transition rates of each molecule to its nearest neighbors
        rates = transition_rates(mnnenergies[:,0], mnnenergies[:,1:], temperature=temp_i)
        # Compute the cumulative sum of the rates
        Kcumsum = cp.cumsum(rates)
        # Randomly choose the event time interval and transitioned molecule
        chosenK_id = cp.searchsorted(Kcumsum, rands[i,0]*Kcumsum[-1], side="right")
        # Get indices for the randomly chosen molecule and nearest neighbor to transition to
        mol_id = chosenK_id // 4
        nn_id = chosenK_id % 4
        # Update the molecule's new position to the chosen nearest neighbor
        ids[0,mol_id] = lnnids[ids[0]][mol_id,nn_id+1]        
        if i % update_interval == 0:
            print(f'Warm-up step {i+1:g} / {nwarmupsteps:g} completed.'+textpad, end='\r', flush=True)
    print(f'Warm-up step {nwarmupsteps:g} / {nwarmupsteps:g} completed.'+textpad, end='\r', flush=True)
    
    # Timing the warm-up steps
    end_cpu = time.perf_counter()
    end_gpu.record()
    end_gpu.synchronize()
    print(f'\nWarm-up steps completed in: {(end_cpu - start_cpu)/60:.3f} min CPU | {(cp.cuda.get_elapsed_time(start_gpu, end_gpu)/1000)/60:.3f} min GPU.')

    # Actual KMC steps
    print(f'\nStarting KMC steps...')
    i = 0  # Elementary KMC step counter
    time_i = float(0.0)   # Initial time of the simulation
    framenum = 0  # Sampled frame counter 
    temp_target = float(temp_i - (1/frames_per_kelvin))
    progress_freq = int(progress_freq)
    while temp_i >= Tf:
        # Generate new random numbers every nwarmupsteps
        j = i % progress_freq
        if j == 0:
            rands = cp.random.uniform(0., 0.9999999, size=(progress_freq+1, 2))
            print(f'KMC step {i+1:g} | Temperature {temp_i:.4f} K | Simulation time = {time_i:g} s')
        # Get the temperature at the current time
        temp_i = float(temp(time_i))
        # Get the xy-coordinates of the molecules and their nearest neighbors
        mnnxy = lxy[lnnids[ids[framenum]]]
        # Compute the pairwise distances between all nearest neighbors sites and molecules
        mnndists = pairwise_distance3(mnnxy, lattice._dims)
        # Compute the interaction energies for all pairwise distances in mnndists
        mnnenergies = int_energies(mnndists)
        # Ensure that the diagonal elements are zero (self-interaction energies)
        cp.einsum('iji->ij', mnnenergies)[...] = 0.0
        # Sum up the interaction energy contributions from each molecule
        mnnenergies = cp.sum(mnnenergies, axis=2)
        # Add the energy from the lattice if it exists
        if lat_energies is not None:
            mnnenergies = cp.add(mnnenergies, lat_energies.U(mnnxy[:,:,0], mnnxy[:,:,1]), out=mnnenergies)
        # Compute the transition rates of each molecule to its nearest neighbors
        rates = transition_rates(mnnenergies[:,0], mnnenergies[:,1:], temperature=temp_i)
        # Compute the cumulative sum of the rates
        Kcumsum = cp.cumsum(rates)
        # Randomly choose the event time interval and transitioned molecule
        chosenK_id = cp.searchsorted(Kcumsum, rands[j,0]*Kcumsum[-1], side="right")
        # Get indices for the randomly chosen molecule and nearest neighbor to transition to
        mol_id = chosenK_id // 4
        nn_id = chosenK_id % 4
        # Compute the time interval for the event
        dt = -cp.log(rands[j,1]) / (Kcumsum[-1] * lattice.debye_frequency * cp.exp(-lattice.energy_barrier / temp_i))
        # Update the time of the simulation
        time_i = time_i + dt.get()
        if temp_i <= temp_target:
            # Copy the previous positions to the current step
            ids[framenum+1] = ids[framenum] 
            # Update the molecule's new position to the chosen nearest neighbor
            ids[framenum+1,mol_id] = lnnids[ids[framenum]][mol_id,nn_id+1]
            # Update the extra simulation data
            times[framenum+1] = time_i
            deltatimes[framenum+1] = dt
            temperatures[framenum+1] = temp_i
            totalenergies[framenum+1] = cp.sum(mnnenergies[:,0])
            globalboops[framenum+1] = get_global_orientational_order(lattice, lxy[ids[framenum+1]].get(), box=box, vor=vor, psi6=psi6)
            # Update the target temperature for the next frame
            temp_target = temp_i - (1/frames_per_kelvin)
            # Update the frame number
            framenum += 1
            print(f'KMC step {i+1:g} | Temperature {temp_i:.4f} K | Simulation time = {time_i:g} s')
        else:
            # Update the molecule's new position to the chosen nearest neighbor
            ids[framenum,mol_id] = lnnids[ids[framenum]][mol_id,nn_id+1]
        # Apply early stopping if the KMC step time intervals are too large
        if dt > max_dt:
            print(f'\nKMC step time intervals are too large: {dt:g} seconds. Ending simulation...')
            break
        elif (temp_i < Tf) | (temp_i > Ti):
            print(f'\nTemperature out of range: {temp_i:.4f} K. Ending simulation...')
            break
        # Increment the KMC step counter
        i += 1
    results = {
        'n_steps': i,
        'n_frames': framenum,
        'times': times[:framenum].get(),
        'ids': ids[:framenum].get(),
        'deltatimes': deltatimes[:framenum].get(),
        'temperatures': temperatures[:framenum].get(),
        'totalenergies': totalenergies[:framenum].get(),
        'globalboops': globalboops[:framenum],
        'energy_params': energy_params,
        'temp_params': temp_params,
        'molecule_params': molecule_params,
        'lattice_params': {
            'boxwidth': lattice._bw,
            'lattice_constant': lattice._lc,
            'energy_barrier': lattice._eb,
            'debye_freq': lattice._freq,
            'transition_type': lattice._ttype,
            'intdtype': lattice._idtype,
            'floatdtype': lattice._fdtype
        }
    }
    print('\nSimulation complete!')
    return results