# src/simulation.py

import cupy as cp
import time

from .energy import TriangularLatticeEnergies, interaction_energy_function, transition_rates
from .pairwise_distance import pairwise_distance3
from .temperature import temperature_function


def simulate(lattice, energy_params, temp_params, nmolecules, duration, threshold=6, frames_per_kelvin=60, nwarmupsteps=50000, seed=None): 
    
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
    mids = lattice.generate_randomids(n_molecules=nmolecules, sites=lxy, n_attempts=1000, threshold=threshold, seed=seed)  

    # Initialize the arrays to store the simulation data
    times = cp.zeros((nframes+1), dtype=lattice._fdtype)
    ids = cp.zeros((nframes+1, nmolecules), dtype=lattice._idtype)
    deltatimes = cp.zeros((nframes+1), dtype=lattice._fdtype)
    temperatures = cp.zeros((nframes+1), dtype=lattice._fdtype)
    totalenergies = cp.zeros((nframes+1), dtype=lattice._fdtype)

    # Store the initial positions of the molecules
    ids[0,:] = mids 

    # Generate random numbers for warmup steps
    rands = cp.random.uniform(0., 0.9999999, size=(nwarmupsteps, 2))

    # Get the temperature for the warm-up steps
    temp_i = temp(0)

    # Timing the warm-up steps
    start_gpu = cp.cuda.Event()
    end_gpu = cp.cuda.Event()
    start_gpu.record()
    start_cpu = time.perf_counter()
    
    # Warm up steps
    print(f'Starting simulation warm-up steps at {temp_i:.2f} K...') 
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
        # Add the energy from the lattice
        if lat_energies is not None:
            mnnenergies = cp.add(mnnenergies, lat_energies(mnnxy[:,:,0], mnnxy[:,:,1]), out=mnnenergies)
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
            print(f'Warm-up step {i+1:g} / {nwarmupsteps:g} completed.', end='\r', flush=True)
    print(f'Warm-up step {nwarmupsteps:g} / {nwarmupsteps:g} completed.', end='\r', flush=True)
    
    # Timing the warm-up steps
    end_cpu = time.perf_counter()
    end_gpu.record()
    end_gpu.synchronize()
    print(f'\nWarm-up steps completed in: {(end_cpu - start_cpu)/60:.3f} min CPU | {(cp.cuda.get_elapsed_time(start_gpu, end_gpu)/1000)/60:.3f} min GPU.')

    # Actual KMC steps
    print(f'\nStarting KMC steps...')
    i = 0  # Elementary KMC step counter
    time_i = 0   # Initial time of the simulation
    framenum = 0  # Sampled frame counter 
    temp_target = temp_i - (1/frames_per_kelvin)
    while temp_i >= Tf:
        # Generate new random numbers every nwarmupsteps
        j = i % nwarmupsteps
        if j == 0:
            rands = cp.random.uniform(0., 0.9999999, size=(nwarmupsteps, 2))
            print(f'KMC step {i+1:g} | Temperature {temp_i:.4f} K | Simulation time = {time_i:g} s', end='\r', flush=True)
        # Get the temperature at the current time
        temp_i = temp(time_i)
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
        # Add the energy from the lattice
        if lat_energies is not None:
            mnnenergies = cp.add(mnnenergies, lat_energies(mnnxy[:,:,0], mnnxy[:,:,1]), out=mnnenergies)
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
        time_i += dt
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
            # Update the target temperature for the next frame
            temp_target = temp_i - (1/frames_per_kelvin)
            # Update the frame number
            framenum += 1 
            print(f'KMC step {i+1:g} | Temperature {temp_i:.4f} K | Simulation time = {times[framenum+1]:g} s', end='\r', flush=True)
        else:
            # Update the molecule's new position to the chosen nearest neighbor
            ids[framenum,mol_id] = lnnids[ids[framenum]][mol_id,nn_id+1]
        # Apply early stopping if the KMC step time intervals are too large
        if dt > 10:
            print(f'\nKMC step time intervals are too large: {dt:g} seconds. Exiting simulation...')
            break
        # Increment the KMC step counter
        i += 1
    print(f'KMC step {i+1:g} | Temperature {temp_i:.4f} K | Simulation time = {times[framenum+1]:g} s', end='\r', flush=True)
    results = {
        'times': times.get(),
        'ids': ids.get(),
        'deltatimes': deltatimes.get(),
        'temperatures': temperatures.get(),
        'totalenergies': totalenergies.get()
    }
    print('\nSimulation complete!')
    return results