# src/main.py

import pickle

from .lattice import KagomeLattice
from .simulation import simulate
    
def main():
    """
    Main function to initialize parameters, run the kinetic Monte Carlo simulation,
    and save the results.
    """    
    # Simulation parameters
    duration          = 0.00001  # Duration of the simulation in seconds
    frames_per_kelvin = 30      # Number of frames to sample per temperature change in Kelvin
    nwarmupsteps      = 100000   # Number of warm-up steps to take before the simulation starts
    progress_freq     = 250000  # Frequency of the progress output in KMC steps
    max_dt            = 60      # Maximum time step interval threshold in seconds
    save_file_path    = 'data/simdata.pkl'  # Path to save the simulation data

    # Define the Kagome lattice object for the simulation (all its parameters are shown here)
    # Note that the data types defined here are used for the entire simulation
    kagome = KagomeLattice(
        boxwidth=102.336, 
        lattice_constant=0.246, 
        energy_barrier=275.0, 
        debye_freq=10e10, 
        transition_type='t', 
        intdtype='int32', 
        floatdtype='float64'
    )

    # Energy parameters for the simulation
    energy_params = {
        'lattice':'triangular',  # Lattice energy can either be 'triangular' or 'none'
        'lattice_params':{
            'lattice_constant':3.936,  # Lattice constant of the triangular lattice (nm)
            'amplitude':-39.8,          # Amplitude of the triangular lattice energy (meV)
            'angle':0.0,               # Angle of the triangular lattice (degrees)
            'shift':[0.0, 0.0]         # Shift of the triangular lattice (nm)
        },
        'interaction':'total_impurity',  # Interaction energy parameters
        'interaction_params':{
            'Zval':1.0,  # Impurity charge value
            'epsr':4.22  # Dielectric constant of the medium 
        }
    }

    # Temperature parameters for the simulation
    temperature_params = {
        'temp_initial':30,  # Initial temperature of the simulation (K)
        'temp_final':8.5,   # Final temperature of the simulation (K)
        'method':'linear'   # Cooling method can be 'linear', 'exponential', or 'inv_exponential'
    }

    # Molecule parameters for the simulation
    molecule_params = {
        'n_molecules':800,  # Number of molecules to generate
        'threshold':12,      # Minimum distance (in lattice constants) between initial molecule positions
    }


    """ NOTHING TO CHANGE BELOW THIS LINE """
    # The `simulate()` function is used to run the simulation and return the simulation data as a dictionary
    try:
        simdata = simulate(
            kagome, 
            energy_params, 
            temperature_params, 
            molecule_params, 
            duration, 
            frames_per_kelvin=frames_per_kelvin, 
            nwarmupsteps=nwarmupsteps,
            progress_freq=progress_freq,
            max_dt=max_dt
        )
    except Exception as e:
        print(f'\nAn error occurred during the simulation: {e}')
        return

    # Save the simulation data
    with open(save_file_path, 'wb') as f:
        pickle.dump(simdata, f)

if __name__ == "__main__":
    main()