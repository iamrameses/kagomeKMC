# kagomeKMC: Kagome kinetic Monte Carlo simulation

<img src="images/example_molecules_and_disclinations_plot.png" alt="Example molecules and disclinations plot" width="900"><br/>

[kagomeKMC](https://github.com/iamrameses/kagomeKMC) is a rejection-free *on-lattice* kinetic Monte Carlo simulation of the time-evolution of nanometer-sized molecules on a surface of Kagome lattice sites. Some key features of *kagomeKMC* are:
- Periodic boundary conditions.
- Transition rates takes into consideration both the lattice potential energy surface and lateral interactions between molecules.
- System temperature changes according to a defined temperature curve.
- Computations are performed on GPU via CuPy operations or Numba kernels.

For basic functionality and running a simulation, see the example notebook `240609_kagomeKMC_v1_tutorial.ipynb`.

## Setting up environment on Berkeley Savio Servers
On the Open OnDemand Dashboard, go to "Clusters" -> ">_BRC Shell Access", then load the modules we need.

```commandline
module load python/3.11.4 cuda/12.2 gcc git
```

Navigate to location where you want the repository and clone it.

```command line
cd /global/home/users/<your_username>/
git clone https://github.com/iamrameses/kagomeKMC.git
```
Since Savio gives users a limited amount of space in our home directories, its best to install all conda environments and packages in our scratch folder. We also set conda's default solver to `libmamba` as it usually much faster at installing packages (be patient, as it can take a while). If you encounter an error due to `libmamba`, just revert it back to the `classic` solver. It is important that `cupy` be installed last so that it is version `12.3.0` (for some reason, more recent versions result in a very slow simulation).

```commandline
KMCENV=/global/scratch/users/<your_username>/environments/kagomekmc
rm -rf $KMCENV
export CONDA_PKGS_DIRS=/global/scratch/users/<your_username>/tmp/.conda
conda create --prefix $KMCENV python=3.11
source activate $KMCENV
conda config --set solver libmamba
conda install -c conda-forge cudatoolkit
conda install -c conda-forge numba
conda install -c conda-forge freud
conda install -c conda-forge pandas matplotlib ipympl
conda install -c conda-forge cupy=12.3.0
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=kagomekmc
```
Trying to install `cudatoolkit`, `cupy`, and `numba` in one line caused issues for me, so I recommend installing them separately. It's also recommended to modify your `.bashsrc` file in your root folder by adding the following lines at the very end of the script, otherwise you'll have to keep typing the above commands just to activate your environment each time you open a new shell.

```commandline
export KMCENV=/global/scratch/users/<your_username>/environments/kagomekmc
export CONDA_PKGS_DIRS=/global/scratch/users/<your_username>/tmp/.conda
```

## Requesting a Jupyter Server interactive session with GPU 
Go to "Interactive Apps" and select "Jupyter Server - compute via Slurm using Savio partitions". Name the job whatever you like, but I would recommend the below settings as a V100 GPU is ideal for this simulation.

<img src="images/jupyter_server_request_recommended_settings.png" alt="Example molecules and disclinations plot" width="500">