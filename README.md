# kagomeKMC: Kagome kinetic Monte Carlo simulation

<img src="images/example_molecules_and_disclinations_plot.png" alt="Example molecules and disclinations plot" width="900">\

# Setting up environment on Berkeley Savio Servers
On the Open OnDemand Dashboard, go to "Clusters" -> ">_BRC Shell Access", then load the modules we need.

```commandline
module load cuda gcc git
```

Navigate to location where you want the repository and clone it.

```command line
cd /global/home/users/<your_username>/
git clone https://github.com/iamrameses/kagomeKMC.git
```
Since Savio gives users a limited amount of space in our home directories, its best to install all conda environments and packages in our scratch folder. We also set conda's default solver to `libmamba` as it usually much faster at installing packages (be patient, as it can take a while).

```commandline
KMCENV=/global/scratch/users/<your_username>/environments/kagomekmc
rm -rf $KMCENV
export CONDA_PKGS_DIRS=/global/scratch/users/<your_username>/tmp/.conda
conda create --prefix $KMCENV python=3.11.5
source activate $KMCENV
conda config --set solver libmamba
conda install -c conda-forge cudatoolkit 
conda install -c conda-forge cupy 
conda install -c conda-forge numba 
conda install -c conda-forge scipy pandas matplotlib ipympl freud
pip install notebook
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=kagomekmc
```
Trying to install `cudatoolkit`, `cupy`, and `numba` in one line caused issues for me, so I recommend installing them separately. It's also recommended to modify your `.bashsrc` file in your root folder by adding the following lines at the very end of the script, otherwise you'll have to keep typing the above commands just to activate your environment each time you open a new shell.

```commandline
export KMCENV=/global/scratch/users/<your_username>/environments/kagomekmc
export CONDA_PKGS_DIRS=/global/scratch/users/<your_username>/tmp/.conda
```

# Requesting a Jupyter Server interactive session with GPU 
Go to "Interactive Apps" and select "Jupyter Server - compute via Slurm using Savio partitions". Name the job whatever you like, but I would recommend the below settings as a V100 GPU is ideal for this simulation.

<img src="images/jupyter_server_request_recommended_settings.png" alt="Example molecules and disclinations plot" width="500">\