# Machine Learning of Vertical Fluxes by Unresolved Midlatitude Mesoscale Processes

This repository documents the code used for the study **"Machine Learning of Vertical Fluxes by Unresolved Midlatitude Mesoscale Processes"** by *Erisa Ismaili*, *Robert Jnglin Wills*, *Tom Beucler*.

## General

### Conda Environment

This project uses a Conda environment named after the project:

`mesoflux-ml`

and its specification, including exact package versions, can be found in the `environment.yml`

### Source Code
See the most important python source files `src/` directory

### Data
Climate data (coarse-grained NetCDF files and processed NumPy arrays) can be stored in the `data/` directory.

### Physics 

Physical diagnostic functions are implemented in:

- `src/physics/hybrid_sigma_pressure.py`
- `src/physics/thermo.py`

These modules contain functions to:
- compute the pressure field on hybrid sigma-pressure coordinates,
- convert specific humidity to relative humidity (RH),
- compute buoyancy.

Relative humidity is used for clustering.

### K-means Clustering 

K-means clustering is performed directly on data from the NetCDF file.

Functions for:
- performing the clustering,
- assigning cluster labels,
- analysing cluster properties

are located in `src/clustering/tools.py`

### Plotting

Various plotting functions (distributions, profiles, maps, zonal means) are found in `src/data/plot.py` 

## Machine learning 

### 1. Coarse graining
Simulation output is coarse-grained using the scripts in: `coarse_graining/`
These scripts generate the coarse-resolution fields in the NetCDF file used throughout the project.

### 2. Preprocessing

The file `src/data/preprocessing.py` 
contains functions that transform coarse-grained NetCDF data into NumPy arrays to feed in machine-learning models.

This includes:

- converting NetCDF files to NumPy arrays,
- selecting input and output variables,
- normalization,
- removal of extreme values above a specified threshold,
- optional removal of entire clusters,
- saving processed datasets.

### 3.Machine Learning Utilities

Machine learning utilities are found in `src/data/ml`:

This includes:
- loss functions,
- conversion of NumPy arrays to PyTorch Dataset objects,
- a trainer class implementing training and evaluation loops over epochs.

### 4. Training

Hyperparameter optimization with optuna, cross-validation, final training can be performed with the scripts in `scripts/train/`

### 5. Predict
Predictions are performed on th bootstrapping dataset (includes the test set), using scripts in `scripts/predict/`

### 6. Shapley values
Shapley values are computed with the scripts in `scripts/shapley/'


## Notebooks