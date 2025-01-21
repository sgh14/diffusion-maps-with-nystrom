import os
import h5py
import numpy as np

from DiffusionMaps import DiffusionMaps
from aux_functions import get_sigma, find_optimal_hyperparameters
from experiments.swiss_roll.load_data import get_datasets
from experiments.metrics import mae

# output_dir = '/scratch/sgarcia/nystrom_dm/experiments/swiss_roll/results'
output_dir = 'experiments/swiss_roll/results'
os.makedirs(output_dir, exist_ok=True)

# Get the data
(X_a, y_a), (X_b, y_b) = get_datasets(npoints=2000, split=0.5, seed=123, noise=0)
X = np.vstack([X_a, X_b])

# Find optimal values for n_components, q, steps and alpha
q_vals = np.linspace(0, 1, 200)
alpha_vals = np.array([0, 1])
steps_vals = np.array([2**i for i in range(10)])
# n_components, q, alpha, steps = find_optimal_hyperparameters(X_a, q_vals, alpha_vals, steps_vals)
n_components, q, alpha, steps = 2, 5e-3, 1, 100
sigma = get_sigma(X_a, q)
DM = DiffusionMaps(sigma=sigma, n_components=n_components, steps=steps, alpha=alpha)

# Approach 1: original Diffusion Maps
X_red_1 = DM.fit_transform(X)
X_b_red_1 = X_red_1[len(X_a):]

# Approach 3: Nyström to extend existing embedding
X_a_red_3 = DM.fit_transform(X_a)
X_b_red_3 = DM.transform(X_b)
mae_3, mae_3_conf_int = mae(X_b_red_1, X_b_red_3)


with h5py.File(os.path.join(output_dir, 'results.h5'), "w") as file:
    # Group for hyperparameters
    group_hyperparameters = file.create_group("hyperparameters")
    group_hyperparameters.create_dataset("n_components", data=n_components)
    group_hyperparameters.create_dataset("q", data=q)
    group_hyperparameters.create_dataset("alpha", data=alpha)
    group_hyperparameters.create_dataset("steps", data=steps)
    group_hyperparameters.create_dataset("sigma", data=sigma)

    # Group for original data
    group_0 = file.create_group("original")
    group_0.create_dataset("X_b", data=X_b, compression='gzip')
    group_0.create_dataset("y_b", data=y_b, compression='gzip')

    # Group for approach 1
    group_1 = file.create_group("difussion_maps")
    group_1.create_dataset("X_b_red", data=X_b_red_1, compression='gzip')

    # Group for approach 3
    group_3 = file.create_group("nystrom_extend")
    group_3.create_dataset("X_b_red", data=X_b_red_3, compression='gzip')
    group_3.create_dataset("mae", data=mae_3)
    # group_3.create_dataset("mae_conf_int", data=mae_3_conf_int)
