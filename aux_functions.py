import numpy as np
from scipy.spatial.distance import pdist
from tqdm import tqdm

from DiffusionMaps import DiffusionMaps


def get_sigma(X, q=0.5):
    X_flat = X.reshape((X.shape[0], -1))
    distances = pdist(X_flat, metric='euclidean')
    sigma = np.quantile(distances, q)
    
    return sigma


def gaussian_density(x, mean, var):
    return 1/np.sqrt(2*np.pi*var)*np.exp(-(x-mean)**2/(2*var))


def log_likelihood(eigenvalues_1, eigenvalues_2):
    p, q = len(eigenvalues_1), len(eigenvalues_2)
    sample_mean_1 = np.mean(eigenvalues_1)
    sample_var_1 = np.var(eigenvalues_1, ddof=1 if len(eigenvalues_1) > 1 else 0)
    sample_mean_2 = np.mean(eigenvalues_2)
    sample_var_2 = np.var(eigenvalues_2, ddof=1 if len(eigenvalues_1) > 1 else 0)
    var = ((p - 1)*sample_var_1 + (q - 1)*sample_var_2)/(p + q - 2)
    l = np.sum(gaussian_density(eigenvalues_1, sample_mean_1, var))\
        + np.sum(gaussian_density(eigenvalues_2, sample_mean_2, var))
    
    return l


def find_optimal_hyperparameters(X, q_vals, alpha_vals, steps_vals, max_components=None):
    max_components = max_components if max_components else (X.shape[-1] - 1)
    l_max = -np.inf
    for n_components in tqdm(range(1, max_components + 1)):
        for q in q_vals:
            for alpha in alpha_vals:
                for steps in steps_vals:
                    sigma = get_sigma(X, q)
                    DM = DiffusionMaps(sigma, n_components, steps, alpha)
                    _ = DM.fit_transform(X)
                    eigenvalues = DM.lambdas[1:]
                    l = log_likelihood(eigenvalues[:n_components], eigenvalues[n_components:])
                    if l > l_max:
                        l_max = l
                        best_hyperparameters = (n_components, q, alpha, steps)
    
    return best_hyperparameters


