import os
import numpy as np
from scipy.spatial.distance import pdist
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from DiffusionMaps import DiffusionMaps

plt.style.use('experiments/science.mplstyle')
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
linestyles = ['-', '--', '-.', ':', (5, (10, 3)), (0, (3, 1, 1, 1, 1, 1))]


def get_sigma(X, q=0.5):
    X_flat = X.reshape((X.shape[0], -1))
    distances = pdist(X_flat, metric='euclidean')
    sigma = np.quantile(distances, q)
    
    return sigma


def gaussian_density(x, mean, var):
    y = 1/np.sqrt(2*np.pi*var)*np.exp(-(x-mean)**2/(2*var))
    ids = y == 0 | np.isnan(y) | np.isinf(y)
    y = np.where(ids, np.mean(y[~ids]), y)

    return y


def log_likelihood(eigenvalues_1, eigenvalues_2):
    p, q = len(eigenvalues_1), len(eigenvalues_2)
    sample_mean_1 = np.mean(eigenvalues_1)
    sample_var_1 = np.var(eigenvalues_1, ddof=1 if len(eigenvalues_1) > 1 else 0)
    sample_mean_2 = np.mean(eigenvalues_2)
    sample_var_2 = np.var(eigenvalues_2, ddof=1 if len(eigenvalues_1) > 1 else 0)
    var = ((p - 1)*sample_var_1 + (q - 1)*sample_var_2)/(p + q - 2)
    l = np.sum(np.log(gaussian_density(eigenvalues_1, sample_mean_1, var)))\
        + np.sum(np.log(gaussian_density(eigenvalues_2, sample_mean_2, var)))
    
    return l


def find_optimal_hyperparameters(X, q_vals, alpha_vals, steps_vals, output_dir='', max_components=None):
    max_components = max_components if max_components else (X.shape[-1] - 1)
    n_components_vals = np.arange(1, max_components + 1)
    l_max = -np.inf
    fig, axes = plt.subplots(len(alpha_vals), 1, figsize=(6, 6), sharex=True, sharey=True)
    for ax, alpha in zip(axes, alpha_vals):
        for j, steps in enumerate(steps_vals):
            for k, q in enumerate(q_vals):
                l_vals = []
                for n_components in n_components_vals:
                    DM = DiffusionMaps(get_sigma(X, q), n_components, steps, alpha)
                    _ = DM.fit_transform(X)
                    eigenvalues = DM.lambdas[1:]**steps
                    l = log_likelihood(eigenvalues[:n_components], eigenvalues[n_components:])
                    l_vals.append(l)
                    if l > l_max:
                        l_max = l
                        best_hyperparameters = (n_components, q, alpha, steps)
                        
                ax.plot(n_components_vals, l_vals, color=colors[j], linestyle=linestyles[k])
                ax.set_title(f'$\\alpha = {alpha}$')
                ax.set_ylabel('log-likelihood')

    ax.set_xlabel('$d$')
    ax.set_xticks(n_components_vals)
    # Create custom legends
    q_legend = {f'${q:.2f}$': Line2D([0], [0], linewidth=2, color='gray', linestyle=linestyles[i]) for i, q in enumerate(q_vals)}
    steps_legend = {f'${steps}$': Line2D([0], [0], linewidth=2, color=colors[i]) for i, steps in enumerate(steps_vals)}
    fig.legend(q_legend.values(), q_legend.keys(), title="Valor de $q$", loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=6, handletextpad=0.3, columnspacing=0.3)
    fig.legend(steps_legend.values(), steps_legend.keys(), title="Valor de $t$", loc='lower center', bbox_to_anchor=(0.5, -0.13), ncol=7, handletextpad=0.3, columnspacing=0.3, handlelength=2.5)
    for format in ('.pdf', '.png', '.svg'):
        fig.savefig(os.path.join(output_dir, 'l_values' + format))
    
    return best_hyperparameters