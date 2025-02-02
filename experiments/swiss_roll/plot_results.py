import os
import h5py
import numpy as np
from matplotlib import pyplot as plt

plt.style.use('experiments/science.mplstyle')


def get_max_range(X):
    x_max = np.max(X, axis=0)
    x_min = np.min(X, axis=0)
    x_range = x_max - x_min
    max_range = np.max(x_range)

    return max_range


def set_equal_ranges(ax, max_range):
    # Get the current axis limits
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # Set new limits with the same range for both axes
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2

    max_range = max_range * 1.05

    ax.set_xlim(x_center - max_range / 2, x_center + max_range / 2)
    ax.set_ylim(y_center - max_range / 2, y_center + max_range / 2)

    # Set equal aspect ratio
    ax.set_aspect('equal', adjustable='box')

    return ax


def plot_original(X, y, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(3, 3), subplot_kw={"projection": "3d"})
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, alpha=1)
    # ax.set_xlabel('$x_1$')
    ax.set_xlim([-13, 13])
    # ax.set_ylabel('$x_2$')
    ax.set_ylim([-3, 23])
    # ax.set_zlabel('$x_3$')
    ax.set_zlim([-13, 13])
    ax.view_init(15, -72)
    # ax.dist = 12
    ax.grid(False)
    fig.tight_layout()
    
    for format in ('.pdf', '.png', '.svg'):
        fig.savefig(os.path.join(output_dir, filename + format))

    plt.close(fig)


def plot_projection(X, y, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    max_range = get_max_range(X)
    ndims = X.shape[-1]
    if ndims > 1:
        for dim1 in range(0, ndims):
            for dim2 in range(dim1 + 1, ndims):
                fig, ax = plt.subplots(figsize=(3, 3), constrained_layout=True)
                ax.scatter(X[:, dim1], X[:, dim2], c=y)
                # Remove the ticks
                ax.set_xticks([])
                ax.set_yticks([])
                # Remove the tick labels
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                # ax.set_xlabel(r'$\Psi_1$')
                # ax.set_ylabel(r'$\Psi_2$')
                ax = set_equal_ranges(ax, max_range) # ax.set_box_aspect(1)

                for format in ('.pdf', '.png', '.svg'):
                    fig.savefig(os.path.join(output_dir, filename + f'_dims_{dim1+1}_{dim2+1}' + format))
                
                plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(3, 3), constrained_layout=True)
        ax.scatter(X[:, 0], np.zeros(X.shape[0]), c=y)
        # Remove the ticks
        ax.set_xticks([])
        ax.set_yticks([])
        # Remove the tick labels
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # ax.set_xlabel(r'$\Psi_1$')
        # ax.set_ylabel(r'$\Psi_2$')
        ax = set_equal_ranges(ax, max_range) # ax.set_box_aspect(1)

        for format in ('.pdf', '.png', '.svg'):
            fig.savefig(os.path.join(output_dir, filename + f'_dims_{1}' + format))
        
        plt.close(fig)


output_dir = '/scratch/sgarcia/nystrom_dm/experiments/swiss_roll/results'
# Load data from the HDF5 file
with h5py.File(os.path.join(output_dir, 'results.h5'), "r") as file:
    # Load original data
    group_0 = file["original"]
    X_a = np.array(group_0["X_a"][:])
    y_a = np.array(group_0["y_a"][:])
    X_b = np.array(group_0["X_b"][:])
    y_b = np.array(group_0["y_b"][:])

    # Load data for approach 1
    group_1 = file["difussion_maps"]
    X_a_red_1 = np.array(group_1["X_a_red"][:])
    X_b_red_1 = np.array(group_1["X_b_red"][:])

    # Load data for approach 3
    group_3 = file["nystrom"]
    X_a_red_3 = np.array(group_3["X_a_red"][:])
    X_b_red_3 = np.array(group_3["X_b_red"][:])

            
plot_original(X_a, y_a, output_dir, 'orig_a')
plot_original(X_b, y_b, output_dir, 'orig_b')
plot_projection(X_a_red_1, y_a, output_dir, 'red_a_dm')
plot_projection(X_a_red_3, y_a, output_dir, 'red_a_nys')
plot_projection(X_b_red_1, y_b, output_dir, 'red_b_dm')
plot_projection(X_b_red_3, y_b, output_dir, 'red_b_nys')