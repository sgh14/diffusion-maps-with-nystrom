# diffusion-maps-with-nystrom

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<!-- Add other badges if you set up CI/CD or publish to PyPI -->
<!-- e.g., [![PyPI version](https://badge.fury.io/py/diffusionmaps.svg)](https://badge.fury.io/py/diffusionmaps) -->
<!-- e.g., [![Build Status](https://github.com/your_username/diffusion-maps-repo/actions/workflows/python-package.yml/badge.svg)](https://github.com/your_username/diffusion-maps-repo/actions/workflows/python-package.yml) -->


A Python implementation of the Diffusion Maps algorithm for non-linear dimensionality reduction, designed to follow scikit-learn conventions.

Diffusion Maps model data as a graph and analyze the diffusion process (random walk) on this graph. This allows capturing the underlying geometry and connectivity of the data, embedding it into a lower-dimensional space where Euclidean distances correspond to "diffusion distances" in the original space.

## Key Features

*   Implements the Diffusion Maps algorithm with alpha-normalization.
*   Scikit-learn compatible API (`fit`, `transform`, `fit_transform`).
*   Nystrom extension for out-of-sample transformation.
*   RBF (Gaussian) kernel for affinity calculation.
*   Configurable diffusion time steps (`steps`).

## Installation

You can install this package directly from GitHub using pip:

```bash
pip install git+https://github.com/sgh14/diffusion-maps-with-nystrom.git 
```
Or, to install a specific version (e.g., v0.1.0 tag):
```bash
pip install git+https://github.com/sgh14/diffusion-maps-with-nystrom.git@v0.1.0
```

## Usage Example

Here's a basic example of how to use the DiffusionMaps class:

```python
import numpy as np
from diffusionmaps import DiffusionMaps # Import the class
import matplotlib.pyplot as plt

# 1. Create some sample data (e.g., points along a curve)
n_samples = 200
t = np.linspace(0, 4 * np.pi, n_samples)
x = t * np.cos(t)
y = t * np.sin(t)
noise = 0.5
X = np.vstack((x, y)).T + noise * np.random.randn(n_samples, 2)

# 2. Initialize and fit the Diffusion Maps model
# sigma controls the kernel width (locality)
# n_components is the target dimension
dm = DiffusionMaps(n_components=2, sigma=2.0, steps=1, alpha=0.5)

# 3. Compute the diffusion map embedding
X_diff_map = dm.fit_transform(X)

# The first component (X_diff_map[:, 0]) often captures the main progression
# The second component (X_diff_map[:, 1]) captures other variations

# 4. Visualize the results
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Original data colored by the first diffusion component
sc1 = axes[0].scatter(X[:, 0], X[:, 1], c=X_diff_map[:, 0], cmap='viridis')
axes[0].set_title('Original Data (Colored by DiffMap[0])')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')
fig.colorbar(sc1, ax=axes[0], label='1st Diffusion Component')

# Diffusion Map embedding colored by the first component
sc2 = axes[1].scatter(X_diff_map[:, 0], X_diff_map[:, 1], c=X_diff_map[:, 0], cmap='viridis')
axes[1].set_title('Diffusion Map Embedding')
axes[1].set_xlabel('Diffusion Component 1')
axes[1].set_ylabel('Diffusion Component 2')
fig.colorbar(sc2, ax=axes[1], label='1st Diffusion Component')

plt.tight_layout()
plt.show()

# --- Example of transforming new data ---
# Create some new points along the same curve
t_new = np.linspace(np.pi, 3 * np.pi, 50) # Points in the middle
x_new = t_new * np.cos(t_new)
y_new = t_new * np.sin(t_new)
X_new = np.vstack((x_new, y_new)).T + noise * np.random.randn(50, 2)

# Transform using the fitted model (Nystrom extension)
X_new_diff_map = dm.transform(X_new)

# Add new points to the plots
axes[0].scatter(X_new[:, 0], X_new[:, 1], c='red', marker='x', s=50, label='New Data')
axes[1].scatter(X_new_diff_map[:, 0], X_new_diff_map[:, 1], c='red', marker='x', s=50, label='New Data Transformed')
axes[0].legend()
axes[1].legend()

# Re-display plot if running interactively or save figure
# plt.show() or fig.savefig('diffusion_map_example.png')
```

## API Overview
The main class is diffusionmaps.DiffusionMaps.

```python
DiffusionMaps(n_components: int, sigma: float, steps: int = 1, alpha: float = 0.0)
```

* `n_components`: Target dimensionality.
* `sigma`: Scale parameter for the RBF kernel ($\exp(-||x-y||^2 / (2 \sigma^2))$). Controls locality.
* `steps`: Diffusion time (exponent t for eigenvalues lambda^t). Default is 1.
* `alpha`: Kernel normalization parameter (0.0, 0.5, or 1.0 are common). Controls density influence.

See the class docstrings for detailed information on methods and attributes.

## Contributing
Contributions are welcome! Please feel free to open an issue or submit a pull request.
(Add more specific guidelines if desired - e.g., coding standards, testing).

## License
This project is licensed under the MIT License - see the LICENSE file for details.
