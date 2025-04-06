# Diffusion Maps with Nyström out-of-sample extension

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

## Usage

Here's a basic template of how to use the DiffusionMaps class:

```python
from diffusionmaps import DiffusionMaps

# Prepare your data
X_train = ...  # Your training data

# Initialize and fit the Diffusion Maps model
dm = DiffusionMaps(n_components=2, sigma=2.0, steps=1, alpha=0.5)
embeddings_train = dm.fit_transform(X_train)

# Use Nyström's method for embedding new data
X_new = ...  # New data
embeddings_new = dm.transform(X_new)
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

## License
This project is licensed under the MIT License - see the LICENSE file for details.
