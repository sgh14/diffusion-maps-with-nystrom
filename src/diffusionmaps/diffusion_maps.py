import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class DiffusionMaps(BaseEstimator, TransformerMixin):
    """
    Diffusion Maps.

    Performs non-linear dimensionality reduction using the diffusion maps
    algorithm. It models the data as a graph where nodes are data points and
    edges are weighted by similarity (kernel). It then analyzes the random walk
    on this graph to find low-dimensional embeddings that preserve diffusion
    distances.

    Parameters
    ----------
    n_components : int
        The number of diffusion components (dimensions) to return.
    sigma : float
        The scale parameter for the RBF (Gaussian) kernel. It controls the
        locality of the connections in the graph. Corresponds to the `sigma`
        in `exp(-||x-y||^2 / (2 * sigma^2))`.
    steps : int, default=1
        The number of time steps for the diffusion process (exponent `t` for
        the eigenvalues `lambda^t`). Larger `steps` emphasize global structure.
    alpha : float, default=0.0
        Normalization parameter for the kernel matrix.
        - `alpha = 0`: Standard diffusion maps (row-stochastic P).
        - `alpha = 0.5`: Fokker-Planck diffusion.
        - `alpha = 1`: Laplace-Beltrami diffusion (requires manifold assumption).
        Controls the influence of data density on the diffusion process.

    Attributes
    ----------
    gamma : float
        Computed parameter for the RBF kernel, `gamma = 1 / (2 * sigma^2)`.
    X : np.ndarray of shape (n_samples, n_features)
        The training data used to fit the model.
    d_K : np.ndarray of shape (n_samples,)
        Degree vector computed from the raw RBF kernel matrix K.
    W : np.ndarray of shape (n_samples, n_samples)
        The alpha-normalized kernel matrix.
    d_W : np.ndarray of shape (n_samples,)
        Degree vector computed from the alpha-normalized kernel matrix W.
    pi : np.ndarray of shape (n_samples,)
        The stationary distribution derived from `d_W`.
    lambdas : np.ndarray of shape (n_samples,)
        The eigenvalues of the matrix A used for spectral decomposition, sorted
        in descending order. The first eigenvalue should be close to 1.
    phis : np.ndarray of shape (n_samples, n_samples)
        The corresponding eigenvectors of the matrix A, sorted by eigenvalue
        and orientation-fixed. `phis[:, i]` is the eigenvector for `lambdas[i]`.
    is_fitted_ : bool
        True if the estimator has been fitted.

    References
    ----------
    Coifman, R. R., & Lafon, S. (2006). Diffusion maps. Applied and
    computational harmonic analysis, 21(1), 5-30.
    """
    def __init__(self, n_components: int, sigma: float, steps: int = 1, alpha: float = 0.0):
        if not isinstance(n_components, int) or n_components < 1:
            raise ValueError("n_components must be a positive integer.")
        if not isinstance(sigma, (int, float)) or sigma <= 0:
            raise ValueError("sigma must be a positive number.")
        if not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be a positive integer.")
        if not isinstance(alpha, (int, float)):
            raise ValueError("alpha must be a number.")

        self.n_components = n_components
        self.sigma = sigma
        self.steps = steps
        self.alpha = alpha
        self.gamma = 1 / (2 * sigma**2)


    @staticmethod
    def _rbf_kernel(X: np.ndarray, Y: np.ndarray = None, gamma: float = 1.0) -> np.ndarray:
        """
        Computes the RBF (Gaussian) kernel between two sets of points.

        K(x, y) = exp(-gamma * ||x - y||^2)

        Parameters
        ----------
        X : np.ndarray of shape (n_samples_X, n_features)
            First set of data points.
        Y : np.ndarray of shape (n_samples_Y, n_features), optional
            Second set of data points. If None, computes kernel between X and itself.
        gamma : float
            Parameter for the RBF kernel (1 / (2 * sigma^2)).

        Returns
        -------
        np.ndarray
            The computed kernel matrix of shape (n_samples_X, n_samples_Y).
        """
        # pairwise_distances computes squared Euclidean distances efficiently
        distances_sq = pairwise_distances(X, Y, metric='sqeuclidean', n_jobs=-1)
        K = np.exp(-gamma * distances_sq)

        return K

    
    @staticmethod
    def _degree_vector(K: np.ndarray) -> np.ndarray:
        """
        Computes the degree vector (row sums) of a kernel/affinity matrix.

        d_i = sum_j K_ij

        Parameters
        ----------
        K : np.ndarray of shape (n_samples, n_samples) or (n_samples_X, n_samples_Y)
            The kernel or affinity matrix.

        Returns
        -------
        np.ndarray
            The degree vector (sum along axis 1).
        """
        # Sum along rows (axis=1)
        return np.sum(K, axis=1)


    @staticmethod
    def _normalize_by_degree(M: np.ndarray, d_i: np.ndarray, d_j: np.ndarray = None, alpha: float = 0.0) -> np.ndarray:
        """
        Performs alpha-normalization on a matrix M using degree vectors.

        M_norm[i, j] = M[i, j] / (d_i[i]**alpha * d_j[j]**alpha)

        Parameters
        ----------
        M : np.ndarray of shape (n_samples_i, n_samples_j)
            The matrix to normalize.
        d_i : np.ndarray of shape (n_samples_i,)
            Degree vector corresponding to the rows of M.
        d_j : np.ndarray of shape (n_samples_j,), optional
            Degree vector corresponding to the columns of M. If None, uses d_i.
        alpha : float
            The exponent for the degree normalization.

        Returns
        -------
        np.ndarray
            The normalized matrix M_norm.
        """
        d_i_alpha = np.power(d_i, alpha)
        d_j_alpha = d_i_alpha if d_j is None else np.power(d_j, alpha)
        M_alpha = M / np.outer(d_i_alpha, d_j_alpha)

        return M_alpha


    @staticmethod
    def _fix_vector_orientation(vectors: np.ndarray) -> np.ndarray:
        """
        Ensures a canonical orientation for eigenvectors.

        Fixes the sign of each eigenvector such that the first non-zero
        element in the vector is positive. This provides deterministic output.

        Parameters
        ----------
        vectors : np.ndarray of shape (n_samples, n_vectors)
            The eigenvectors (or other vectors) to orient.

        Returns
        -------
        np.ndarray
            The eigenvectors with fixed orientation. Modified in-place.
        """
        for i in range(vectors.shape[1]):
            # Find the first non-zero element in the vector
            first_nonzero = np.nonzero(vectors[:, i])[0][0]
            # If that element is negative, flip the sign of the whole vector
            if vectors[first_nonzero, i] < 0:
                vectors[:, i] *= -1

        return vectors


    @staticmethod
    def _spectral_decomposition(A: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Performs spectral decomposition of a symmetric matrix A.

        Computes eigenvalues and eigenvectors, sorts them by eigenvalue
        in descending order, and fixes eigenvector orientation.

        Parameters
        ----------
        A : np.ndarray of shape (n_samples, n_samples)
            The symmetric matrix for decomposition (e.g., normalized affinity).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            - eigenvalues : np.ndarray of shape (n_samples,) sorted descending.
            - eigenvectors : np.ndarray of shape (n_samples, n_samples) sorted
                             according to eigenvalues and orientation-fixed.
        """
        # Compute the eigenvalues and right eigenvectors for the symmetric matrix A
        eigenvalues, eigenvectors = np.linalg.eigh(A)
        # eigh returns eigenvalues in ascending order, so we reverse the order
        order = np.argsort(eigenvalues)[::-1]
        # Sort eigenvalues and eigenvectors in descending order
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        # Fix eigenvectors orientation for deterministic output
        eigenvectors = DiffusionMaps._fix_vector_orientation(eigenvectors)
        
        return eigenvalues, eigenvectors


    @staticmethod
    def _stationary_distribution(d_W: np.ndarray) -> np.ndarray:
        """
        Computes the stationary distribution pi of the Markov chain defined by W.

        For the diffusion process associated with the alpha-normalized kernel W,
        the stationary distribution pi is proportional to the degree vector d_W.

        pi_i = d_W_i / sum_k d_W_k

        Parameters
        ----------
        d_W : np.ndarray of shape (n_samples,)
            Degree vector derived from the normalized kernel W (sum of rows of W).

        Returns
        -------
        np.ndarray
            The stationary distribution pi, shape (n_samples,). Sums to 1.
        """
        pi = d_W / np.sum(d_W)

        return pi
    

    def _get_embedding(self, phis: np.ndarray, lambdas: np.ndarray, pi: np.ndarray) -> np.ndarray:
        """
        Computes the diffusion map embedding coordinates.

        This involves transforming the eigenvectors of the symmetric matrix A (phis)
        into the right eigenvectors of the transition matrix P (psis), and then
        scaling by the eigenvalues raised to the power of the number of steps (`self.steps`).

        The diffusion map embedding Psi_t(x_i) is given by the vector:
        [ lambda_1^t * psi_1(i), lambda_2^t * psi_2(i), ..., lambda_k^t * psi_k(i) ]
        where psi_j are the right eigenvectors of P, and k is `n_components`.

        The right eigenvectors of P (psis) relate to eigenvectors of A (phis) by:
        psi_j = phi_j / sqrt(pi)  (where pi is the stationary distribution).

        Parameters
        ----------
        phis : np.ndarray of shape (n_samples, n_components)
            Selected eigenvectors of the symmetric matrix A (corresponding to the
            largest eigenvalues, excluding the first trivial one if appropriate).
        lambdas : np.ndarray of shape (n_components,)
            Corresponding selected eigenvalues of A.
        pi : np.ndarray of shape (n_samples,)
            The stationary distribution.

        Returns
        -------
        np.ndarray
            The diffusion map embedding coordinates, shape (n_samples, n_components).
        """
        # Compute P right eigenvectors (psis) from A eigenvectors (phis)
        psis = phis / np.sqrt(pi[:, np.newaxis])
        # Compute the final embedding coordinates by scaling psis
        embedding = psis * np.power(lambdas[np.newaxis, :], self.steps)

        return embedding


    def fit(self, X: np.ndarray, y: None = None) -> 'DiffusionMaps':
        """
        Fits the Diffusion Maps model to the data X.

        Computes the kernel matrix, performs normalization based on `alpha`,
        and executes the spectral decomposition of the resulting symmetric
        matrix A to find eigenvalues and eigenvectors.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The training input samples.
        y : None
            Ignored. Present for compatibility with scikit-learn pipelines.

        Returns
        -------
        self : DiffusionMaps
            The fitted estimator instance.
        """
        self.X = X.reshape((X.shape[0], -1))
        # Compute the kernel
        K = self._rbf_kernel(self.X, self.X, self.gamma)
        # Compute degree vector        
        self.d_K = self._degree_vector(K)
        # Compute the normalized kernel
        self.W = self._normalize_by_degree(K, self.d_K, self.d_K, self.alpha)
        # Compute degree vector
        self.d_W = self._degree_vector(self.W)
        # Compute the stationary distribution
        self.pi = self._stationary_distribution(self.d_W)
        # Compute the matrix A
        A = self._normalize_by_degree(self.W, self.d_W, self.d_W, 0.5)
        # Get the eigenvectors and eigenvalues
        self.lambdas, self.phis = self._spectral_decomposition(A)
        # Scikit-learn convention
        self.is_fitted_ = True 

        return self


    def fit_transform(self, X: np.ndarray, y: None = None) -> np.ndarray:
        """
        Fits the model to X and returns the diffusion map embedding of X.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            The training input samples.
        y : None
            Ignored. Present for compatibility with scikit-learn pipelines.

        Returns
        -------
        np.ndarray
            The diffusion map embedding of the training data, shape
            (n_samples, n_components). Uses eigenvectors 1 to n_components.
        """
        self.fit(X)
        # Reduce dimension
        lambdas_red = self.lambdas[1:self.n_components + 1]
        phis_red = self.phis[:, 1:self.n_components + 1]            
        # Compute the new coordinates
        X_red = self._get_embedding(phis_red, lambdas_red, self.pi)

        return X_red

    
    @staticmethod
    def _nystrom_extension(K_mix: np.ndarray, old_eigenvectors: np.ndarray, old_eigenvalues: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Applies the Nystrom extension to approximate eigenvectors for new data.

        The formula used is: phi_new = K_mix @ phi / lambda

        Parameters
        ----------
        K_mix : np.ndarray of shape (n_new_samples, n_old_samples)
            Matrix relating new samples to original samples (e.g., A_mix).
        old_eigenvectors : np.ndarray of shape (n_old_samples, n_components)
            Eigenvectors computed on the original data (subset corresponding
            to selected eigenvalues).
        old_eigenvalues : np.ndarray of shape (n_components,)
            Eigenvalues corresponding to the `old_eigenvectors`.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            - new_eigenvectors : np.ndarray of shape (n_new_samples, n_components)
                Approximated eigenvectors for the new data points.
            - new_eigenvalues : np.ndarray of shape (n_components,)
                The original eigenvalues (unchanged by Nystrom approximation).
        """
        # Compute the approximate eigenvectors for the new points
        new_eigenvectors = (K_mix @ old_eigenvectors) / old_eigenvalues[np.newaxis, :]
        # Nystrom approximation uses the original eigenvalues
        new_eigenvalues = old_eigenvalues

        return new_eigenvectors, new_eigenvalues


    def _get_K_alpha_approx(self, X_new: np.ndarray) -> np.ndarray:
        """
        Computes the approximated alpha-normalized kernel between new and old data.

        This is needed for the Nystrom extension when alpha != 0.
        Calculates K(X_new, X_old), computes the degree vector for X_new based
        on this mixed kernel (d_mix), and then performs alpha-normalization using
        d_mix and the degree vector from the original data (d_old = self.d_K).

        K_alpha_mix[i, j] = K_mix[i, j] / (d_mix[i]^alpha * d_old[j]^alpha)

        Parameters
        ----------
        X_new : np.ndarray of shape (n_new_samples, n_features)
            The new input samples for which to compute the mixed kernel.

        Returns
        -------
        np.ndarray
            The alpha-normalized mixed kernel matrix K_alpha_mix of shape
            (n_new_samples, n_old_samples).
        """
        # Compute the kernel between new data and original fitted data
        K_mix = self._rbf_kernel(X_new, self.X, gamma=self.gamma)
        # Degree vector of original data (computed during fit)
        d_old = self.d_K
        # Compute degree vector for new data based on connections to old data
        d_mix = self._degree_vector(K_mix)
        # Perform alpha-normalization using mixed and old degrees
        K_alpha_mix = self._normalize_by_degree(K_mix, d_mix, d_old, self.alpha)

        return K_alpha_mix
    

    def _get_A_approx(self, X_new: np.ndarray) -> np.ndarray:
        """
        Computes the approximated matrix A_mix for Nystrom extension.

        This matrix relates the new data points to the old data points in the
        symmetric space where the original spectral decomposition was performed.
        It requires computing the mixed alpha-normalized kernel (K_alpha_mix)
        and then normalizing it appropriately using degrees derived from
        alpha-normalized kernels (d_W).

        Parameters
        ----------
        X_new : np.ndarray of shape (n_new_samples, n_features)
            The new input samples.

        Returns
        -------
        np.ndarray
            The approximated matrix A_mix of shape (n_new_samples, n_old_samples)
            used for the Nystrom extension.
        """
        if self.alpha != 0:
            # Compute alpha-normalized mixed kernel if alpha is non-zero
            K_alpha_mix = self._get_K_alpha_approx(X_new)
        else:
            # If alpha is 0, W = K, so K_alpha_mix = K_mix
            K_alpha_mix = self._rbf_kernel(X_new, self.X, gamma=self.gamma)

        # Degree vector from alpha-normalized kernel on original data (d_W)
        d_alpha_old = self.d_W
        # Degree vector for new points based on K_alpha_mix
        d_alpha_mix = self._degree_vector(K_alpha_mix) # Sum rows of K_alpha_mix

        # Symmetrically normalize K_alpha_mix using sqrt of degrees to get A_mix
        A_mix = self._normalize_by_degree(K_alpha_mix, d_alpha_mix, d_alpha_old, 0.5)

        return A_mix


    def transform(self, X_new: np.ndarray) -> np.ndarray:
        """
        Transforms new data points using the fitted Diffusion Maps model.

        Applies the Nystrom extension to approximate the eigenvectors for X_new
        based on the eigenvectors and eigenvalues learned during `fit`. It then
        computes the diffusion map coordinates for X_new.

        Parameters
        ----------
        X_new : np.ndarray of shape (n_samples, n_features)
            The new input samples to transform.

        Returns
        -------
        np.ndarray
            The diffusion map embedding of the new data, shape
            (n_samples, n_components).
        """
        check_is_fitted(self) # Ensure fit has been called
        X_new_flat = X_new.reshape((X_new.shape[0], -1))
        # Select the eigenvalues/vectors needed for Nystrom and embedding.
        lambdas_red = self.lambdas[:self.n_components + 1]
        phis_red = self.phis[:, :self.n_components + 1]
        # Compute the mixed matrix A_mix needed for Nystrom
        A_mix = self._get_A_approx(X_new_flat)
        # Apply Nystrom extension to get approximate eigenvectors for new data
        new_phis, new_lambdas = self._nystrom_extension(A_mix, phis_red, lambdas_red)
        # Approximate the stationary distribution for the new points
        new_pi = np.power(new_phis[:, 0], 2)
        # Compute the embedding coordinates
        X_new_red = self._get_embedding(new_phis[:, 1:], new_lambdas[1:], new_pi)

        return X_new_red


    # --- Optional Utility Methods (can be made public if desired) ---


    @staticmethod
    def transition_probabilities(W: np.ndarray) -> np.ndarray:
        """
        Computes the row-stochastic transition matrix P from an affinity matrix W.

        P_ij = W_ij / sum_k W_ik

        Parameters
        ----------
        W : np.ndarray of shape (n_samples, n_samples)
            Affinity matrix (e.g., the alpha-normalized kernel).

        Returns
        -------
        np.ndarray
            The row-stochastic transition matrix P.
        """
        d = np.sum(W, axis=1)
        P = W / d[:, np.newaxis]

        return P
    

    @staticmethod
    def diffusion_distances(P: np.ndarray, pi: np.ndarray) -> np.ndarray:
        """
        Computes the pairwise diffusion distances between rows (states) of P.

        The squared diffusion distance D(i, j)^2 between state i and state j
        at time t=1 is defined as the weighted L2 distance between the rows
        of the transition matrix P, where the weighting is given by the
        inverse of the stationary distribution pi:

        D(i, j)^2 = sum_k ( (P_ik - P_jk)^2 / pi_k )

        Parameters
        ----------
        P : np.ndarray of shape (n_samples, n_samples)
            The transition probability matrix (or its t-step version P^t).
        pi : np.ndarray of shape (n_samples,)
            The stationary distribution of the Markov chain defined by P.

        Returns
        -------
        np.ndarray
            Matrix of pairwise diffusion distances, shape (n_samples, n_samples).
        """
        metric = lambda P_i, P_j: np.sqrt(np.sum(np.power(P_i - P_j, 2) / pi))
        D = pairwise_distances(P, metric=metric, n_jobs=-1)

        return D
