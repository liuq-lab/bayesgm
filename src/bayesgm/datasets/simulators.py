import numpy as np
from sklearn.datasets import make_low_rank_matrix


def simulate_regression(n_samples, n_features, n_targets, effective_rank=None, variance=None, random_state=123):
    np.random.seed(random_state)
    if effective_rank is None:
        X = np.random.normal(size=(n_samples, n_features))
    else:
        X = 100*make_low_rank_matrix(n_samples=n_samples, n_features=n_features, effective_rank=effective_rank, random_state=random_state)

    X_aug = np.c_[np.ones(n_samples), X]  # n x (p+1) matrix
    beta = 0.1 * np.random.uniform(low=0.0, high=1.0, size=(1+n_features, n_targets))    # Coefficients for the mean (includes intercept)
    mu = np.dot(X_aug, beta)
    if variance is None:
        variance = 0.01*np.mean(X**2, axis=1)
    variance = np.tile(variance,(n_targets,1)).T
    Y = np.random.normal(loc=mu, scale=np.sqrt(variance))
    return X, Y


def simulate_low_rank_data(n_samples=10000, z_dim=2, x_dim=4, rank=2, sigma_z=False, random_state=123):
    """
    Simulate low-rank observed data with latent variables.

    The generator is:
    - ``Z ~ N(0, I)``
    - ``X | Z ~ N(mu(Z), Sigma(Z))``

    Parameters
    ----------
    n_samples : int, default=10000
        Number of samples.
    z_dim : int, default=2
        Dimension of latent variable ``Z``.
    x_dim : int, default=4
        Dimension of observed variable ``X``.
    rank : int, default=2
        Rank for the low-rank covariance component.
    sigma_z : bool, default=False
        If ``True``, covariance ``Sigma`` depends on ``Z`` (scaled by ``z[0]``).
        If ``False``, ``Sigma`` is constant across samples.
    random_state : int, default=123
        Random seed for reproducibility.

    Returns
    -------
    X : np.ndarray
        Observed data with shape ``(n_samples, x_dim)``.
    Z : np.ndarray
        Latent variables with shape ``(n_samples, z_dim)``.
    """
    np.random.seed(random_state)

    # Generate Z ~ N(0, I)
    Z = np.random.randn(n_samples, z_dim).astype(np.float32) 

    # Set fixed parameters for the simulation
    # Randomly generate A and b, or choose them to illustrate particular structure.
    A = np.array([[ 1.0, -0.5],
                  [ 0.3,  0.8],
                  [-0.7,  0.2],
                  [ 0.5,  1.0]])
    b = np.array([0.0, 0.5, 1.0, 2.0])

    # Compute the mean for X given Z: μ(Z) = A Z + b
    mu = Z.dot(A.T) + b  # Shape: (n_samples, x_dim)

    W = np.array([[ 0.25, 0.],
                  [ 0.25, 0.],
                  [ 0., 0.25],
                  [ 0., 0.25]])
    diag_values = np.array([0.1, 0.1, 0.2, 0.2])
    D = np.diag(diag_values)
    
    X = np.zeros((n_samples, x_dim), dtype=np.float32)
    
    for i in range(n_samples):
        if sigma_z:
            scale_factor = Z[i, 0]
            W_scaled = W * scale_factor  # Element-wise multiplication
            Sigma = D * scale_factor**2 + np.dot(W_scaled, W_scaled.T)
        else:
            # Use constant covariance: Σ = D + W W^T
            Sigma = D + np.dot(W, W.T)

        # Sample X_i ~ N(μ_i, Σ)
        X[i] = np.random.multivariate_normal(mean=mu[i], cov=Sigma)
    
    return X, Z

def simulate_heteroskedastic_data(n=1000, d=5, seed=42):
    np.random.seed(seed)
    X = np.random.randn(n, d)
    X1 = X[:, 0]
    X2 = X[:, 1]

    # Define heteroskedastic noise
    sigma = np.where(
        X2 < -2, 0.1,
        np.where(X2 > 2, 2.0, 0.5 + 0.5 * np.sin(2 * np.pi * X2))
    )

    epsilon = np.random.randn(n) * sigma
    Y = X1 + epsilon
    return X, Y, sigma

def simulate_z_hetero(n=20000, k=3, d=20-1, seed=42):
    np.random.seed(seed)

    Z = np.random.randn(n, k)

    # Low-rank X from latent Z
    A = np.random.randn(d, k)
    X = 0.2*Z @ A.T + 0.1 * np.random.randn(n, d)

    # Define nonlinear mean and heteroskedastic std
    w = np.random.randn(k)
    u = np.random.randn(k)

    mean_Y = np.sin(Z @ w)
    std_Y = 0.1 + 0.5 * 1 / (1 + np.exp(-(Z @ u)))

    Y = mean_Y + std_Y * np.random.randn(n)
    return X, Y
