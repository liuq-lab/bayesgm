import numpy as np
from sklearn.datasets import make_low_rank_matrix


def simulate_regression(n_samples, n_features, n_targets, effective_rank=None, variance=None, random_state=123):
    """Simulate a linear regression dataset with optional low-rank design.

    Generates :math:`X` (optionally low-rank) and
    :math:`Y = X_{\\text{aug}} \\beta + \\varepsilon` where
    :math:`X_{\\text{aug}}` includes an intercept column.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    n_features : int
        Number of features (columns of :math:`X`).
    n_targets : int
        Number of target (response) variables.
    effective_rank : int or None, optional
        If provided, the design matrix :math:`X` is generated as a low-rank
        matrix with this effective rank.  Otherwise :math:`X` is i.i.d.
        standard normal.
    variance : np.ndarray or None, optional
        Per-sample noise variance.  If ``None``, defaults to
        ``0.01 * mean(X^2)`` per sample.
    random_state : int, default=123
        Random seed for reproducibility.

    Returns
    -------
    X : np.ndarray
        Feature matrix with shape ``(n_samples, n_features)``.
    Y : np.ndarray
        Response matrix with shape ``(n_samples, n_targets)``.
    """
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
    """Simulate a heteroskedastic regression dataset.

    The noise standard deviation depends on the second feature :math:`X_2`:
    ``sigma = 0.5 + 0.5 * sin(2 * pi * X_2)`` (clipped to [0.1, 2.0]
    outside :math:`|X_2| > 2`).

    Parameters
    ----------
    n : int, default=1000
        Number of samples.
    d : int, default=5
        Number of features.
    seed : int, default=42
        Random seed.

    Returns
    -------
    X : np.ndarray
        Feature matrix with shape ``(n, d)``.
    Y : np.ndarray
        Response vector with shape ``(n,)``.
    sigma : np.ndarray
        True noise standard deviation with shape ``(n,)``.
    """
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
    """Simulate a latent-factor heteroskedastic regression dataset.

    Observed features :math:`X` are a noisy low-rank projection of a
    :math:`k`-dimensional latent variable :math:`Z`.  The response
    :math:`Y` depends nonlinearly on :math:`Z` with heteroskedastic noise.

    Parameters
    ----------
    n : int, default=20000
        Number of samples.
    k : int, default=3
        Dimension of the latent variable :math:`Z`.
    d : int, default=19
        Number of observed features.
    seed : int, default=42
        Random seed.

    Returns
    -------
    X : np.ndarray
        Observed feature matrix with shape ``(n, d)``.
    Y : np.ndarray
        Response vector with shape ``(n,)``.
    """
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

def demand_design_h(time):
    """Demand-design seasonality function used in DFIV/DeepIV benchmarks."""
    time = np.asarray(time, dtype=np.float64)
    return 2.0 * (
        ((time - 5.0) ** 4) / 600.0
        + np.exp(-4.0 * (time - 5.0) ** 2)
        + time / 10.0
        - 2.0
    )


def demand_design_structural_function(price, time, customer_group):
    """Ground-truth structural function for the demand-design IV benchmark."""
    price = np.asarray(price, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)
    customer_group = np.asarray(customer_group, dtype=np.float64)
    return 100.0 + (10.0 + price) * customer_group * demand_design_h(time) - 2.0 * price


def simulate_demand_design_iv(
    n_samples=5000,
    rho=0.5,
    seed=0,
):
    """Simulate the low-dimensional demand-design IV benchmark from DFIV.

    The observed tuple is ``(X, Y, V, W)`` with

    - ``X``: price
    - ``Y``: demand
    - ``V``: observed confounders ``(T, S)``
    - ``W``: instrument ``C``

    Returns a dictionary that also includes the noiseless structural response.
    """
    rng = np.random.default_rng(seed)
    customer_group = rng.integers(1, 8, size=n_samples)
    time = rng.uniform(0.0, 10.0, size=n_samples)
    instrument = rng.normal(0.0, 1.0, size=n_samples)
    latent_confounder = rng.normal(0.0, 1.0, size=n_samples)

    noise_scale = np.sqrt(max(1.0 - float(rho) ** 2, 1e-6))
    epsilon = rho * latent_confounder + rng.normal(0.0, noise_scale, size=n_samples)

    price = 25.0 + (instrument + 3.0) * demand_design_h(time) + latent_confounder
    structural_mean = demand_design_structural_function(price, time, customer_group)
    demand = structural_mean + epsilon

    covariates = np.column_stack([time, customer_group]).astype(np.float32)
    # extra_covariate_dim = int(extra_covariate_dim)
    # if extra_covariate_dim > 0:
    #     nuisance_covariates = rng.normal(
    #         0.0, 1.0, size=(n_samples, extra_covariate_dim)
    #     ).astype(np.float32)
    #     covariates = np.concatenate([covariates, nuisance_covariates], axis=1)
    return {
        "x": price.reshape(-1, 1).astype(np.float32),
        "y": demand.reshape(-1, 1).astype(np.float32),
        "v": covariates,
        "w": instrument.reshape(-1, 1).astype(np.float32),
        "y_struct": structural_mean.reshape(-1, 1).astype(np.float32),
        "time": time.reshape(-1, 1).astype(np.float32),
        "customer_group": customer_group.reshape(-1, 1).astype(np.float32),
    }


def make_demand_design_grid(
    price_points=20,
    time_points=20,
    #extra_covariate_dim=0,
    #noise_repeats=1,
    #noise_seed=0,
):
    """Construct the demand-design evaluation grid with optional nuisance repeats."""
    prices = np.linspace(10.0, 25.0, num=price_points, dtype=np.float64)
    times = np.linspace(0.0, 10.0, num=time_points, dtype=np.float64)
    customer_groups = np.arange(1.0, 8.0, dtype=np.float64)

    price_grid, time_grid, customer_grid = np.meshgrid(
        prices, times, customer_groups, indexing="ij"
    )
    x = price_grid.reshape(-1, 1).astype(np.float32)
    v = np.column_stack(
        [time_grid.reshape(-1), customer_grid.reshape(-1)]
    ).astype(np.float32)
    y_struct = demand_design_structural_function(
        price_grid.reshape(-1),
        time_grid.reshape(-1),
        customer_grid.reshape(-1),
    ).reshape(-1, 1)

    # extra_covariate_dim = int(extra_covariate_dim)
    # noise_repeats = int(noise_repeats)
    # if noise_repeats < 1:
    #     raise ValueError("noise_repeats must be >= 1")

    # n_base = x_base.shape[0]
    # x = np.repeat(x_base, noise_repeats, axis=0)
    # v = np.repeat(v_base, noise_repeats, axis=0)
    # y_struct = np.repeat(y_struct_base, noise_repeats, axis=0)
    # group_index = np.repeat(np.arange(n_base, dtype=np.int64), noise_repeats)

    # if extra_covariate_dim > 0:
    #     rng = np.random.default_rng(noise_seed)
    #     nuisance_covariates = rng.normal(
    #         0.0, 1.0, size=(x.shape[0], extra_covariate_dim)
    #     ).astype(np.float32)
    #     v = np.concatenate([v, nuisance_covariates], axis=1)

    return {
        "x": x,
        "v": v,
        "y_struct": y_struct.astype(np.float32),
        "time": time_grid.reshape(-1, 1).astype(np.float32),
        "customer_group": customer_grid.reshape(-1, 1).astype(np.float32),
        # "time": np.repeat(
        #     time_grid.reshape(-1, 1).astype(np.float32), noise_repeats, axis=0
        # ),
        # "customer_group": np.repeat(
        #     customer_grid.reshape(-1, 1).astype(np.float32), noise_repeats, axis=0
        # ),
        # "group_index": group_index,
        # "noise_repeats": noise_repeats,
    }
