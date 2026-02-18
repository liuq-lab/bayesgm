import numpy as np
import scipy.linalg as linalg
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings


def get_ADRF(x_values=None, x_min=None, x_max=None, nb_intervals=None, dataset='Imbens'):
    """
    Compute the values of the Average Dose-Response Function (ADRF).
    
    Parameters
    ----------
    x_values : list or np.ndarray, optional
        A list or array of values at which to evaluate the ADRF.
        If provided, overrides x_min, x_max, and nb_intervals.
    x_min : float, optional
        The minimum value of the range (used when x_values is not provided).
    x_max : float, optional
        The maximum value of the range (used when x_values is not provided).
    nb_intervals : int, optional
        The number of intervals in the range (used when x_values is not provided).
    dataset : str, optional
        The dataset name (default: 'Imbens'). Must be one of {'Imbens', 'Sun', 'Lee'}.
    
    Returns
    -------
    true_values : np.ndarray
        The computed ADRF values.
    
    Notes
    -----
    - Either `x_values` or (`x_min`, `x_max`, `nb_intervals`) must be provided.
    - Supported datasets:
        - 'Imbens': ADRF = x + 2 / (1 + x)^3
        - 'Sun': ADRF = x - 1/2 + exp(-0.5) + 1
        - 'Lee': ADRF = 1.2 * x + x^3
    """
    # Validate dataset name
    valid_datasets = {'Imbens', 'Sun', 'Lee'}
    if dataset not in valid_datasets:
        raise ValueError(f"`dataset` must be one of {valid_datasets}, but got '{dataset}'.")

    # Input validation for x_values or range parameters
    if x_values is not None:
        if not isinstance(x_values, (list, np.ndarray)):
            raise ValueError("`x_values` must be a list or numpy array.")
        x_values = np.array(x_values, dtype='float32')
    elif x_min is not None and x_max is not None and nb_intervals is not None:
        if x_min >= x_max:
            raise ValueError("`x_min` must be less than `x_max`.")
        if nb_intervals <= 0:
            raise ValueError("`nb_intervals` must be a positive integer.")
        x_values = np.linspace(x_min, x_max, nb_intervals, dtype='float32')
    else:
        raise ValueError("Either `x_values` or (`x_min`, `x_max`, `nb_intervals`) must be provided.")
    
    # Compute ADRF values based on the selected dataset
    if dataset == 'Imbens':
        true_values = x_values + 2 / (1 + x_values)**3
    elif dataset == 'Sun':
        true_values = x_values - 0.5 + np.exp(-0.5) + 1
    elif dataset == 'Lee':
        true_values = 1.2 * x_values + x_values**3

    return true_values


def slice_y(y, n_slices=10):
    """Determine non-overlapping slices based on the target variable, y.

    Parameters
    ----------
    y : array_like, shape (n_samples,)
        The target values (class labels in classification, real numbers
        in regression).

    n_slices : int (default=10)
        The number of slices used when calculating the inverse regression
        curve. Truncated to at most the number of unique values of ``y``.

    Returns
    -------
    slice_indicator : ndarray, shape (n_samples,)
        Index of the slice (from 0 to n_slices) that contains this
        observation.
    slice_counts :  ndarray, shape (n_slices,)
        The number of counts in each slice.
    """
    unique_y_vals, counts = np.unique(y, return_counts=True)
    cumsum_y = np.cumsum(counts)

    # `n_slices` must be less-than or equal to the number of unique values
    # of `y`.
    n_y_values = unique_y_vals.shape[0]
    if n_y_values == 1:
        raise ValueError("The target only has one unique y value. It does "
                         "not make sense to fit SIR or SAVE in this case.")
    elif n_slices >= n_y_values:
        if n_slices > n_y_values:
            warnings.warn(
                "n_slices greater than the number of unique y values. "
                "Setting n_slices equal to {0}.".format(counts.shape[0]))
        # each y value gets its own slice. usually the case for classification
        slice_partition = np.hstack((0, cumsum_y))
    else:
        # attempt to put this many observations in each slice.
        # not always possible since we need to group common y values together
        n_obs = np.floor(y.shape[0] / n_slices)

        # Loop through the unique y value sums and group
        # slices together with the goal of 2 <= # in slice <= n_obs
        # Find index in y unique where slice begins and ends
        n_samples_seen = 0
        slice_partition = [0]  # index in y of start of a new slice
        while n_samples_seen < y.shape[0] - 2:
            slice_start = np.where(cumsum_y >= n_samples_seen + n_obs)[0]
            if slice_start.shape[0] == 0:  # this means we've reached the end
                slice_start = cumsum_y.shape[0] - 1
            else:
                slice_start = slice_start[0]

            n_samples_seen = cumsum_y[slice_start]
            slice_partition.append(n_samples_seen)

    # turn partitions into an indicator
    slice_indicator = np.ones(y.shape[0], dtype='int64')
    for j, (start_idx, end_idx) in enumerate(
            zip(slice_partition, slice_partition[1:])):

        # this just puts any remaining observations in the last slice
        if j == len(slice_partition) - 2:
            slice_indicator[start_idx:] = j
        else:
            slice_indicator[start_idx:end_idx] = j

    slice_counts = np.bincount(slice_indicator)
    return slice_indicator, slice_counts

def get_SDR_dim(X, y, n_slices = 10, ratio = 0.8):
    '''
    Calculate the SDR dimension of the X.
    Input:
        X: array-like, shape (n_samples, n_features)
        y: array-like, shape (n_samples, 1)
        n_slices: int, the number of slices used when calculating the inverse regression curve.
    Output:
        dim: int, the SDR dimension of X.
    '''
    if len(y.shape) == 2:
        assert y.shape[1] == 1, "The shape of y should be (n_samples, 1)."
        y = np.squeeze(y)
    n_samples, n_features = X.shape

    # normalize the data
    X = X - np.mean(X, axis=0)
    Q, R = linalg.qr(X, mode='economic')
    Z = np.sqrt(n_samples) * Q
    Z = Z[np.argsort(y), :]

    # determine slice indices and counts per slice
    slices, counts = slice_y(y, n_slices)

    # Sums an array by groups. Groups are assumed to be contiguous by row.
    inv_idx = np.concatenate(([0], np.diff(slices).nonzero()[0] + 1))
    Z_sum = np.add.reduceat(Z, inv_idx)
    # means in each slice (sqrt factor takes care of the weighting)
    Z_means = Z_sum / np.sqrt(counts.reshape(-1, 1))
    
    M = np.dot(Z_means.T, Z_means) / n_samples
    # eigen-decomposition of slice matrix
    evals, evecs = linalg.eigh(M)
    evals = evals[::-1]
    #n_directions = np.argmax(np.abs(np.diff(evals))) + 1
    total_sum = np.sum(evals)
    cumulative_sum = np.cumsum(evals)
    threshold_index = np.argmax(cumulative_sum >= ratio * total_sum)
    n_directions = threshold_index + 1
    return n_directions

def estimate_latent_dims(x, y, v, v_ratio=0.7, z0_dim=3, max_total_dim=64, min_z3_dim=3):
    """Estimate the latent-dimension split for CausalBGM.

    Uses Sliced Inverse Regression (SIR) and PCA to automatically choose
    dimensions ``[z0, z1, z2, z3]`` for the four latent sub-vectors.

    Parameters
    ----------
    x : np.ndarray
        Treatment variable with shape ``(n, 1)``.
    y : np.ndarray
        Outcome variable with shape ``(n, 1)``.
    v : np.ndarray
        Covariates with shape ``(n, v_dim)``.
    v_ratio : float, default=0.7
        Cumulative PCA variance ratio used to determine total latent
        dimension.
    z0_dim : int, default=3
        Fixed dimension for the confounding sub-vector :math:`Z_0`.
    max_total_dim : int, default=64
        Upper bound on the total latent dimension.
    min_z3_dim : int, default=3
        Minimum dimension for the residual sub-vector :math:`Z_3`.

    Returns
    -------
    list of int
        A list ``[z0_dim, z1_dim, z2_dim, z3_dim]``.
    """
    v = StandardScaler().fit_transform(v)
    y = StandardScaler().fit_transform(y)
    z1_dim = get_SDR_dim(v, y, n_slices=10, ratio=0.8)
    z2_dim = get_SDR_dim(v, x, n_slices=10, ratio=0.8)
    pca = PCA().fit(v)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    threshold_index = np.argmax(cumulative_variance >= v_ratio)
    total_z_dim = threshold_index + 1
    total_z_dim = min(max_total_dim, total_z_dim)
    z3_dim = total_z_dim - z0_dim - z1_dim - z2_dim
    if z3_dim<=min_z3_dim:
        z3_dim = min_z3_dim
    return [z0_dim, z1_dim, z2_dim, z3_dim]

def mnist_mask_indices(
    shape=(28, 28),
    mode="hole",
    center=(14, 14),
    num_holes=1,
    hole_size=3,
    orientation="horizontal",
    stripe_width=4,
    stripe_pos=14,  
    seed=None,
):
    """
    Create pixel masks on a 2D grid and return flattened index arrays.
    
    Parameters
    ----------
    shape : (H, W)
        Image height and width.
    mode : str
        One of:
          - 'hole' : mask a hole with size `hole_size`×`hole_size`.
          - 'edge_stripe'  : mask a stripe along the edges; choose `side` and `stripe_width`.
          - 'upper_half'   : mask rows [0 : H//2)
          - 'lower_half'   : mask rows [H//2 : H)
          - 'left_half'    : mask cols [0 : W//2)
          - 'right_half'   : mask cols [W//2 : W)
    center : tuple (row, col)
        Center of the hole when mode='hole'.
    hole_size : int
        Side length of each square hole (odd is best).
    orientation : str
        Which edges to mask for mode='edge_stripe'.
        'horizontal' masks horizontal strip; 'vertical' masks vertical strip.
    stripe_width : int
        Stripe thickness in pixels (for edge stripes).
    stripe_pos : int
        Position of the stripe when mode='edge_stripe'.
    seed : int or None
        RNG seed for reproducibility.

    Returns
    -------
    ind_x1 : np.ndarray (1D, dtype=int)
        Flattened indices of **unmasked** pixels.
    ind_x2 : np.ndarray (1D, dtype=int)
        Flattened indices of **masked** pixels.
    """
    H, W = shape
    mask = np.zeros((H, W), dtype=bool)  # False=keep (unmasked), True=mask out

    if mode == "holes":
        rng = np.random.default_rng(seed)
        r = hole_size
        # ensure holes stay inside bounds
        r2 = r // 2
        valid_rows = np.arange(r2, H - (r - r2 - 1))
        valid_cols = np.arange(r2, W - (r - r2 - 1))
        if center is None:
            center = (rng.choice(valid_rows), rng.choice(valid_cols))
        (cy, cx) = center
        y0, y1 = cy - r2, cy - r2 + r
        x0, x1 = cx - r2, cx - r2 + r
        mask[y0:y1, x0:x1] = True

    elif mode == "edge_stripe":
        w = int(stripe_width)
        start_idx = stripe_pos - w // 2
        end_idx = stripe_pos - w // 2 + w
        if orientation == 'horizontal':
            mask[start_idx:end_idx, :] = True
        elif orientation == 'vertical':
            mask[: start_idx:end_idx] = True
        else:
            raise ValueError(f"Unknown orientation: {orientation}")

    elif mode == "upper_half":
        mask[: H // 2, :] = True
    elif mode == "lower_half":
        mask[H // 2 :, :] = True
    elif mode == "left_half":
        mask[:, : W // 2] = True
    elif mode == "right_half":
        mask[:, W // 2 :] = True
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Flattened index outputs
    ind_x2 = np.flatnonzero(mask)         # masked
    ind_x1 = np.flatnonzero(~mask)        # unmasked
    return ind_x1, ind_x2
