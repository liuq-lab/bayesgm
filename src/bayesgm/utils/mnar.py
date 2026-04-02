"""Utilities for MNAR imputation workflows.

The utilities in this module are model-free. They cover:

- mask handling and reconstruction
- observed-only standardization
- RMSE and posterior interval helpers
- baseline imputers and benchmark tables for MNAR experiments
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer


def _to_float32_matrix(data: np.ndarray) -> np.ndarray:
    """Convert an input array to a 2-D float32 matrix."""
    array = np.asarray(data, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D array, received shape {array.shape}.")
    return array


def _is_tensorflow_value(value) -> bool:
    """Return ``True`` when the value is a TensorFlow tensor or variable."""

    module_name = type(value).__module__
    return module_name.startswith("tensorflow")


def validate_mask(mask: np.ndarray, expected_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Validate and normalise an observation mask.

    The convention is fixed throughout the MNAR implementation:
    ``mask == 1`` means the entry is observed and ``mask == 0`` means missing.
    """

    mask_array = np.asarray(mask, dtype=np.float32)
    if mask_array.ndim != 2:
        raise ValueError(f"Expected a 2-D mask, received shape {mask_array.shape}.")
    if expected_shape is not None and mask_array.shape != expected_shape:
        raise ValueError(
            f"Mask shape {mask_array.shape} does not match expected shape {expected_shape}."
        )
    if not np.all((mask_array == 0.0) | (mask_array == 1.0)):
        raise ValueError("Mask entries must be binary with values in {0, 1}.")
    return mask_array.astype(np.float32, copy=False)


def infer_mask(data: np.ndarray) -> np.ndarray:
    """Infer an observation mask from a matrix that uses ``NaN`` for missing values."""

    array = _to_float32_matrix(data)
    return (~np.isnan(array)).astype(np.float32)


def prepare_masked_data(
    data: np.ndarray,
    mask: Optional[np.ndarray] = None,
    initialization: Optional[str] = None,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a matrix with the resolved mask.

    By default, missing entries are zero-filled so observed values stay isolated
    in the returned matrix. When ``initialization`` is ``"mean"`` or
    ``"missforest"``, the returned matrix is fully imputed instead.
    """

    array = _to_float32_matrix(data)
    if mask is None:
        resolved_mask = infer_mask(array)
    else:
        resolved_mask = validate_mask(mask, expected_shape=array.shape)
    x_obs = np.where(resolved_mask == 1.0, np.nan_to_num(array, nan=0.0), 0.0).astype(np.float32)

    if initialization is None or initialization == "zero":
        return x_obs, resolved_mask
    if initialization == "mean":
        return mean_impute(x_obs, resolved_mask), resolved_mask
    if initialization == "missforest":
        return missforest_imputation_baseline(
            x_obs,
            resolved_mask,
            random_state=random_state,
        ), resolved_mask
    raise ValueError(f"Unsupported initialization strategy: {initialization}")


def reconstruct_from_mask(x_obs, mask, x_mis):
    """Reconstruct the full matrix while keeping observed entries fixed.

    This helper accepts either NumPy arrays or TensorFlow tensors/variables and
    returns the matching type.
    """

    if any(_is_tensorflow_value(value) for value in (x_obs, mask, x_mis)):
        import tensorflow as tf

        x_obs_tensor = tf.cast(x_obs, tf.float32)
        mask_tensor = tf.cast(mask, tf.float32)
        x_mis_tensor = tf.cast(x_mis, tf.float32)

        tf.debugging.assert_rank(x_obs_tensor, 2, message="x_obs must be a 2-D tensor.")
        tf.debugging.assert_rank(mask_tensor, 2, message="mask must be a 2-D tensor.")
        tf.debugging.assert_rank(x_mis_tensor, 2, message="x_mis must be a 2-D tensor.")
        tf.debugging.assert_equal(tf.shape(x_obs_tensor), tf.shape(mask_tensor))
        tf.debugging.assert_equal(tf.shape(x_obs_tensor), tf.shape(x_mis_tensor))
        tf.debugging.assert_equal(mask_tensor, tf.round(mask_tensor), message="mask must be binary.")
        tf.debugging.assert_greater_equal(mask_tensor, 0.0)
        tf.debugging.assert_less_equal(mask_tensor, 1.0)
        return mask_tensor * x_obs_tensor + (1.0 - mask_tensor) * x_mis_tensor

    x_obs_array = _to_float32_matrix(x_obs)
    x_mis_array = _to_float32_matrix(x_mis)
    resolved_mask = validate_mask(mask, expected_shape=x_obs_array.shape)
    if x_mis_array.shape != x_obs_array.shape:
        raise ValueError(
            f"x_mis shape {x_mis_array.shape} does not match x_obs shape {x_obs_array.shape}."
        )
    return resolved_mask * x_obs_array + (1.0 - resolved_mask) * x_mis_array


def apply_mask(data: np.ndarray, mask: np.ndarray, missing_value: float = np.nan) -> np.ndarray:
    """Apply a mask to fully observed data and write ``missing_value`` at missing entries."""

    array = _to_float32_matrix(data)
    resolved_mask = validate_mask(mask, expected_shape=array.shape)
    masked = np.array(array, copy=True)
    masked[resolved_mask == 0.0] = missing_value
    return masked.astype(np.float32)


def observed_feature_index_list(mask: np.ndarray) -> List[List[int]]:
    """Convert a binary observation mask into per-row observed feature indices."""

    resolved_mask = validate_mask(mask)
    return [np.flatnonzero(row).astype(np.int32).tolist() for row in resolved_mask]


def mean_impute(data: np.ndarray, mask: np.ndarray, fill_values: Optional[np.ndarray] = None) -> np.ndarray:
    """Mean-impute missing entries while leaving observed entries untouched."""

    x_obs, resolved_mask = prepare_masked_data(data, mask)
    if fill_values is None:
        counts = resolved_mask.sum(axis=0)
        safe_counts = np.maximum(counts, 1.0)
        fill_values = (x_obs * resolved_mask).sum(axis=0) / safe_counts
        fill_values = np.where(counts > 0.0, fill_values, 0.0)
    fill_values = np.asarray(fill_values, dtype=np.float32)
    if fill_values.ndim != 1 or fill_values.shape[0] != x_obs.shape[1]:
        raise ValueError("fill_values must be a 1-D vector with length equal to the feature dimension.")
    filled = np.where(resolved_mask == 1.0, x_obs, fill_values[None, :])
    return filled.astype(np.float32)


@dataclass
class ObservedStandardizer:
    """Feature-wise standardizer fit on observed entries only."""

    mean_: Optional[np.ndarray] = None
    scale_: Optional[np.ndarray] = None
    min_scale: float = 1e-6

    def fit(self, x_obs: np.ndarray, mask: np.ndarray) -> "ObservedStandardizer":
        """Estimate means and scales from observed training entries only."""

        observed, resolved_mask = prepare_masked_data(x_obs, mask)
        counts = resolved_mask.sum(axis=0)
        safe_counts = np.maximum(counts, 1.0)

        mean = (observed * resolved_mask).sum(axis=0) / safe_counts
        centered = (observed - mean[None, :]) * resolved_mask
        variance = (centered ** 2).sum(axis=0) / safe_counts
        scale = np.sqrt(np.maximum(variance, self.min_scale ** 2))

        mean = np.where(counts > 0.0, mean, 0.0)
        scale = np.where(counts > 0.0, scale, 1.0)

        self.mean_ = mean.astype(np.float32)
        self.scale_ = scale.astype(np.float32)
        return self

    def _check_is_fitted(self) -> None:
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("ObservedStandardizer must be fit before calling transform methods.")

    def transform_full(self, data: np.ndarray) -> np.ndarray:
        """Standardize a fully specified matrix using stored parameters."""

        self._check_is_fitted()
        array = _to_float32_matrix(data)
        return ((array - self.mean_[None, :]) / self.scale_[None, :]).astype(np.float32)

    def transform_observed(self, x_obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Standardize observed entries and keep missing entries equal to zero."""

        observed, resolved_mask = prepare_masked_data(x_obs, mask)
        transformed = self.transform_full(observed)
        transformed[resolved_mask == 0.0] = 0.0
        return transformed.astype(np.float32)

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Undo feature standardization."""

        self._check_is_fitted()
        array = _to_float32_matrix(data)
        return (array * self.scale_[None, :] + self.mean_[None, :]).astype(np.float32)

    def to_dict(self) -> dict:
        """Serialize standardization parameters into Python lists."""

        self._check_is_fitted()
        return {
            "mean": self.mean_.tolist(),
            "scale": self.scale_.tolist(),
            "min_scale": float(self.min_scale),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ObservedStandardizer":
        """Restore a standardizer from a serialized dictionary."""

        standardizer = cls(min_scale=float(payload.get("min_scale", 1e-6)))
        standardizer.mean_ = np.asarray(payload["mean"], dtype=np.float32)
        standardizer.scale_ = np.asarray(payload["scale"], dtype=np.float32)
        return standardizer


def rmse_on_missing_entries(x_true: np.ndarray, x_imputed: np.ndarray, mask: np.ndarray) -> float:
    """Compute RMSE on truly missing entries only."""

    truth = _to_float32_matrix(x_true)
    imputed = _to_float32_matrix(x_imputed)
    resolved_mask = validate_mask(mask, expected_shape=truth.shape)
    if imputed.shape != truth.shape:
        raise ValueError(f"x_imputed shape {imputed.shape} does not match x_true shape {truth.shape}.")
    missing = resolved_mask == 0.0
    if not np.any(missing):
        return 0.0
    mse = np.mean((truth[missing] - imputed[missing]) ** 2, dtype=np.float64)
    return float(np.sqrt(mse))


def prediction_intervals_from_samples(
    samples: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.05,
) -> Sequence[np.ndarray]:
    """Construct prediction intervals on missing entries only."""

    sample_array = np.asarray(samples, dtype=np.float32)
    if sample_array.ndim != 3:
        raise ValueError(
            f"Expected posterior samples with shape (n_samples, n_rows, n_features), received {sample_array.shape}."
        )
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1.")

    resolved_mask = validate_mask(mask, expected_shape=sample_array.shape[1:])
    missing_mask = resolved_mask == 0.0
    same_pattern = np.all(missing_mask == missing_mask[0])

    if same_pattern:
        miss_idx = np.flatnonzero(missing_mask[0])
        if miss_idx.size == 0:
            return np.zeros((sample_array.shape[1], 0, 2), dtype=np.float32)
        miss_samples = sample_array[:, :, miss_idx]
        lower = np.quantile(miss_samples, alpha / 2.0, axis=0)
        upper = np.quantile(miss_samples, 1.0 - alpha / 2.0, axis=0)
        return np.stack([lower, upper], axis=-1).astype(np.float32)

    intervals = []
    for row_idx in range(sample_array.shape[1]):
        miss_idx = np.flatnonzero(missing_mask[row_idx])
        if miss_idx.size == 0:
            intervals.append(np.zeros((0, 2), dtype=np.float32))
            continue
        row_samples = sample_array[:, row_idx, miss_idx]
        lower = np.quantile(row_samples, alpha / 2.0, axis=0)
        upper = np.quantile(row_samples, 1.0 - alpha / 2.0, axis=0)
        intervals.append(np.stack([lower, upper], axis=-1).astype(np.float32))
    return intervals


class MissForestRegressorImputer:
    """Simple MissForest-style imputer for continuous features."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_iter: int = 5,
        random_state: Optional[int] = 123,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.max_iter = int(max_iter)
        self.random_state = None if random_state is None else int(random_state)

    def fit_transform(self, x_obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Iteratively fit feature-wise regressors and update missing entries."""

        observed, resolved_mask = prepare_masked_data(x_obs, mask)
        imputed = mean_impute(observed, resolved_mask)
        n_rows, n_features = imputed.shape
        if n_rows == 0 or n_features == 0:
            return imputed.astype(np.float32)

        missing_fraction = (1.0 - resolved_mask).mean(axis=0)
        update_order = np.argsort(missing_fraction)

        for iteration in range(self.max_iter):
            for feature_idx in update_order:
                missing_rows = resolved_mask[:, feature_idx] == 0.0
                observed_rows = ~missing_rows
                if not np.any(missing_rows) or observed_rows.sum() < 2:
                    continue

                predictors = np.ones(n_features, dtype=bool)
                predictors[feature_idx] = False
                if not np.any(predictors):
                    continue

                x_train = imputed[observed_rows][:, predictors]
                y_train = imputed[observed_rows, feature_idx]
                x_pred = imputed[missing_rows][:, predictors]
                if x_train.shape[1] == 0:
                    continue

                model = RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    random_state=None if self.random_state is None else self.random_state + iteration + feature_idx,
                    max_features="sqrt",
                    n_jobs=-1,
                )
                model.fit(x_train, y_train)
                imputed[missing_rows, feature_idx] = model.predict(x_pred).astype(np.float32)

        return reconstruct_from_mask(observed, resolved_mask, imputed).astype(np.float32)


def mean_imputation_baseline(x_obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Mean imputation baseline."""

    observed, resolved_mask = prepare_masked_data(x_obs, mask)
    return mean_impute(observed, resolved_mask)


def knn_imputation_baseline(
    x_obs: np.ndarray,
    mask: np.ndarray,
    n_neighbors: int = 5,
) -> np.ndarray:
    """KNN imputation baseline."""

    observed, resolved_mask = prepare_masked_data(x_obs, mask)
    masked = np.where(resolved_mask == 1.0, observed, np.nan).astype(np.float32)
    imputer = KNNImputer(n_neighbors=int(n_neighbors))
    filled = imputer.fit_transform(masked).astype(np.float32)
    return reconstruct_from_mask(observed, resolved_mask, filled)


def missforest_imputation_baseline(
    x_obs: np.ndarray,
    mask: np.ndarray,
    n_estimators: int = 100,
    max_iter: int = 5,
    random_state: Optional[int] = 123,
) -> np.ndarray:
    """MissForest-style baseline imputer for continuous features."""

    imputer = MissForestRegressorImputer(
        n_estimators=n_estimators,
        max_iter=max_iter,
        random_state=random_state,
    )
    return imputer.fit_transform(x_obs, mask)


def _mnar_config_summary(params: Optional[dict]) -> str:
    """Compact JSON summary for experiment tables."""

    params = dict(params or {})
    summary = {
        "dataset": params.get("dataset", "Synthetic_MNAR"),
        "n_samples": int(params.get("n_samples", 5000)),
        "x_dim": int(params.get("x_dim", 50)),
        "z_dim": int(params.get("z_dim", 5)),
        "missing_rate": float(params.get("missing_rate", 0.1)),
        "epochs": int(params.get("epochs", 100)),
        "batch_size": int(params.get("batch_size", 128)),
        "knn_neighbors": int(params.get("knn_neighbors", 5)),
        "missforest_iterations": int(params.get("missforest_iterations", 5)),
        "missforest_trees": int(params.get("missforest_trees", 100)),
    }
    return json.dumps(summary, sort_keys=True)


def benchmark_mnar_imputers(
    x_true: np.ndarray,
    x_obs: np.ndarray,
    mask: np.ndarray,
    params: Optional[dict] = None,
    methods: Iterable[str] = ("mean", "knn", "missforest"),
    missing_rate: float = 0.1,
    seed: Optional[int] = None,
    results_path: Optional[str] = None,
) -> pd.DataFrame:
    """Evaluate baseline imputers on a fixed MNAR dataset.

    Parameters
    ----------
    x_true, x_obs, mask
        Fully observed truth, masked observations, and binary observation mask.
    params
        Optional configuration dictionary. Only baseline hyperparameters are
        read from it.
    methods
        Baseline methods to evaluate. Supported values are ``mean``, ``knn``,
        and ``missforest``.
    missing_rate
        Label written to the result table.
    seed
        Seed recorded in the result table and used for MissForest.
    results_path
        Optional CSV output path.
    """

    params = dict(params or {})
    method_list = tuple(str(method) for method in methods)
    records = []

    for method in method_list:
        if method == "mean":
            x_imputed = mean_imputation_baseline(x_obs, mask)
        elif method == "knn":
            x_imputed = knn_imputation_baseline(
                x_obs,
                mask,
                n_neighbors=int(params.get("knn_neighbors", 5)),
            )
        elif method == "missforest":
            x_imputed = missforest_imputation_baseline(
                x_obs,
                mask,
                n_estimators=int(params.get("missforest_trees", 100)),
                max_iter=int(params.get("missforest_iterations", 5)),
                random_state=seed,
            )
        else:
            raise ValueError(f"Unsupported baseline method: {method}")

        records.append(
            {
                "method_name": method,
                "missingness_rate": float(missing_rate),
                "rmse": rmse_on_missing_entries(x_true, x_imputed, mask),
                "seed": None if seed is None else int(seed),
                "config_summary": _mnar_config_summary(params),
            }
        )

    results = pd.DataFrame.from_records(records).sort_values("rmse", kind="stable").reset_index(drop=True)
    if results_path:
        results_dir = os.path.dirname(results_path)
        if results_dir:
            os.makedirs(results_dir, exist_ok=True)
        results.to_csv(results_path, index=False)
    return results
