import numpy as np

from bayesgm.utils.mnar import benchmark_mnar_imputers, reconstruct_from_mask


def test_mask_reconstruction():
    x_obs = np.array([[1.0, 2.0, 0.0], [0.0, 5.0, 6.0]], dtype=np.float32)
    mask = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]], dtype=np.float32)
    x_mis = np.array([[9.0, 9.0, 3.0], [4.0, 9.0, 9.0]], dtype=np.float32)

    reconstructed = reconstruct_from_mask(x_obs, mask, x_mis)

    expected = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    np.testing.assert_allclose(reconstructed, expected)


def test_baseline_benchmark_returns_finite_scores():
    rng = np.random.default_rng(3)
    x_true = rng.normal(size=(20, 4)).astype(np.float32)
    mask = np.ones_like(x_true, dtype=np.float32)
    mask[:10, 0] = 0.0
    mask[5:15, 2] = 0.0
    x_obs = x_true.copy()
    x_obs[mask == 0.0] = np.nan

    results = benchmark_mnar_imputers(
        x_true=x_true,
        x_obs=x_obs,
        mask=mask,
        params={"knn_neighbors": 3, "missforest_trees": 10, "missforest_iterations": 2},
        missing_rate=0.1,
        seed=123,
    )

    assert list(results["method_name"]) == ["mean", "knn", "missforest"] or set(results["method_name"]) == {"mean", "knn", "missforest"}
    assert np.isfinite(results["rmse"]).all()
