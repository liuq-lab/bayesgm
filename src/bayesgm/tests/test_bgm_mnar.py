import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

from bayesgm.datasets import simulate_mnar_factor_data
from bayesgm.models.bgm.mnar import BGM_MNAR
from bayesgm.utils import prepare_masked_data


def _small_params(seed=7):
    return {
        "dataset": "Synthetic_MNAR_Test",
        "output_dir": ".",
        "save_model": False,
        "save_res": False,
        "use_bnn": False,
        "n_samples": 32,
        "x_dim": 6,
        "z_dim": 2,
        "batch_size": 8,
        "epochs": 2,
        "epochs_per_eval": 1,
        "lr": 1e-3,
        "lr_theta": 1e-3,
        "lr_phi": 1e-3,
        "lr_z": 1e-2,
        "lr_x": 1e-2,
        "gamma": 0.0,
        "alpha": 0.0,
        "kl_weight": 5e-5,
        "g_d_freq": 1,
        "seed": seed,
        "missing_rate": 0.3,
        "g_units": [16, 16],
        "e_units": [16, 16],
        "dz_units": [8, 4],
        "dx_units": [8, 4],
        "missingness_units": [16, 16],
        "egm_init": {
            "enabled": False,
            "n_iter": 1,
            "batches_per_eval": 1,
        },
        "posterior": {
            "z_map_steps": 1,
            "x_map_steps": 1,
            "test_epochs": 2,
            "test_z_map_steps": 1,
            "test_x_map_steps": 1,
            "clip_grad_norm": 10.0,
            "x_clip_value": 6.0,
            "x_refine_steps_for_mcmc": 1,
            "z_update_mode": "map",
            "x_update_mode": "map",
        },
    }


def test_local_posterior_updates_only_change_missing_entries():
    params = _small_params()
    model = BGM_MNAR(params, random_seed=params["seed"])

    dataset = simulate_mnar_factor_data(
        n_samples=16,
        x_dim=params["x_dim"],
        z_dim=params["z_dim"],
        missing_rate=0.4,
        seed=params["seed"],
    )
    x_init, mask = prepare_masked_data(
        dataset["x_obs"],
        mask=dataset["mask"],
        initialization="missforest",
        random_state=params["seed"],
    )
    x_obs, _ = prepare_masked_data(dataset["x_obs"], mask)

    z_init = model.e_net(x_init, training=False).numpy().astype(np.float32)
    batch_z = tf.Variable(z_init, trainable=True)
    batch_x_obs = tf.convert_to_tensor(x_obs, dtype=tf.float32)
    batch_mask = tf.convert_to_tensor(mask, dtype=tf.float32)
    batch_x = tf.Variable(x_init + (1.0 - mask) * 0.25, trainable=True)
    batch_indices = tf.convert_to_tensor(np.arange(x_obs.shape[0]), dtype=tf.int32)
    x_optimizer = tf.keras.optimizers.Adam(params["lr_x"], beta_1=0.9, beta_2=0.99)
    before = batch_x.numpy().copy()

    for _ in range(3):
        model._map_update_x(batch_z, batch_x, batch_indices, batch_x_obs, batch_mask, x_optimizer)

    after = batch_x.numpy()
    np.testing.assert_allclose(after[mask == 1.0], x_obs[mask == 1.0], atol=1e-6)
    assert np.any(np.abs(after[mask == 0.0] - before[mask == 0.0]) > 1e-7)


def test_observed_entries_stay_fixed_during_prediction():
    params = _small_params(seed=9)
    dataset = simulate_mnar_factor_data(
        n_samples=24,
        x_dim=params["x_dim"],
        z_dim=params["z_dim"],
        missing_rate=0.35,
        seed=params["seed"],
    )

    model = BGM_MNAR(params, random_seed=params["seed"])
    model.fit(data=dataset["x_obs"], mask=dataset["mask"], x_true=dataset["x_full"], verbose=0)
    imputed, _ = model.predict(
        data=dataset["x_obs"],
        mask=dataset["mask"],
        return_samples=False,
        n_mcmc=4,
        burn_in=4,
        step_size=0.01,
        num_leapfrog_steps=4,
        seed=params["seed"],
    )

    np.testing.assert_allclose(imputed[dataset["mask"] == 1.0], dataset["x_full"][dataset["mask"] == 1.0], atol=1e-5)


def test_same_training_data_skips_map_stage():
    params = _small_params(seed=13)
    dataset = simulate_mnar_factor_data(
        n_samples=24,
        x_dim=params["x_dim"],
        z_dim=params["z_dim"],
        missing_rate=0.3,
        seed=params["seed"],
    )

    model = BGM_MNAR(params, random_seed=params["seed"])
    model.fit(data=dataset["x_obs"], mask=dataset["mask"], x_true=dataset["x_full"], verbose=0)

    def fail_run_map(*args, **kwargs):
        raise AssertionError("_run_map_inference should not be called for training data.")

    model._run_map_inference = fail_run_map
    imputed, _ = model.predict(
        data=dataset["x_obs"],
        mask=dataset["mask"],
        return_samples=False,
        n_mcmc=4,
        burn_in=4,
        step_size=0.01,
        num_leapfrog_steps=4,
        seed=params["seed"],
        verbose=0,
    )

    assert np.isfinite(imputed).all()


def test_adapt_prediction_updates_theta_and_phi():
    params = _small_params(seed=15)
    train = simulate_mnar_factor_data(
        n_samples=24,
        x_dim=params["x_dim"],
        z_dim=params["z_dim"],
        missing_rate=0.3,
        seed=params["seed"],
    )
    shifted = simulate_mnar_factor_data(
        n_samples=24,
        x_dim=params["x_dim"],
        z_dim=params["z_dim"],
        missing_rate=0.3,
        seed=params["seed"] + 1,
    )

    model = BGM_MNAR(params, random_seed=params["seed"])
    model.fit(data=train["x_obs"], mask=train["mask"], x_true=train["x_full"], verbose=0)

    theta_calls = {"count": 0}
    phi_calls = {"count": 0}
    original_update_theta = model._update_theta
    original_update_phi = model._update_phi

    def counted_theta(*args, **kwargs):
        theta_calls["count"] += 1
        return original_update_theta(*args, **kwargs)

    def counted_phi(*args, **kwargs):
        phi_calls["count"] += 1
        return original_update_phi(*args, **kwargs)

    model._update_theta = counted_theta
    model._update_phi = counted_phi
    imputed, _ = model.predict(
        data=shifted["x_obs"],
        mask=shifted["mask"],
        x_true=shifted["x_full"],
        adapt=True,
        return_samples=False,
        n_mcmc=4,
        burn_in=4,
        step_size=0.01,
        num_leapfrog_steps=4,
        seed=params["seed"],
        verbose=0,
    )

    assert theta_calls["count"] > 0
    assert phi_calls["count"] > 0
    assert np.isfinite(imputed).all()


def test_smoke_fit_and_predict_are_finite():
    params = _small_params(seed=11)
    dataset = simulate_mnar_factor_data(
        n_samples=24,
        x_dim=params["x_dim"],
        z_dim=params["z_dim"],
        missing_rate=0.3,
        seed=params["seed"],
    )

    model = BGM_MNAR(params, random_seed=params["seed"])
    model.fit(data=dataset["x_obs"], mask=dataset["mask"], x_true=dataset["x_full"], verbose=0)

    assert model.training_history_, "Training history should not be empty."
    for metrics in model.training_history_:
        for value in metrics.values():
            if isinstance(value, (int, float)):
                assert np.isfinite(value)

    imputed, intervals = model.predict(
        data=dataset["x_obs"],
        mask=dataset["mask"],
        return_samples=False,
        n_mcmc=4,
        burn_in=4,
        step_size=0.01,
        num_leapfrog_steps=4,
        seed=params["seed"],
    )
    assert np.isfinite(imputed).all()
    assert model.last_prediction_ is not None
    assert np.isfinite(model.last_prediction_["map_imputed"]).all()
    if isinstance(intervals, list):
        for item in intervals:
            assert np.isfinite(item).all()
    else:
        assert np.isfinite(intervals).all()
