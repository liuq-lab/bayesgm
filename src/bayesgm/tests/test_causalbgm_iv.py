import numpy as np
import tensorflow as tf

from bayesgm.datasets import (
    demand_design_h,
    demand_design_structural_function,
    make_demand_design_grid,
    simulate_demand_design_iv,
)
from bayesgm.models.causalbgm import CausalBGM_IV

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


def _make_params(output_dir):
    return {
        "dataset": "DemandDesignIVSmoke",
        "output_dir": str(output_dir),
        "save_res": False,
        "save_model": False,
        "binary_treatment": False,
        "use_bnn": False,
        "z_dims": [1, 1, 1, 1],
        "v_dim": 2,
        "w_dim": 1,
        "lr_theta": 5e-4,
        "lr_z": 5e-4,
        "g_units": [16, 16],
        "e_units": [16, 16],
        "f_units": [16, 8],
        "h_units": [16, 8],
        "dz_units": [16, 8],
        "kl_weight": 0.0,
        "lr": 5e-4,
        "g_d_freq": 1,
        "use_z_rec": True,
        "iv_mc_samples": 4,
        "eval_mc_samples": 4,
        "first_stage_warmup_epochs": 1,
        "structural_map_steps": 3,
    }


def test_demand_design_formula_matches_dfiv_reference():
    time = np.array([0.0, 2.5, 5.0, 7.5, 10.0], dtype=np.float32)
    price = np.array([10.0, 12.5, 15.0, 17.5, 20.0], dtype=np.float32)
    group = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)

    psi_ref = 2.0 * (
        ((time - 5.0) ** 4) / 600.0
        + np.exp(-4.0 * (time - 5.0) ** 2)
        + time / 10.0
        - 2.0
    )
    structural_ref = 100.0 + (10.0 + price) * group * psi_ref - 2.0 * price

    np.testing.assert_allclose(demand_design_h(time), psi_ref, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        demand_design_structural_function(price, time, group),
        structural_ref,
        rtol=1e-6,
        atol=1e-6,
    )


def test_causalbgm_iv_smoke(tmp_path):
    train = simulate_demand_design_iv(n_samples=96, rho=0.5, seed=7)
    params = _make_params(tmp_path)

    model = CausalBGM_IV(params=params, random_seed=13)
    model.fit(
        data=(train["x"], train["y"], train["v"], train["w"]),
        epochs=2,
        epochs_per_eval=1,
        batch_size=24,
        use_egm_init=False,
        verbose=0,
    )

    causal_pre, mse_x, mse_y, mse_v = model.evaluate(
        data=(train["x"], train["y"], train["v"], train["w"]),
        data_z=None,
        nb_intervals=6,
    )
    assert causal_pre.shape == (6,)
    assert np.isfinite(float(mse_x))
    assert np.isfinite(float(mse_y))
    assert np.isfinite(float(mse_v))

    grid = make_demand_design_grid(price_points=4, time_points=3)
    structural_pred = model.predict_structural(
        grid["x"], grid["v"], latent_method="map", map_steps=3
    )
    structural_mse = model.evaluate_structural_mse(
        grid["x"], grid["v"], grid["y_struct"], latent_method="map", map_steps=3
    )
    assert structural_pred.shape == grid["y_struct"].shape
    assert np.isfinite(structural_mse)

    structural_pred_mcmc, structural_interval_mcmc = model.predict_structural(
        grid["x"][:6],
        grid["v"][:6],
        latent_method="mcmc",
        n_mcmc=3,
        burn_in=3,
        q_sd=0.5,
        bs=3,
        return_interval=True,
    )
    assert structural_pred_mcmc.shape == (6, 1)
    assert structural_interval_mcmc.shape == (6, 2)

    structural_mse_mcmc = model.evaluate_structural_mse(
        grid["x"][:6],
        grid["v"][:6],
        grid["y_struct"][:6],
        latent_method="mcmc",
        n_mcmc=3,
        burn_in=3,
        q_sd=0.5,
        bs=3,
    )
    assert np.isfinite(structural_mse_mcmc)

    outcome_mean, outcome_interval = model.predict_outcome(
        data=(train["x"][:8], train["v"][:8], train["w"][:8]),
        n_mcmc=3,
        burn_in=3,
        q_sd=0.5,
        sample_y=False,
        bs=4,
    )
    assert outcome_mean.shape == (8,)
    assert outcome_interval.shape == (8, 2)

    adrf, adrf_interval = model.predict(
        data=(train["x"][:8], train["v"][:8], train["w"][:8]),
        n_mcmc=3,
        burn_in=3,
        x_values=[12.0, 18.0],
        q_sd=0.5,
        sample_y=False,
        bs=4,
    )
    assert adrf.shape == (2,)
    assert adrf_interval.shape == (2, 2)


def test_causalbgm_iv_records_structural_history(tmp_path):
    train = simulate_demand_design_iv(n_samples=64, rho=0.5, seed=11)
    grid = make_demand_design_grid(price_points=3, time_points=2)
    params = _make_params(tmp_path)
    params["use_bnn"] = False
    params["fit_model_selection_metric"] = "structural_mse"
    params["fit_restore_best_weights"] = True

    model = CausalBGM_IV(params=params, random_seed=13)

    def callback(model, stage, epoch, metrics):
        return {
            "structural_mse": model.evaluate_structural_mse(
                grid["x"],
                grid["v"],
                grid["y_struct"],
                latent_method="encoder",
            )
        }

    model.fit(
        data=(train["x"], train["y"], train["v"], train["w"]),
        epochs=1,
        epochs_per_eval=1,
        batch_size=16,
        use_egm_init=True,
        egm_n_iter=0,
        egm_batches_per_eval=1,
        verbose=0,
        first_stage_warmup_epochs=0,
        evaluation_callback=callback,
    )

    assert len(model.training_history) >= 2
    assert model.training_history[0]["stage"] == "post_egm"
    assert np.isfinite(model.training_history[0]["structural_mse"])
    assert model.best_training_record["epoch"] == 0
