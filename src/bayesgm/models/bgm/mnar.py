r"""MNAR-aware Bayesian generative model built on top of BGM.

The model factorizes as

.. math::

    p(X, R, Z, \theta, \phi)
    = \pi(\theta)\pi(\phi)\prod_i \pi_Z(z_i) p_\theta(x_i \mid z_i) p_\phi(r_i \mid x_i),

where missing entries of ``X`` are latent variables updated during inference.
"""

from __future__ import annotations

import json
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
tfm = tfp.mcmc

from bayesgm.utils import (
    prediction_intervals_from_samples,
    prepare_masked_data,
    reconstruct_from_mask,
    rmse_on_missing_entries,
)
from ..networks import BaseFullyConnectedNet
from .base import BGM


class BGM_MNAR(BGM):
    """BGM extension for MNAR imputation with latent missing values."""

    def __init__(
        self,
        params: dict,
        timestamp: Optional[str] = None,
        random_seed: Optional[int] = None,
    ) -> None:
        super().__init__(params=params, timestamp=timestamp, random_seed=random_seed)

        self.rng = np.random.default_rng() if random_seed is None else np.random.default_rng(random_seed)
        self.egm_params = self.params["egm_init"]
        self.posterior_params = self.params["posterior"]

        self.phi_net = BaseFullyConnectedNet(
            input_dim=self.params["x_dim"],
            output_dim=self.params["x_dim"],
            model_name="missingness_net",
            nb_units=self.params["missingness_units"],
        )
        self.phi_net(tf.zeros((1, self.params["x_dim"]), dtype=tf.float32))

        self.z_optimizer = tf.keras.optimizers.Adam(self.params["lr_z"], beta_1=0.9, beta_2=0.99)
        self.x_optimizer = tf.keras.optimizers.Adam(self.params["lr_x"], beta_1=0.9, beta_2=0.99)
        self.phi_optimizer = tf.keras.optimizers.Adam(self.params["lr_phi"], beta_1=0.9, beta_2=0.99)

        self.ckpt = tf.train.Checkpoint(
            g_net=self.g_net,
            e_net=self.e_net,
            dz_net=self.dz_net,
            dx_net=self.dx_net,
            phi_net=self.phi_net,
            g_pre_optimizer=self.g_pre_optimizer,
            d_pre_optimizer=self.d_pre_optimizer,
            g_optimizer=self.g_optimizer,
            z_optimizer=self.z_optimizer,
            x_optimizer=self.x_optimizer,
            phi_optimizer=self.phi_optimizer,
        )
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_path, max_to_keep=100)
        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)

    def get_config(self) -> dict:
        """Return the model configuration."""

        return {"params": self.params}

    @tf.function
    def _generator_nll(
        self,
        data_z: tf.Tensor,
        data_x: tf.Tensor,
        training: bool = False,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Negative log-likelihood under ``p_theta(x | z)``."""

        mu_x, sigma_square_x = self.g_net(data_z, training=training)
        sigma_square_x = tf.clip_by_value(sigma_square_x, 1e-6, 1e6)
        nll = tf.reduce_sum(
            ((data_x - mu_x) ** 2) / (2.0 * sigma_square_x) + 0.5 * tf.math.log(sigma_square_x),
            axis=1,
        )
        mse = tf.reduce_mean((data_x - mu_x) ** 2)
        return nll, mse

    @tf.function
    def _mask_nll(
        self,
        data_x: tf.Tensor,
        mask: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        """Negative log-likelihood under ``p_phi(r | x)``."""

        logits = self.phi_net(data_x, training=training)
        bce = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(mask, tf.float32), logits=logits)
        return tf.reduce_sum(bce, axis=1)
    

    def _map_update_z(
        self,
        data_z: tf.Variable,
        data_x: tf.Variable,
        batch_indices: tf.Tensor,
        z_optimizer: tf.keras.optimizers.Optimizer,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Run one Adam MAP step for the latent state ``z``."""

        with tf.GradientTape() as tape:
            batch_z = tf.gather(data_z, batch_indices, axis=0)
            batch_x = tf.gather(data_x, batch_indices, axis=0)
            loss_px_z = tf.reduce_mean(self._generator_nll(batch_z, batch_x, training=False)[0])
            loss_prior_z = tf.reduce_mean(0.5 * tf.reduce_sum(batch_z ** 2, axis=1))
            loss = loss_px_z + loss_prior_z

        gradients = tape.gradient(loss, data_z)
        gradients = self._clip_gradient(gradients, self.posterior_params["clip_grad_norm"])
        z_optimizer.apply_gradients([(gradients, data_z)])
        return loss, loss_px_z, loss_prior_z
    
    def _map_update_x(
        self,
        data_z: tf.Tensor,
        data_x: tf.Variable,
        batch_indices: tf.Tensor,
        x_obs: tf.Tensor,
        mask: tf.Tensor,
        x_optimizer: tf.keras.optimizers.Optimizer,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Run one Adam MAP step for the current completed matrix."""

        with tf.GradientTape() as tape:
            batch_z = tf.gather(data_z, batch_indices, axis=0)
            batch_x = tf.gather(data_x, batch_indices, axis=0)
            loss_px_z = tf.reduce_mean(self._generator_nll(batch_z, batch_x, training=False)[0])
            loss_pr_x = tf.reduce_mean(self._mask_nll(batch_x, mask, training=False))
            loss = loss_px_z + loss_pr_x

        gradients = tape.gradient(loss, data_x)
        gradients = self._clip_gradient(
            gradients,
            self.posterior_params["clip_grad_norm"],
            batch_mask=mask,
        )
        x_optimizer.apply_gradients([(gradients, data_x)])

        updated_batch = tf.gather(data_x, batch_indices, axis=0)
        # updated_batch = tf.clip_by_value(
        #     updated_batch,
        #     -self.posterior_params["x_clip_value"],
        #     self.posterior_params["x_clip_value"],
        # )
        updated_batch = reconstruct_from_mask(x_obs, mask, updated_batch)
        data_x.scatter_nd_update(tf.expand_dims(batch_indices, axis=1), updated_batch)
        return loss, loss_px_z, loss_pr_x

    @tf.function
    def _update_theta(
        self,
        data_z: tf.Tensor,
        data_x: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Update the generator parameters ``theta``."""

        with tf.GradientTape() as tape:
            loss_x, loss_mse = self._generator_nll(data_z, data_x, training=True)
            loss_x = tf.reduce_mean(loss_x)
            if self.params["use_bnn"]:
                loss_kl = sum(self.g_net.losses)
                loss_x += loss_kl * self.params["kl_weight"]

        gradients = tape.gradient(loss_x, self.g_net.trainable_variables)
        # gradients = [
        #     None if grad is None else tf.clip_by_norm(grad, self.posterior_params["clip_grad_norm"])
        #     for grad in gradients
        # ]
        self.g_optimizer.apply_gradients(zip(gradients, self.g_net.trainable_variables))
        return loss_x, loss_mse

    @tf.function
    def _update_phi(
        self,
        data_x: tf.Tensor,
        mask: tf.Tensor,
    ) -> tf.Tensor:
        """Update the missingness-model parameters ``phi``."""

        with tf.GradientTape() as tape:
            loss_r_x = tf.reduce_mean(self._mask_nll(data_x, mask, training=True))
            if self.params["use_bnn"]:
                loss_kl = sum(self.phi_net.losses)
                loss_r_x += loss_kl * self.params["kl_weight"]

        gradients = tape.gradient(loss_r_x, self.phi_net.trainable_variables)
        # gradients = [
        #     None if grad is None else tf.clip_by_norm(grad, self.posterior_params["clip_grad_norm"])
        #     for grad in gradients
        # ]
        self.phi_optimizer.apply_gradients(zip(gradients, self.phi_net.trainable_variables))
        return loss_r_x

    def _evaluate_state(
        self,
        x_obs: np.ndarray,
        mask: np.ndarray,
        data_z: tf.Variable,
        data_x: tf.Variable,
        x_true: Optional[np.ndarray] = None,
    ) -> dict:
        """Evaluate the current completed matrix and latent state."""

        x_full = reconstruct_from_mask(x_obs, mask, data_x.numpy())
        z_tensor = tf.convert_to_tensor(data_z.numpy(), dtype=tf.float32)
        x_tensor = tf.convert_to_tensor(x_full, dtype=tf.float32)
        mask_tensor = tf.convert_to_tensor(mask, dtype=tf.float32)

        loss_px_z, loss_mse = self._generator_nll(z_tensor, x_tensor, training=False)
        loss_pr_x = self._mask_nll(x_tensor, mask_tensor, training=False)
        loss_prior_z = tf.reduce_mean(0.5 * tf.reduce_sum(z_tensor ** 2, axis=1))

        metrics = {
            "generator_nll": float(tf.reduce_mean(loss_px_z).numpy()),
            "mask_nll": float(tf.reduce_mean(loss_pr_x).numpy()),
            "latent_prior": float(loss_prior_z.numpy()),
            "reconstruction_mse": float(loss_mse.numpy()),
        }
        if x_true is not None:
            metrics["rmse_missing_only"] = rmse_on_missing_entries(x_true, x_full, mask)
        return metrics

    def fit(
        self,
        data: np.ndarray,
        mask: Optional[np.ndarray] = None,
        x_true: Optional[np.ndarray] = None,
        verbose: int = 1,
    ) -> "BGM_MNAR":
        """Fit the MNAR model on incomplete data."""

        self.train_data = np.array(data, copy=True)
        
        # fill missing values with missforest and infer mask from NaN in data
        x_obs, resolved_mask = prepare_masked_data(
            data,
            mask=mask,
            initialization="mean"
        )

        if x_obs.shape[1] != self.params["x_dim"]:
            raise ValueError(
                f"Expected feature dimension {self.params['x_dim']}, received {x_obs.shape[1]}."
            )

        if self.params["save_res"]:
            os.makedirs(self.save_dir, exist_ok=True)
            with open(os.path.join(self.save_dir, "params.txt"), "w", encoding="utf-8") as handle:
                handle.write(json.dumps(self.get_config(), indent=2))

        # EGM warm start
        if self.egm_params["enabled"]:
            self.egm_init(
                data=x_obs,
                egm_n_iter=self.egm_params["n_iter"],
                batch_size=self.params["batch_size"],
                egm_batches_per_eval=self.egm_params["batches_per_eval"],
                verbose=verbose,
            )
            z_init = self.e_net(x_obs, training=False).numpy().astype(np.float32)
        else:
            z_init = self.rng.normal(size=(x_obs.shape[0], self.params["z_dim"])).astype(np.float32)

        self.data_z = tf.Variable(z_init, name="LatentZ", trainable=True)
        self.data_x = tf.Variable(x_obs, name="CompletedX", trainable=True)

        self.training_history_ = []
        self.last_prediction_ = None
        n_rows = x_obs.shape[0]
        batch_size = min(self.params["batch_size"], n_rows)

        for epoch in range(1, self.params["epochs"] + 1):
            permutation = self.rng.permutation(n_rows)
            epoch_metrics = {
                "epoch": epoch,
                "latent_map_loss": [],
                "x_map_loss": [],
                "theta_loss": [],
                "phi_loss": [],
            }

            for start in range(0, n_rows, batch_size):
                batch_idx = permutation[start:start + batch_size]
                batch_indices = tf.convert_to_tensor(batch_idx, dtype=tf.int32)
                batch_x_obs = tf.convert_to_tensor(x_obs[batch_idx], dtype=tf.float32)
                batch_mask = tf.convert_to_tensor(resolved_mask[batch_idx], dtype=tf.float32)

                # Alternate local posterior updates and global parameter updates.
                for _ in range(self.posterior_params["z_map_steps"]):
                    latent_map_loss, _, _ = self._map_update_z(
                        self.data_z,
                        self.data_x,
                        batch_indices,
                        z_optimizer = self.z_optimizer,
                    )
                for _ in range(self.posterior_params["x_map_steps"]):
                    x_map_loss, _, _ = self._map_update_x(
                        self.data_z,
                        self.data_x,
                        batch_indices,
                        batch_x_obs,
                        batch_mask,
                        x_optimizer = self.x_optimizer,
                    )

                batch_z = tf.gather(self.data_z, batch_indices, axis=0)
                batch_x = tf.gather(self.data_x, batch_indices, axis=0)

                theta_loss, _ = self._update_theta(batch_z, batch_x)
                phi_loss = self._update_phi(batch_x, batch_mask)

                self._assert_finite_numpy(batch_z.numpy(), "batch_z")
                self._assert_finite_numpy(batch_x.numpy(), "batch_x")

                epoch_metrics["latent_map_loss"].append(float(latent_map_loss.numpy()))
                epoch_metrics["x_map_loss"].append(float(x_map_loss.numpy()))
                epoch_metrics["theta_loss"].append(float(theta_loss.numpy()))
                epoch_metrics["phi_loss"].append(float(phi_loss.numpy()))

            if epoch % self.params["epochs_per_eval"] == 0 or epoch == self.params["epochs"]:
                metrics = self._evaluate_state(x_obs, resolved_mask, self.data_z, self.data_x, x_true=x_true)
                metrics["epoch"] = epoch
                metrics["latent_map_loss_mean"] = float(np.mean(epoch_metrics["latent_map_loss"]))
                metrics["x_map_loss_mean"] = float(np.mean(epoch_metrics["x_map_loss"]))
                metrics["theta_loss_mean"] = float(np.mean(epoch_metrics["theta_loss"]))
                metrics["phi_loss_mean"] = float(np.mean(epoch_metrics["phi_loss"]))
                self.training_history_.append(metrics)
                if verbose:
                    print(
                        "Epoch [{}/{}] latent_map={:.4f} x_map={:.4f} theta={:.4f} phi={:.4f} rmse={}".format(
                            epoch,
                            self.params["epochs"],
                            metrics["latent_map_loss_mean"],
                            metrics["x_map_loss_mean"],
                            metrics["theta_loss_mean"],
                            metrics["phi_loss_mean"],
                            "n/a" if "rmse_missing_only" not in metrics else f"{metrics['rmse_missing_only']:.4f}",
                        )
                    )

        if self.params["save_res"]:
            pd.DataFrame(self.training_history_).to_csv(
                os.path.join(self.save_dir, "training_history.csv"),
                index=False,
            )

        return self

    def _run_map_inference(
        self,
        x_obs: np.ndarray,
        mask: np.ndarray,
        epochs: int,
        z_map_steps: int,
        x_map_steps: int,
        x_true: Optional[np.ndarray] = None,
        verbose: int = 1,
        adapt: bool = False,
    ) -> Tuple[tf.Variable, tf.Variable]:
        """Run alternating MAP updates, optionally adapting ``theta`` and ``phi``."""

        if self.egm_params["enabled"]:
            z_init = self.e_net(x_obs, training=False).numpy().astype(np.float32)
        else:
            z_init = self.rng.normal(size=(x_obs.shape[0], self.params["z_dim"])).astype(np.float32)

        data_z = tf.Variable(z_init, trainable=True)
        data_x = tf.Variable(x_obs, trainable=True)   

        z_optimizer = tf.keras.optimizers.Adam(self.params["lr_z"], beta_1=0.9, beta_2=0.99)
        x_optimizer = tf.keras.optimizers.Adam(self.params["lr_x"], beta_1=0.9, beta_2=0.99)
        self.inference_history_ = []

        n_rows = x_obs.shape[0]
        batch_size = min(self.params["batch_size"], n_rows)

        for epoch in range(epochs):
            permutation = self.rng.permutation(n_rows)
            epoch_metrics = {
                "latent_map_loss": [],
                "x_map_loss": [],
                "theta_loss": [],
                "phi_loss": [],
            }
            for start in range(0, n_rows, batch_size):
                batch_idx = permutation[start:start + batch_size]
                batch_indices = tf.convert_to_tensor(batch_idx, dtype=tf.int32)
                batch_x_obs = tf.convert_to_tensor(x_obs[batch_idx], dtype=tf.float32)
                batch_mask = tf.convert_to_tensor(mask[batch_idx], dtype=tf.float32)

                for _ in range(z_map_steps):
                    latent_map_loss, _, _ = self._map_update_z(
                        data_z,
                        data_x,
                        batch_indices,
                        z_optimizer = z_optimizer,
                    )
                    epoch_metrics["latent_map_loss"].append(float(latent_map_loss.numpy()))
                for _ in range(x_map_steps):
                    x_map_loss, _, _ = self._map_update_x(
                        data_z,
                        data_x,
                        batch_indices,
                        batch_x_obs,
                        batch_mask,
                        x_optimizer = x_optimizer,
                    )
                    epoch_metrics["x_map_loss"].append(float(x_map_loss.numpy()))

                batch_z = tf.gather(data_z, batch_indices, axis=0)
                batch_x = tf.gather(data_x, batch_indices, axis=0)
                if adapt:
                    theta_loss, _ = self._update_theta(batch_z, batch_x)
                    phi_loss = self._update_phi(batch_x, batch_mask)
                    epoch_metrics["theta_loss"].append(float(theta_loss.numpy()))
                    epoch_metrics["phi_loss"].append(float(phi_loss.numpy()))
                self._assert_finite_numpy(batch_z.numpy(), "batch_z")
                self._assert_finite_numpy(batch_x.numpy(), "batch_x")
            
            if epoch % self.params["epochs_per_eval"] == 0:
                metrics = self._evaluate_state(x_obs, mask, data_z, data_x, x_true=x_true)
                metrics["epoch"] = epoch
                metrics["latent_map_loss_mean"] = float(np.mean(epoch_metrics["latent_map_loss"]))
                metrics["x_map_loss_mean"] = float(np.mean(epoch_metrics["x_map_loss"]))
                if adapt:
                    metrics["theta_loss_mean"] = float(np.mean(epoch_metrics["theta_loss"]))
                    metrics["phi_loss_mean"] = float(np.mean(epoch_metrics["phi_loss"]))
                self.inference_history_.append(metrics)
                if verbose:
                    if adapt:
                        print(
                            "Adapt Epoch [{}/{}] latent_map={:.4f} x_map={:.4f} theta={:.4f} phi={:.4f} rmse={}".format(
                                epoch,
                                epochs,
                                metrics["latent_map_loss_mean"],
                                metrics["x_map_loss_mean"],
                                metrics["theta_loss_mean"],
                                metrics["phi_loss_mean"],
                                "n/a" if "rmse_missing_only" not in metrics else f"{metrics['rmse_missing_only']:.4f}",
                            )
                        )
                    else:
                        print(
                            "Test Epoch [{}/{}] latent_map={:.4f} x_map={:.4f} rmse={}".format(
                                epoch,
                                epochs,
                                metrics["latent_map_loss_mean"],
                                metrics["x_map_loss_mean"],
                                "n/a" if "rmse_missing_only" not in metrics else f"{metrics['rmse_missing_only']:.4f}",
                            )
                        )

        return data_z, data_x

    def _assert_finite_numpy(self, array: np.ndarray, name: str) -> None:
        """Raise a clear error when a latent state becomes non-finite."""

        if not np.isfinite(array).all():
            raise FloatingPointError(f"{name} became non-finite during MNAR inference.")

    @staticmethod
    def _clip_gradient(gradient: tf.Tensor, clip_norm: float, batch_mask: Optional[tf.Tensor] = None):
        """Clip gradients and zero out observed-entry updates when needed."""

        if gradient is None:
            raise FloatingPointError("Received a None gradient during MNAR posterior updates.")

        if isinstance(gradient, tf.IndexedSlices):
            values = gradient.values
            if batch_mask is not None:
                values = values * (1.0 - tf.cast(batch_mask, values.dtype))
            values = tf.clip_by_norm(values, clip_norm)
            return tf.IndexedSlices(values=values, indices=gradient.indices, dense_shape=gradient.dense_shape)

        return tf.clip_by_norm(gradient, clip_norm)


    def _posterior_outputs_from_state(
        self,
        x_obs: np.ndarray,
        mask: np.ndarray,
        x_map: np.ndarray,
        z_map: Optional[np.ndarray] = None,
        alpha: Optional[float] = None,
        n_mcmc: Optional[int] = None,
        burn_in: Optional[int] = None,
        step_size: Optional[float] = None,
        num_leapfrog_steps: Optional[int] = None,
        seed: Optional[int] = None,
        return_samples: bool = False,
        verbose: int = 1,
    ) -> dict:
        """Build MAP and MCMC imputations from the current completed state."""

        x_imputed_map = reconstruct_from_mask(x_obs, mask, x_map).astype(np.float32)

        x_obs_tf = tf.convert_to_tensor(x_obs, dtype=tf.float32)
        mask_tf = tf.convert_to_tensor(mask, dtype=tf.float32)
        has_missing = bool(np.any(mask == 0.0))
        n_rows = int(x_obs.shape[0])

        z_state = tf.convert_to_tensor(z_map, dtype=tf.float32)
        x_state = tf.convert_to_tensor(x_imputed_map, dtype=tf.float32)

        num_adaptation_steps = max(0, int(0.8 * int(burn_in)))
        target_accept_prob = 0.75

        x_full_for_z = tf.Variable(
            reconstruct_from_mask(x_obs_tf, mask_tf, x_state),
            trainable=False,
            dtype=tf.float32,
        )
        z_for_x = tf.Variable(z_state, trainable=False, dtype=tf.float32)

        def z_target_log_prob_fn(candidate_z):
            return self.get_log_posterior(candidate_z, x_full_for_z, None, None)

        def x_target_log_prob_fn(candidate_x):
            candidate_full = reconstruct_from_mask(x_obs_tf, mask_tf, candidate_x)
            return -(
                self._generator_nll(z_for_x, candidate_full, training=False)[0]
                + self._mask_nll(candidate_full, mask_tf, training=False)
            )

        def make_adaptive_hmc_kernel(target_log_prob_fn):
            hmc_kernel = tfm.HamiltonianMonteCarlo(
                target_log_prob_fn=target_log_prob_fn,
                step_size=step_size,
                num_leapfrog_steps=num_leapfrog_steps,
            )
            return tfm.SimpleStepSizeAdaptation(
                inner_kernel=hmc_kernel,
                num_adaptation_steps=num_adaptation_steps,
                target_accept_prob=target_accept_prob,
            )

        def refresh_conditional_results(kernel_results, current_state, target_log_prob_fn):
            """Keep adaptive state but refresh HMC cache for the current Gibbs conditional."""

            with tf.GradientTape() as tape:
                tape.watch(current_state)
                target_log_prob = target_log_prob_fn(current_state)
            grads_target_log_prob = tape.gradient(target_log_prob, current_state)
            if grads_target_log_prob is None:
                raise FloatingPointError("Unable to compute HMC target-log-probability gradients.")

            accepted_results = kernel_results.inner_results.accepted_results._replace(
                log_acceptance_correction=tf.zeros_like(target_log_prob),
                target_log_prob=target_log_prob,
                grads_target_log_prob=[grads_target_log_prob],
            )
            proposed_results = kernel_results.inner_results.proposed_results._replace(
                log_acceptance_correction=tf.zeros_like(target_log_prob),
                target_log_prob=target_log_prob,
                grads_target_log_prob=[grads_target_log_prob],
            )
            inner_results = kernel_results.inner_results._replace(
                accepted_results=accepted_results,
                log_accept_ratio=tf.zeros_like(target_log_prob),
                proposed_state=current_state,
                proposed_results=proposed_results,
            )
            return kernel_results._replace(inner_results=inner_results)

        z_kernel = make_adaptive_hmc_kernel(z_target_log_prob_fn)
        x_kernel = make_adaptive_hmc_kernel(x_target_log_prob_fn) if has_missing else None
        z_kernel_results = z_kernel.bootstrap_results(z_state)
        x_kernel_results = x_kernel.bootstrap_results(x_state) if has_missing else None

        z_samples = []
        x_samples = []
        z_acceptance = []
        x_acceptance = []
        generator_mean_sum = np.zeros_like(x_imputed_map, dtype=np.float64)
        retained_steps = 0
        total_steps = burn_in + n_mcmc

        for step in range(total_steps):
            x_full_current = reconstruct_from_mask(x_obs_tf, mask_tf, x_state)
            x_full_for_z.assign(x_full_current)

            # Block 1: update all subject-specific latent states in parallel.
            z_kernel_results = refresh_conditional_results(
                z_kernel_results,
                z_state,
                z_target_log_prob_fn,
            )
            z_state, z_kernel_results = z_kernel.one_step(
                z_state,
                z_kernel_results,
                seed=None if seed is None else seed + 2 * step,
            )
            z_is_accepted = z_kernel_results.inner_results.is_accepted
            z_acceptance.append(float(tf.reduce_mean(tf.cast(z_is_accepted, tf.float32)).numpy()))

            # Block 2: update all subject-specific x states in parallel, then
            # project the observed entries back to x_obs so only missing values move.
            if has_missing:
                z_for_x.assign(z_state)
                x_kernel_results = refresh_conditional_results(
                    x_kernel_results,
                    x_state,
                    x_target_log_prob_fn,
                )
                x_state, x_kernel_results = x_kernel.one_step(
                    x_state,
                    x_kernel_results,
                    seed=None if seed is None else seed + 2 * step + 1,
                )
                x_is_accepted = x_kernel_results.inner_results.is_accepted
                x_acceptance.append(float(tf.reduce_mean(tf.cast(x_is_accepted, tf.float32)).numpy()))
                x_state = reconstruct_from_mask(x_obs_tf, mask_tf, x_state)
                x_full_current = x_state
            else:
                x_full_current = x_imputed_map

            if step >= burn_in:
                z_samples.append(z_state.numpy().astype(np.float32))
                x_samples.append(np.asarray(x_full_current, dtype=np.float32))
                mu_x_state, _ = self.g_net(z_state, training=False)
                generator_mean_full = reconstruct_from_mask(x_obs_tf, mask_tf, mu_x_state)
                generator_mean_sum += generator_mean_full.numpy().astype(np.float64)
                retained_steps += 1

        posterior_z = np.asarray(z_samples, dtype=np.float32)
        sample_predictions = np.asarray(x_samples, dtype=np.float32)
        if not np.isfinite(posterior_z).all():
            raise FloatingPointError("Posterior z samples contain non-finite values.")
        if not np.isfinite(sample_predictions).all():
            raise FloatingPointError("Posterior x samples contain non-finite values.")

        x_imputed_mcmc = np.mean(sample_predictions, axis=0).astype(np.float32)
        x_imputed_mcmc_median = np.median(sample_predictions, axis=0).astype(np.float32)
        x_imputed_mcmc_generator_mean = (generator_mean_sum / max(retained_steps, 1)).astype(np.float32)

        intervals = prediction_intervals_from_samples(sample_predictions, mask, alpha=alpha)
        missing_mask = mask == 0.0
        diagnostics = {
            "z_acceptance_rate": float(np.mean(z_acceptance)) if z_acceptance else 0.0,
            "z_acceptance_rate_last": float(z_acceptance[-1]) if z_acceptance else 0.0,
            "x_acceptance_rate": float(np.mean(x_acceptance)) if x_acceptance else None,
            "x_acceptance_rate_last": float(x_acceptance[-1]) if x_acceptance else None,
            "x_sample_missing_std": float(np.std(sample_predictions[:, missing_mask])) if np.any(missing_mask) else 0.0,
            "step_size": float(step_size),
            "z_step_size_final": float(z_kernel_results.new_step_size.numpy()),
            "x_step_size_final": float(x_kernel_results.new_step_size.numpy()) if x_kernel_results is not None else None,
            "num_leapfrog_steps": int(num_leapfrog_steps),
            "num_adaptation_steps": int(num_adaptation_steps),
            "target_accept_prob": float(target_accept_prob),
            "burn_in": int(burn_in),
            "n_mcmc": int(n_mcmc),
        }
        if verbose:
            print(
                "MCMC diagnostics: z_accept={:.4f}, x_accept={}, step_size={} -> z_step_size_final={}, x_step_size_final={}, leapfrog={}, burn_in={}, n_mcmc={}".format(
                    diagnostics["z_acceptance_rate"],
                    "n/a" if diagnostics["x_acceptance_rate"] is None else f"{diagnostics['x_acceptance_rate']:.4f}",
                    diagnostics["step_size"],
                    diagnostics["z_step_size_final"],
                    "n/a" if diagnostics["x_step_size_final"] is None else diagnostics["x_step_size_final"],
                    diagnostics["num_leapfrog_steps"],
                    diagnostics["burn_in"],
                    diagnostics["n_mcmc"],
                )
            )

        outputs = {
            "map_imputed": x_imputed_map,
            "mcmc_mean_imputed": x_imputed_mcmc,
            "mcmc_median_imputed": x_imputed_mcmc_median,
            "mcmc_generator_mean_imputed": x_imputed_mcmc_generator_mean,
            "intervals": intervals,
            "z_samples": posterior_z,
            "mcmc_diagnostics": diagnostics,
        }
        if return_samples:
            outputs["x_samples"] = sample_predictions
        return outputs

    def predict(
        self,
        data: np.ndarray,
        mask: Optional[np.ndarray] = None,
        x_true: Optional[np.ndarray] = None,
        adapt: bool = False,
        alpha: float = 0.05,
        return_samples: bool = False,
        n_mcmc: int = 3000,
        burn_in: int = 3000,
        step_size: float = 0.01,
        num_leapfrog_steps: int = 10,
        seed: int = 42,
        verbose: int = 1,
    ):
        """Impute incomplete data with MAP updates followed by MCMC over ``z`` and missing ``x``."""

        x_obs, resolved_mask = prepare_masked_data(
            data,
            mask=mask,
            initialization="mean"
        )

        if x_obs.shape[1] != self.params["x_dim"]:
            raise ValueError(
                f"Expected feature dimension {self.params['x_dim']}, received {x_obs.shape[1]}."
            )

        if np.array_equal(np.asarray(data), np.asarray(self.train_data), equal_nan=True):
            # Prediction data match training data; skipping MAP stage
            data_z = self.data_z
            data_x = self.data_x
            self.inference_history_ = []
        else:
            # Prediction data differ from training data; running adaptation stage
            data_z, data_x = self._run_map_inference(
                x_obs=x_obs,
                mask=resolved_mask,
                epochs=self.posterior_params["test_epochs"],
                z_map_steps=self.posterior_params["test_z_map_steps"],
                x_map_steps=self.posterior_params["test_x_map_steps"],
                x_true=x_true,
                verbose=verbose,
                adapt=adapt,
            )

        # Then switch to MCMC for z and refine x conditionally for posterior summaries.
        outputs = self._posterior_outputs_from_state(
            x_obs=x_obs,
            mask=resolved_mask,
            x_map=data_x.numpy(),
            z_map=data_z.numpy(),
            alpha=alpha,
            n_mcmc=n_mcmc,
            burn_in=burn_in,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps,
            seed=seed,
            return_samples=return_samples,
            verbose=verbose,
        )

        self.last_prediction_ = outputs
        if return_samples:
            return outputs["x_samples"], outputs["intervals"]
        return outputs["mcmc_mean_imputed"], outputs["intervals"]
