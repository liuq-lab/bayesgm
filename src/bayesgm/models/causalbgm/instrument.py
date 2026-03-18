import datetime
import os

import dateutil.tz
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm

from bayesgm.datasets import Gaussian_sampler
from bayesgm.utils.data_io import save_data

from ..networks import BaseFullyConnectedNet, BayesianFullyConnectedNet, Discriminator
from .base import CausalBGM


class CausalBGM_IV(CausalBGM):
    """Instrumental-variable extension of :class:`CausalBGM`.

    The observed tuple is ``(X, Y, V, W)`` where ``W`` denotes instrumental
    variables. The model treats ``W`` as fixed observed variables and learns

    - ``p(V | Z)``
    - ``p(X | W, Z0, Z2)``
    - ``p(Y | x, Z0, Z1)``

    The outcome likelihood is replaced by the IV pseudo-likelihood

    ``p(Y | W, Z) = integral p(Y | x, Z0, Z1) p(x | W, Z0, Z2) dx``

    which is evaluated by Monte Carlo for continuous treatments and exactly for
    binary treatments.
    """

    def __init__(self, params, timestamp=None, random_seed=None):
        if "w_dim" not in params:
            raise KeyError("`w_dim` must be provided in params for CausalBGM_IV.")

        self.params = dict(params)
        self.params.setdefault("iv_mc_samples", 16)
        self.params.setdefault("eval_mc_samples", self.params["iv_mc_samples"])
        self.params.setdefault("first_stage_warmup_epochs", 0)
        self.params.setdefault("structural_latent_method", "map")
        self.params.setdefault("structural_map_steps", 100)
        self.params.setdefault("structural_map_lr", self.params["lr_z"])
        self.params.setdefault("structural_mcmc_n_samples", 3000)
        self.params.setdefault("structural_mcmc_burn_in", 5000)
        self.params.setdefault("structural_mcmc_q_sd", 1.0)
        self.params.setdefault("structural_mcmc_bs", 10000)
        self.timestamp = timestamp

        if random_seed is not None:
            tf.keras.utils.set_random_seed(random_seed)
            os.environ["TF_DETERMINISTIC_OPS"] = "1"
            tf.config.experimental.enable_op_determinism()

        z_dim = sum(self.params["z_dims"])
        z0_dim = self.params["z_dims"][0]
        z1_dim = self.params["z_dims"][1]
        z2_dim = self.params["z_dims"][2]

        if self.params["use_bnn"]:
            net_cls = BayesianFullyConnectedNet
        else:
            net_cls = BaseFullyConnectedNet

        self.g_net = net_cls(
            input_dim=z_dim,
            output_dim=self.params["v_dim"] + 1,
            model_name="g_net",
            nb_units=self.params["g_units"],
        )
        self.e_net = net_cls(
            input_dim=self.params["v_dim"],
            output_dim=z_dim,
            model_name="e_net",
            nb_units=self.params["e_units"],
        )
        self.f_net = net_cls(
            input_dim=z0_dim + z1_dim + 1,
            output_dim=2,
            model_name="f_net",
            nb_units=self.params["f_units"],
        )
        self.h_net = net_cls(
            input_dim=z0_dim + z2_dim + self.params["w_dim"],
            output_dim=2,
            model_name="h_net",
            nb_units=self.params["h_units"],
        )

        self.dz_net = Discriminator(
            input_dim=z_dim, model_name="dz_net", nb_units=self.params["dz_units"]
        )

        self.g_pre_optimizer = tf.keras.optimizers.Adam(
            self.params["lr"], beta_1=0.9, beta_2=0.99
        )
        self.d_pre_optimizer = tf.keras.optimizers.Adam(
            self.params["lr"], beta_1=0.9, beta_2=0.99
        )
        self.z_sampler = Gaussian_sampler(mean=np.zeros(z_dim), sd=1.0)

        self.g_optimizer = tf.keras.optimizers.Adam(
            self.params["lr_theta"], beta_1=0.9, beta_2=0.99
        )
        self.f_optimizer = tf.keras.optimizers.Adam(
            self.params["lr_theta"], beta_1=0.9, beta_2=0.99
        )
        self.h_optimizer = tf.keras.optimizers.Adam(
            self.params["lr_theta"], beta_1=0.9, beta_2=0.99
        )
        self.posterior_optimizer = tf.keras.optimizers.Adam(
            self.params["lr_z"], beta_1=0.9, beta_2=0.99
        )

        self.initialize_nets()

        if self.timestamp is None:
            now = datetime.datetime.now(dateutil.tz.tzlocal())
            self.timestamp = now.strftime("%Y%m%d_%H%M%S")

        self.checkpoint_path = "{}/checkpoints/{}/{}".format(
            self.params["output_dir"], self.params["dataset"], self.timestamp
        )
        if self.params["save_model"] and not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

        self.save_dir = "{}/results/{}/{}".format(
            self.params["output_dir"], self.params["dataset"], self.timestamp
        )
        if self.params["save_res"] and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.ckpt = tf.train.Checkpoint(
            g_net=self.g_net,
            e_net=self.e_net,
            f_net=self.f_net,
            h_net=self.h_net,
            dz_net=self.dz_net,
            g_pre_optimizer=self.g_pre_optimizer,
            d_pre_optimizer=self.d_pre_optimizer,
            g_optimizer=self.g_optimizer,
            f_optimizer=self.f_optimizer,
            h_optimizer=self.h_optimizer,
            posterior_optimizer=self.posterior_optimizer,
        )
        self.ckpt_manager = tf.train.CheckpointManager(
            self.ckpt, self.checkpoint_path, max_to_keep=5
        )

        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print("Latest checkpoint restored!!")

    @staticmethod
    def _to_2d_float32(array):
        array = np.asarray(array, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        return array

    def _parse_train_data(self, data):
        if len(data) != 4:
            raise ValueError(
                "CausalBGM_IV.fit/evaluate expect data=(x, y, v, w)."
            )
        data_x, data_y, data_v, data_w = data
        data_x = self._to_2d_float32(data_x)
        data_y = self._to_2d_float32(data_y)
        data_v = self._to_2d_float32(data_v)
        data_w = self._to_2d_float32(data_w)
        if data_v.shape[1] != self.params["v_dim"]:
            raise ValueError(
                f"`v` has dim {data_v.shape[1]}, expected {self.params['v_dim']}."
            )
        if data_w.shape[1] != self.params["w_dim"]:
            raise ValueError(
                f"`w` has dim {data_w.shape[1]}, expected {self.params['w_dim']}."
            )
        return data_x, data_y, data_v, data_w

    def _parse_predict_data(self, data):
        if len(data) == 3:
            data_x, data_v, data_w = data
            data_y = None
        elif len(data) == 4:
            data_x, data_y, data_v, data_w = data
            data_y = self._to_2d_float32(data_y)
        else:
            raise ValueError(
                "CausalBGM_IV.predict expects data=(x, v, w) or data=(x, y, v, w)."
            )
        data_x = self._to_2d_float32(data_x)
        data_v = self._to_2d_float32(data_v)
        data_w = self._to_2d_float32(data_w)
        if data_v.shape[1] != self.params["v_dim"]:
            raise ValueError(
                f"`v` has dim {data_v.shape[1]}, expected {self.params['v_dim']}."
            )
        if data_w.shape[1] != self.params["w_dim"]:
            raise ValueError(
                f"`w` has dim {data_w.shape[1]}, expected {self.params['w_dim']}."
            )
        return data_x, data_y, data_v, data_w

    def _prepare_target_x(self, reference_x, x_values=None):
        if x_values is None:
            return self._to_2d_float32(reference_x)
        if np.isscalar(x_values):
            return np.full_like(reference_x, float(x_values), dtype=np.float32)

        target_x = self._to_2d_float32(x_values)
        if target_x.shape[0] == 1 and reference_x.shape[0] > 1:
            target_x = np.repeat(target_x, reference_x.shape[0], axis=0)
        if target_x.shape[0] != reference_x.shape[0]:
            raise ValueError(
                "`x_values` must be a scalar or have the same number of rows as `x`."
            )
        return target_x.astype(np.float32)

    def _snapshot_trainable_state(self):
        """Capture the trainable network weights for later restoration."""
        return {
            "g_net": self.g_net.get_weights(),
            "e_net": self.e_net.get_weights(),
            "f_net": self.f_net.get_weights(),
            "h_net": self.h_net.get_weights(),
            "dz_net": self.dz_net.get_weights(),
        }

    def _restore_trainable_state(self, state):
        """Restore a previously captured network-weight snapshot."""
        self.g_net.set_weights(state["g_net"])
        self.e_net.set_weights(state["e_net"])
        self.f_net.set_weights(state["f_net"])
        self.h_net.set_weights(state["h_net"])
        self.dz_net.set_weights(state["dz_net"])

    def _coerce_metric_value(self, value):
        if isinstance(value, tf.Tensor):
            value = value.numpy()
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return float(value)
            return value.tolist()
        if isinstance(value, (np.floating, float)):
            return float(value)
        if isinstance(value, (np.integer, int)):
            return int(value)
        return value

    def _record_training_metrics(
        self,
        stage,
        epoch,
        mse_x,
        mse_y,
        mse_v,
        evaluation_callback=None,
        include_outcome=True,
    ):
        """Append a training-history row and enrich it via an optional callback."""
        record = {
            "stage": stage,
            "epoch": None if epoch is None else int(epoch),
            "include_outcome": bool(include_outcome),
            "mse_x": float(mse_x),
            "mse_y": float(mse_y),
            "mse_v": float(mse_v),
        }
        if evaluation_callback is not None:
            extra_metrics = evaluation_callback(
                model=self,
                stage=stage,
                epoch=epoch,
                metrics=dict(record),
            )
            if extra_metrics:
                for key, value in extra_metrics.items():
                    record[key] = self._coerce_metric_value(value)
        self.training_history.append(record)
        return record

    def _format_training_record(self, record):
        pieces = [
            f"MSE_x={record['mse_x']:.4f}",
            f"MSE_y={record['mse_y']:.4f}",
            f"MSE_v={record['mse_v']:.4f}",
        ]
        if "structural_mse" in record:
            pieces.append(f"structural_MSE={record['structural_mse']:.4f}")
        return ", ".join(pieces)

    def initialize_nets(self, print_summary=False):
        """Build all neural networks once so TensorFlow variables exist.

        Parameters
        ----------
        print_summary : bool, default=False
            If ``True``, print Keras summaries for the main sub-networks.
        """
        z_dim = sum(self.params["z_dims"])
        z0_dim = self.params["z_dims"][0]
        z1_dim = self.params["z_dims"][1]
        z2_dim = self.params["z_dims"][2]
        self.g_net(np.zeros((1, z_dim), dtype=np.float32))
        self.e_net(np.zeros((1, self.params["v_dim"]), dtype=np.float32))
        self.f_net(np.zeros((1, z0_dim + z1_dim + 1), dtype=np.float32))
        self.h_net(
            np.zeros(
                (1, z0_dim + z2_dim + self.params["w_dim"]), dtype=np.float32
            )
        )
        if print_summary:
            print(self.g_net.summary())
            print(self.e_net.summary())
            print(self.f_net.summary())
            print(self.h_net.summary())

    def _split_z(self, data_z):
        z0_dim = self.params["z_dims"][0]
        z1_dim = self.params["z_dims"][1]
        z2_dim = self.params["z_dims"][2]
        data_z0 = data_z[:, :z0_dim]
        data_z1 = data_z[:, z0_dim : z0_dim + z1_dim]
        data_z2 = data_z[:, z0_dim + z1_dim : z0_dim + z1_dim + z2_dim]
        return data_z0, data_z1, data_z2

    def _treatment_output(self, data_z, data_w):
        """Evaluate the treatment network.

        Parameters
        ----------
        data_z : tf.Tensor
            Latent variables with shape ``(n, sum(z_dims))``.
        data_w : tf.Tensor
            Instruments with shape ``(n, w_dim)``.

        Returns
        -------
        tf.Tensor
            Raw treatment-network output with shape ``(n, 2)`` where the
            first column is the treatment mean/logit and the second column
            parameterises the variance for continuous treatment.
        """
        data_z0, _, data_z2 = self._split_z(data_z)
        return self.h_net(tf.concat([data_z0, data_z2, data_w], axis=-1))

    def _outcome_output(self, data_z, data_x):
        """Evaluate the structural outcome network ``f(z0, z1, x)``.

        Parameters
        ----------
        data_z : tf.Tensor
            Latent variables with shape ``(n, sum(z_dims))``.
        data_x : tf.Tensor
            Treatments with shape ``(n, 1)``.

        Returns
        -------
        tf.Tensor
            Outcome-network output with shape ``(n, 2)`` where the first
            column is the conditional mean and the second column
            parameterises the variance.
        """
        data_z0, data_z1, _ = self._split_z(data_z)
        return self.f_net(tf.concat([data_z0, data_z1, data_x], axis=-1))

    def _continuous_sigma(self, net_output, sigma_key, eps=1e-6):
        if sigma_key in self.params:
            sigma_square = tf.cast(self.params[sigma_key] ** 2, tf.float32)
            sigma_square = tf.ones_like(net_output[:, :1]) * sigma_square
        else:
            sigma_square = tf.nn.softplus(net_output[:, -1:]) + eps
        return sigma_square

    def _gaussian_nll(self, targets, means, sigma_square, event_dim):
        """Compute per-sample Gaussian negative log-likelihood values.

        Parameters
        ----------
        targets : tf.Tensor
            Observations with shape ``(n, event_dim)``.
        means : tf.Tensor
            Conditional means with shape ``(n, event_dim)``.
        sigma_square : tf.Tensor
            Conditional variances with shape ``(n, 1)``.
        event_dim : int
            Dimensionality of the Gaussian observation.

        Returns
        -------
        tf.Tensor
            Pointwise negative log-likelihood values with shape ``(n,)``.
        """
        sq_error = tf.reduce_sum((targets - means) ** 2, axis=1, keepdims=True)
        nll = sq_error / (2.0 * sigma_square) + 0.5 * event_dim * tf.math.log(
            sigma_square
        )
        return tf.squeeze(nll, axis=1)

    def _treatment_mean(self, data_z, data_w):
        treatment_output = self._treatment_output(data_z, data_w)
        mu_x = treatment_output[:, :1]
        if self.params["binary_treatment"]:
            return tf.sigmoid(mu_x)
        return mu_x

    @tf.function
    def _covariate_neg_log_posterior(self, data_v, data_z, eps=1e-6):
        """Average negative log-posterior for the covariate-only latent update."""
        return -tf.reduce_mean(self.get_log_covariate_posterior(data_v, data_z, eps=eps))

    def infer_latent_from_covariates(
        self,
        data_v,
        method=None,
        map_steps=None,
        map_lr=None,
        initial_z=None,
        n_mcmc=None,
        burn_in=None,
        q_sd=None,
        initial_q_sd=1.0,
        adaptive_sd=None,
        return_posterior=False,
        bs=None,
        verbose=0,
    ):
        """Infer latent variables using the covariate-only posterior ``p(z | v)``.

        This is the latent quantity needed by the structural benchmark
        ``f_struct(x, v)`` where the final prediction must not depend on the
        instrument.

        Parameters
        ----------
        data_v : np.ndarray
            Covariates with shape ``(n, v_dim)``.
        method : {"encoder", "map", "mcmc"} or None, optional
            Latent inference rule. ``None`` uses
            ``params["structural_latent_method"]``.
        map_steps : int or None, optional
            Number of gradient steps for the MAP refinement when
            ``method="map"``.
        map_lr : float or None, optional
            Learning rate for the MAP refinement.
        initial_z : np.ndarray or None, optional
            Optional initialization with shape ``(n, sum(z_dims))``.
            By default the encoder output ``e(v)`` is used.
        n_mcmc : int or None, optional
            Number of retained MCMC samples when ``method="mcmc"``.
        burn_in : int or None, optional
            Number of burn-in iterations for ``method="mcmc"``.
        q_sd : float or None, optional
            Proposal standard deviation for ``method="mcmc"``.
        initial_q_sd : float, default=1.0
            Initial proposal standard deviation when adaptive MCMC is used.
        adaptive_sd : bool or None, optional
            Whether to adapt the proposal scale during burn-in.
        return_posterior : bool, default=False
            If ``True`` and ``method="mcmc"``, return full latent posterior
            draws with shape ``(n_mcmc, n, sum(z_dims))``. Otherwise return a
            point estimate with shape ``(n, sum(z_dims))``.
        bs : int or None, optional
            Batch size for covariate-only MCMC. Ignored for encoder/MAP.
        verbose : int, default=0
            Verbosity level for MAP progress logging.

        Returns
        -------
        np.ndarray
            Encoder/MAP: latent point estimate with shape
            ``(n, sum(z_dims))``. MCMC with ``return_posterior=True``:
            posterior samples with shape ``(n_mcmc, n, sum(z_dims))``.
        """
        data_v = self._to_2d_float32(data_v)
        if method is None:
            method = self.params["structural_latent_method"]

        if initial_z is None:
            initial_z = self.e_net(data_v).numpy()
        else:
            initial_z = self._to_2d_float32(initial_z)

        if method == "encoder":
            return initial_z.astype(np.float32)

        if method == "mcmc":
            if n_mcmc is None:
                n_mcmc = int(self.params["structural_mcmc_n_samples"])
            if burn_in is None:
                burn_in = int(self.params["structural_mcmc_burn_in"])
            if q_sd is None:
                q_sd = float(self.params["structural_mcmc_q_sd"])
            if bs is None:
                bs = int(self.params["structural_mcmc_bs"])

            n_obs = len(data_v)
            bs = max(1, int(bs))
            posterior_batches = []
            latent_mean = np.zeros_like(initial_z, dtype=np.float32)

            for start in range(0, n_obs, bs):
                end = min(start + bs, n_obs)
                batch_v = data_v[start:end]
                batch_initial_z = initial_z[start:end]
                batch_posterior_z = self.metropolis_hastings_covariate_sampler(
                    batch_v,
                    initial_state=batch_initial_z,
                    initial_q_sd=initial_q_sd,
                    q_sd=q_sd,
                    burn_in=burn_in,
                    n_keep=n_mcmc,
                    adaptive_sd=adaptive_sd,
                )
                latent_mean[start:end] = np.mean(batch_posterior_z, axis=0)
                if return_posterior:
                    posterior_batches.append(batch_posterior_z)

            if return_posterior:
                return np.concatenate(posterior_batches, axis=1).astype(np.float32)
            return latent_mean.astype(np.float32)

        if method != "map":
            raise ValueError("`method` must be one of {'encoder', 'map', 'mcmc'}.")

        if map_steps is None:
            map_steps = int(self.params["structural_map_steps"])
        if map_lr is None:
            map_lr = float(self.params["structural_map_lr"])

        if map_steps <= 0:
            return initial_z.astype(np.float32)

        data_v_tf = tf.convert_to_tensor(data_v, dtype=tf.float32)
        data_z = tf.Variable(initial_z.astype(np.float32), trainable=True)
        optimizer = tf.keras.optimizers.Adam(map_lr, beta_1=0.9, beta_2=0.99)

        for step in range(map_steps):
            with tf.GradientTape() as tape:
                loss = self._covariate_neg_log_posterior(data_v_tf, data_z)
            gradients = tape.gradient(loss, [data_z])
            optimizer.apply_gradients(zip(gradients, [data_z]))
            if verbose and ((step + 1) % max(1, map_steps // 5) == 0 or step == 0):
                print(
                    "Structural latent MAP step [%d/%d]: loss_vz = %.4f"
                    % (step + 1, map_steps, float(loss.numpy()))
                )

        return data_z.numpy().astype(np.float32)

    def _run_metropolis_hastings(
        self,
        initial_state,
        log_posterior_fn,
        initial_q_sd=1.0,
        q_sd=None,
        burn_in=5000,
        n_keep=3000,
        target_acceptance_rate=0.25,
        tolerance=0.05,
        adjustment_interval=50,
        adaptive_sd=None,
        window_size=100,
    ):
        """Generic Metropolis-Hastings sampler used by all latent posteriors.

        Parameters
        ----------
        initial_state : np.ndarray
            Initial latent state with shape ``(n, sum(z_dims))``.
        log_posterior_fn : callable
            Function mapping a latent state of shape ``(n, sum(z_dims))`` to
            pointwise log posterior values with shape ``(n,)``.
        initial_q_sd, q_sd, burn_in, n_keep, target_acceptance_rate, tolerance,
        adjustment_interval, adaptive_sd, window_size :
            Standard MH configuration parameters. They have the same meanings as
            in :meth:`metropolis_hastings_sampler`.

        Returns
        -------
        np.ndarray
            Posterior samples with shape ``(n_keep, n, sum(z_dims))``.
        """
        current_state = np.asarray(initial_state, dtype=np.float32).copy()
        samples = []
        counter = 0
        recent_acceptances = []
        n_obs = len(current_state)

        if adaptive_sd is None:
            adaptive_sd = q_sd is None or q_sd <= 0
        if adaptive_sd:
            q_sd = initial_q_sd

        while len(samples) < n_keep:
            proposed_state = current_state + np.random.normal(
                0.0,
                q_sd,
                size=current_state.shape,
            ).astype(np.float32)

            proposed_log_posterior = np.asarray(
                log_posterior_fn(proposed_state), dtype=np.float32
            ).reshape(-1)
            current_log_posterior = np.asarray(
                log_posterior_fn(current_state), dtype=np.float32
            ).reshape(-1)
            if proposed_log_posterior.size != n_obs or current_log_posterior.size != n_obs:
                raise ValueError(
                    "Metropolis-Hastings requires one log-posterior value per "
                    f"observation; got shapes {proposed_log_posterior.shape} and "
                    f"{current_log_posterior.shape} for batch size {n_obs}."
                )
            acceptance_ratio = np.exp(
                np.minimum(proposed_log_posterior - current_log_posterior, 0.0)
            )
            indices = np.random.rand(n_obs) < acceptance_ratio
            current_state[indices] = proposed_state[indices]

            recent_acceptances.append(indices)
            if len(recent_acceptances) > window_size:
                recent_acceptances = recent_acceptances[-window_size:]

            if adaptive_sd and counter < burn_in and counter % adjustment_interval == 0 and counter > 0:
                current_acceptance_rate = np.sum(recent_acceptances) / (
                    len(recent_acceptances) * n_obs
                )
                print(f"Current MCMC Acceptance Rate: {current_acceptance_rate:.4f}")
                if current_acceptance_rate < target_acceptance_rate - tolerance:
                    q_sd *= 0.9
                elif current_acceptance_rate > target_acceptance_rate + tolerance:
                    q_sd *= 1.1
                print(f"MCMC Proposal Standard Deviation (q_sd): {q_sd:.4f}")

            if counter >= burn_in:
                samples.append(current_state.copy())
            counter += 1

        acceptance_rate = np.sum(recent_acceptances) / (
            len(recent_acceptances) * n_obs
        )
        print(f"Final MCMC Acceptance Rate: {acceptance_rate:.4f}")
        return np.array(samples, dtype=np.float32)

    def _sample_treatment(self, data_z, data_w, n_samples=1, use_mean=False, eps=1e-6):
        """Sample treatments from ``p(x | w, z0, z2)``.

        Parameters
        ----------
        data_z : tf.Tensor
            Latent variables with shape ``(n, sum(z_dims))``.
        data_w : tf.Tensor
            Instruments with shape ``(n, w_dim)``.
        n_samples : int, default=1
            Number of Monte Carlo samples per subject.
        use_mean : bool, default=False
            If ``True``, return deterministic mean treatment values instead of
            stochastic samples.

        Returns
        -------
        tuple
            ``(samples, mean_or_prob, variance)`` where ``samples`` has shape
            ``(n_samples, n, 1)``. For binary treatment the third entry is
            ``None``.
        """
        treatment_output = self._treatment_output(data_z, data_w)
        mu_x = treatment_output[:, :1]
        if self.params["binary_treatment"]:
            prob_x = tf.sigmoid(mu_x)
            if use_mean:
                samples = tf.repeat(prob_x[None, :, :], repeats=n_samples, axis=0)
            else:
                uniforms = tf.random.uniform(
                    [n_samples, tf.shape(prob_x)[0], 1], dtype=prob_x.dtype
                )
                samples = tf.cast(uniforms < prob_x[None, :, :], prob_x.dtype)
            return samples, prob_x, None

        sigma_square_x = self._continuous_sigma(
            treatment_output, sigma_key="sigma_x", eps=eps
        )
        if use_mean:
            samples = tf.repeat(mu_x[None, :, :], repeats=n_samples, axis=0)
        else:
            eps_x = tf.random.normal(
                [n_samples, tf.shape(mu_x)[0], 1], dtype=mu_x.dtype
            )
            samples = mu_x[None, :, :] + eps_x * tf.sqrt(sigma_square_x)[None, :, :]
        return samples, mu_x, sigma_square_x

    def _outcome_call_multiplier(self, n_samples):
        if self.params["binary_treatment"]:
            return 2.0
        return float(max(1, n_samples))

    def _outcome_outputs_for_samples(self, data_z, x_samples):
        """Evaluate ``f(z0, z1, x)`` for a stack of treatment samples.

        Parameters
        ----------
        data_z : tf.Tensor
            Latent variables with shape ``(n, sum(z_dims))``.
        x_samples : tf.Tensor
            Treatment samples with shape ``(n_samples, n, 1)``.

        Returns
        -------
        tf.Tensor
            Outcome-network outputs with shape ``(n_samples, n, 2)``.
        """
        z0_dim = self.params["z_dims"][0]
        z1_dim = self.params["z_dims"][1]
        data_z0, data_z1, _ = self._split_z(data_z)
        n_samples = tf.shape(x_samples)[0]
        n_obs = tf.shape(data_z)[0]

        data_z0_tiled = tf.tile(tf.expand_dims(data_z0, axis=0), [n_samples, 1, 1])
        data_z1_tiled = tf.tile(tf.expand_dims(data_z1, axis=0), [n_samples, 1, 1])
        flat_inputs = tf.concat(
            [
                tf.reshape(data_z0_tiled, (-1, z0_dim)),
                tf.reshape(data_z1_tiled, (-1, z1_dim)),
                tf.reshape(x_samples, (-1, 1)),
            ],
            axis=-1,
        )
        flat_outputs = self.f_net(flat_inputs)
        return tf.reshape(flat_outputs, (n_samples, n_obs, 2))

    def _integrated_outcome_log_prob(self, data_z, data_w, data_y, n_samples=None, eps=1e-6):
        """Monte Carlo IV log-likelihood ``log p(y | w, z)``.

        Parameters
        ----------
        data_z : tf.Tensor
            Latent variables with shape ``(n, sum(z_dims))``.
        data_w : tf.Tensor
            Instruments with shape ``(n, w_dim)``.
        data_y : tf.Tensor
            Outcomes with shape ``(n, 1)``.
        n_samples : int or None, optional
            Number of treatment Monte Carlo samples used to approximate the
            law-of-total-probability integral for continuous treatment.

        Returns
        -------
        tf.Tensor
            Pointwise integrated log-likelihood values with shape ``(n,)``.
        """
        if self.params["binary_treatment"]:
            treatment_output = self._treatment_output(data_z, data_w)
            prob_x = tf.sigmoid(treatment_output[:, :1])
            x_one = tf.ones_like(data_y)
            x_zero = tf.zeros_like(data_y)

            y_one_output = self._outcome_output(data_z, x_one)
            y_zero_output = self._outcome_output(data_z, x_zero)

            mu_y_one = y_one_output[:, :1]
            mu_y_zero = y_zero_output[:, :1]
            sigma_square_y_one = self._continuous_sigma(
                y_one_output, sigma_key="sigma_y", eps=eps
            )
            sigma_square_y_zero = self._continuous_sigma(
                y_zero_output, sigma_key="sigma_y", eps=eps
            )

            log_prob_one = -(
                (data_y - mu_y_one) ** 2 / (2.0 * sigma_square_y_one)
                + 0.5 * tf.math.log(sigma_square_y_one)
            )
            log_prob_zero = -(
                (data_y - mu_y_zero) ** 2 / (2.0 * sigma_square_y_zero)
                + 0.5 * tf.math.log(sigma_square_y_zero)
            )

            component_log_prob = tf.stack(
                [
                    tf.squeeze(tf.math.log(prob_x + eps) + log_prob_one, axis=1),
                    tf.squeeze(
                        tf.math.log(1.0 - prob_x + eps) + log_prob_zero, axis=1
                    ),
                ],
                axis=0,
            )
            return tf.reduce_logsumexp(component_log_prob, axis=0)

        if n_samples is None:
            n_samples = int(self.params["iv_mc_samples"])
        x_samples, _, _ = self._sample_treatment(
            data_z, data_w, n_samples=n_samples, use_mean=False, eps=eps
        )
        outcome_outputs = self._outcome_outputs_for_samples(data_z, x_samples)
        mu_y = outcome_outputs[:, :, :1]
        if "sigma_y" in self.params:
            sigma_square_y = tf.cast(self.params["sigma_y"] ** 2, tf.float32)
        else:
            sigma_square_y = tf.nn.softplus(outcome_outputs[:, :, 1:2]) + eps
        data_y = tf.expand_dims(data_y, axis=0)
        log_prob_samples = -(
            (data_y - mu_y) ** 2 / (2.0 * sigma_square_y)
            + 0.5 * tf.math.log(sigma_square_y)
        )
        log_prob_samples = tf.squeeze(log_prob_samples, axis=-1)
        return tf.reduce_logsumexp(log_prob_samples, axis=0) - tf.math.log(
            tf.cast(n_samples, tf.float32)
        )

    def _integrated_outcome_mean(self, data_z, data_w, n_samples=None, sample_y=False, eps=1e-6):
        """Monte Carlo expectation of the IV-integrated outcome model.

        Returns the posterior predictive mean of ``Y`` conditional on ``(W, Z)``
        by integrating over the treatment model.
        """
        if self.params["binary_treatment"]:
            treatment_output = self._treatment_output(data_z, data_w)
            prob_x = tf.sigmoid(treatment_output[:, :1])
            x_one = tf.ones([tf.shape(data_z)[0], 1], dtype=tf.float32)
            x_zero = tf.zeros([tf.shape(data_z)[0], 1], dtype=tf.float32)

            y_one_output = self._outcome_output(data_z, x_one)
            y_zero_output = self._outcome_output(data_z, x_zero)
            mu_y_one = y_one_output[:, :1]
            mu_y_zero = y_zero_output[:, :1]

            if not sample_y:
                return prob_x * mu_y_one + (1.0 - prob_x) * mu_y_zero

            sigma_square_y_one = self._continuous_sigma(
                y_one_output, sigma_key="sigma_y", eps=eps
            )
            sigma_square_y_zero = self._continuous_sigma(
                y_zero_output, sigma_key="sigma_y", eps=eps
            )
            take_one = tf.random.uniform(tf.shape(prob_x), dtype=prob_x.dtype) < prob_x
            chosen_mu = tf.where(take_one, mu_y_one, mu_y_zero)
            chosen_sigma = tf.where(take_one, sigma_square_y_one, sigma_square_y_zero)
            return tf.random.normal(
                tf.shape(chosen_mu),
                mean=chosen_mu,
                stddev=tf.sqrt(chosen_sigma),
            )

        if n_samples is None:
            n_samples = int(self.params["eval_mc_samples"])
        x_samples, _, _ = self._sample_treatment(
            data_z, data_w, n_samples=n_samples, use_mean=False, eps=eps
        )
        outcome_outputs = self._outcome_outputs_for_samples(data_z, x_samples)
        mu_y = outcome_outputs[:, :, :1]
        if not sample_y:
            return tf.reduce_mean(mu_y, axis=0)
        sigma_square_y = self._continuous_sigma(
            tf.reshape(outcome_outputs, (-1, 2)),
            sigma_key="sigma_y",
            eps=eps,
        )
        sigma_square_y = tf.reshape(sigma_square_y, tf.shape(mu_y))
        y_components = tf.random.normal(
            tf.shape(mu_y), mean=mu_y, stddev=tf.sqrt(sigma_square_y)
        )
        return tf.reduce_mean(y_components, axis=0)

    @tf.function
    def update_h_net(self, data_z, data_w, data_x, eps=1e-6):
        """Update the treatment generative model ``p(x | w, z0, z2)``.

        Parameters
        ----------
        data_z : tf.Tensor
            Latent variables with shape ``(batch, sum(z_dims))``.
        data_w : tf.Tensor
            Instruments with shape ``(batch, w_dim)``.
        data_x : tf.Tensor
            Treatments with shape ``(batch, 1)``.

        Returns
        -------
        tuple[tf.Tensor, tf.Tensor]
            Negative log-likelihood loss and mean-squared reconstruction error.
        """
        with tf.GradientTape() as gen_tape:
            treatment_output = self._treatment_output(data_z, data_w)
            mu_x = treatment_output[:, :1]

            if self.params["binary_treatment"]:
                loss_x = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=data_x, logits=mu_x)
                )
                loss_mse = tf.reduce_mean((data_x - tf.sigmoid(mu_x)) ** 2)
            else:
                sigma_square_x = self._continuous_sigma(
                    treatment_output, sigma_key="sigma_x", eps=eps
                )
                loss_mse = tf.reduce_mean((data_x - mu_x) ** 2)
                loss_x = self._gaussian_nll(
                    data_x, mu_x, sigma_square_x, event_dim=1
                )
                loss_x = tf.reduce_mean(loss_x)

            if self.params["use_bnn"]:
                loss_x += sum(self.h_net.losses) * self.params["kl_weight"]

        h_gradients = gen_tape.gradient(loss_x, self.h_net.trainable_variables)
        self.h_optimizer.apply_gradients(zip(h_gradients, self.h_net.trainable_variables))
        return loss_x, loss_mse

    @tf.function
    def update_f_net(self, data_z, data_w, data_y, n_samples=None, eps=1e-6):
        """Update the outcome model using the IV pseudo-likelihood.

        Parameters
        ----------
        data_z : tf.Tensor
            Latent variables with shape ``(batch, sum(z_dims))``.
        data_w : tf.Tensor
            Instruments with shape ``(batch, w_dim)``.
        data_y : tf.Tensor
            Outcomes with shape ``(batch, 1)``.
        n_samples : int or None, optional
            Number of Monte Carlo treatment samples for the integrated
            likelihood.

        Returns
        -------
        tuple[tf.Tensor, tf.Tensor]
            IV pseudo-likelihood loss and integrated-prediction MSE.
        """
        if n_samples is None:
            n_samples = int(self.params["iv_mc_samples"])

        with tf.GradientTape() as gen_tape:
            log_prob = self._integrated_outcome_log_prob(
                data_z, data_w, data_y, n_samples=n_samples, eps=eps
            )
            loss_y = -tf.reduce_mean(log_prob)
            mean_y = self._integrated_outcome_mean(
                data_z, data_w, n_samples=n_samples, sample_y=False, eps=eps
            )
            loss_mse = tf.reduce_mean((data_y - mean_y) ** 2)

            if self.params["use_bnn"]:
                kl_scale = self._outcome_call_multiplier(n_samples)
                loss_y += (sum(self.f_net.losses) / kl_scale) * self.params["kl_weight"]

        f_gradients = gen_tape.gradient(loss_y, self.f_net.trainable_variables)
        self.f_optimizer.apply_gradients(zip(f_gradients, self.f_net.trainable_variables))
        return loss_y, loss_mse

    @tf.function
    def update_latent_variable_sgd(
        self, data_x, data_y, data_v, data_w, batch_idx, include_outcome=True, eps=1e-6
    ):
        """Update subject-specific latent variables by quasi-posterior ascent.

        The optimized objective is based on
        ``p(z) p(v|z) p(x|w,z) p_iv(y|w,z)`` where the outcome term uses the
        IV integrated pseudo-likelihood.
        """
        with tf.GradientTape() as tape:
            data_z = tf.gather(self.data_z, batch_idx, axis=0)

            g_output = self.g_net(data_z)
            mu_v = g_output[:, : self.params["v_dim"]]
            sigma_square_v = self._continuous_sigma(g_output, sigma_key="sigma_v", eps=eps)
            loss_pv_z = self._gaussian_nll(
                data_v,
                mu_v,
                sigma_square_v,
                event_dim=self.params["v_dim"],
            )
            loss_pv_z = tf.reduce_mean(loss_pv_z)

            treatment_output = self._treatment_output(data_z, data_w)
            mu_x = treatment_output[:, :1]
            if self.params["binary_treatment"]:
                loss_px_z = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=data_x, logits=mu_x)
                )
            else:
                sigma_square_x = self._continuous_sigma(
                    treatment_output, sigma_key="sigma_x", eps=eps
                )
                loss_px_z = self._gaussian_nll(
                    data_x, mu_x, sigma_square_x, event_dim=1
                )
                loss_px_z = tf.reduce_mean(loss_px_z)

            if include_outcome:
                loss_py_z = -tf.reduce_mean(
                    self._integrated_outcome_log_prob(
                        data_z,
                        data_w,
                        data_y,
                        n_samples=int(self.params["iv_mc_samples"]),
                        eps=eps,
                    )
                )
            else:
                loss_py_z = tf.constant(0.0, dtype=tf.float32)

            loss_prior_z = tf.reduce_mean(tf.reduce_sum(data_z ** 2, axis=1) / 2.0)
            loss_posterior_z = loss_pv_z + loss_px_z + loss_py_z + loss_prior_z

        posterior_gradients = tape.gradient(loss_posterior_z, [self.data_z])
        self.posterior_optimizer.apply_gradients(
            zip(posterior_gradients, [self.data_z])
        )
        return loss_posterior_z

    @tf.function
    def train_gen_step(self, data_z, data_v, data_w, data_x, data_y):
        """One EGM warm-start step for ``(g, e, h, f)``.

        The warm-start uses deconfounded treatment predictions from the
        treatment model before fitting the outcome network.
        """
        with tf.GradientTape(persistent=True) as gen_tape:
            sigma_square_loss = 0.0
            g_output = self.g_net(data_z)
            data_v_ = g_output[:, : self.params["v_dim"]]
            sigma_square_loss += tf.reduce_mean(tf.square(g_output[:, -1]))

            data_z_ = self.e_net(data_v)
            data_z0, data_z1, data_z2 = self._split_z(data_z_)

            data_z__ = self.e_net(data_v_)
            data_v__ = self.g_net(data_z_)[:, : self.params["v_dim"]]
            data_dz_ = self.dz_net(data_z_)

            l2_loss_v = tf.reduce_mean((data_v - data_v__) ** 2)
            l2_loss_z = tf.reduce_mean((data_z - data_z__) ** 2)
            e_loss_adv = -tf.reduce_mean(data_dz_)

            h_output = self.h_net(tf.concat([data_z0, data_z2, data_w], axis=-1))
            data_x_ = h_output[:, :1]
            if self.params["binary_treatment"]:
                l2_loss_x = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=data_x, logits=data_x_)
                )
                deconfounded_x = tf.sigmoid(data_x_)
            else:
                sigma_square_loss += tf.reduce_mean(tf.square(h_output[:, -1]))
                l2_loss_x = tf.reduce_mean((data_x_ - data_x) ** 2)
                deconfounded_x = data_x_

            f_output = self.f_net(tf.concat([data_z0, data_z1, deconfounded_x], axis=-1))
            data_y_ = f_output[:, :1]
            sigma_square_loss += tf.reduce_mean(tf.square(f_output[:, -1]))
            l2_loss_y = tf.reduce_mean((data_y_ - data_y) ** 2)

            g_e_loss = (
                e_loss_adv
                + (l2_loss_v + self.params["use_z_rec"] * l2_loss_z)
                + l2_loss_x
                + l2_loss_y
                + 0.001 * sigma_square_loss
            )

        trainable_variables = (
            self.g_net.trainable_variables
            + self.e_net.trainable_variables
            + self.f_net.trainable_variables
            + self.h_net.trainable_variables
        )
        g_e_gradients = gen_tape.gradient(g_e_loss, trainable_variables)
        self.g_pre_optimizer.apply_gradients(zip(g_e_gradients, trainable_variables))
        return e_loss_adv, l2_loss_v, l2_loss_z, l2_loss_x, l2_loss_y, g_e_loss

    def egm_init(
        self,
        data,
        egm_n_iter=30000,
        batch_size=32,
        egm_batches_per_eval=500,
        verbose=1,
        evaluation_callback=None,
    ):
        """Run IV-aware EGM initialization on ``(x, y, v, w)`` data."""
        data_x, data_y, data_v, data_w = self._parse_train_data(data)

        print("EGM Initialization Starts ...")
        for batch_iter in range(egm_n_iter + 1):
            for _ in range(self.params["g_d_freq"]):
                batch_idx = np.random.choice(len(data_x), batch_size, replace=False)
                batch_z = self.z_sampler.get_batch(batch_size)
                batch_v = data_v[batch_idx, :]
                dz_loss, d_loss = self.train_disc_step(batch_z, batch_v)

            batch_z = self.z_sampler.get_batch(batch_size)
            batch_idx = np.random.choice(len(data_x), batch_size, replace=False)
            batch_x = data_x[batch_idx, :]
            batch_y = data_y[batch_idx, :]
            batch_v = data_v[batch_idx, :]
            batch_w = data_w[batch_idx, :]
            e_loss_adv, l2_loss_v, l2_loss_z, l2_loss_x, l2_loss_y, g_e_loss = (
                self.train_gen_step(batch_z, batch_v, batch_w, batch_x, batch_y)
            )
            if batch_iter % egm_batches_per_eval == 0:
                loss_contents = (
                    "EGM Initialization Iter [%d] : e_loss_adv [%.4f], l2_loss_v [%.4f], "
                    "l2_loss_z [%.4f], l2_loss_x [%.4f], l2_loss_y [%.4f], g_e_loss [%.4f], "
                    "dz_loss [%.4f], d_loss [%.4f]"
                    % (
                        batch_iter,
                        e_loss_adv,
                        l2_loss_v,
                        l2_loss_z,
                        l2_loss_x,
                        l2_loss_y,
                        g_e_loss,
                        dz_loss,
                        d_loss,
                    )
                )
                if verbose:
                    print(loss_contents)
                causal_pre, _, _, _ = self.evaluate(data=data)
                causal_pre = causal_pre.numpy()
                if self.params["save_res"]:
                    save_data(
                        "{}/causal_pre_egm_init_iter-{}.txt".format(
                            self.save_dir, batch_iter
                        ),
                        causal_pre,
                    )
        _, mse_x, mse_y, mse_v = self.evaluate(data=data, data_z=None)
        record = self._record_training_metrics(
            stage="post_egm",
            epoch=None,
            mse_x=float(mse_x),
            mse_y=float(mse_y),
            mse_v=float(mse_v),
            evaluation_callback=evaluation_callback,
            include_outcome=True,
        )
        if verbose:
            print("Post-EGM encoder metrics:", self._format_training_record(record))
        print("EGM Initialization Ends.")

    def fit(
        self,
        data,
        epochs=100,
        epochs_per_eval=5,
        batch_size=32,
        startoff=0,
        use_egm_init=True,
        egm_n_iter=30000,
        egm_batches_per_eval=500,
        save_format="txt",
        verbose=1,
        first_stage_warmup_epochs=None,
        evaluation_callback=None,
    ):
        """Train ``CausalBGM_IV`` on observed data ``(x, y, v, w)``.

        Parameters
        ----------
        data : tuple of np.ndarray
            Training tuple ``(x, y, v, w)`` with shapes
            ``(n,1)``, ``(n,1)``, ``(n,v_dim)``, and ``(n,w_dim)``.
        epochs, epochs_per_eval, batch_size, startoff, use_egm_init,
        egm_n_iter, egm_batches_per_eval, save_format, verbose :
            Same meanings as in :class:`CausalBGM`.
        first_stage_warmup_epochs : int or None, optional
            Number of epochs to train only ``p(v|z)`` and ``p(x|w,z)`` before
            switching on the IV outcome pseudo-likelihood.
        """
        data_x, data_y, data_v, data_w = self._parse_train_data(data)
        if first_stage_warmup_epochs is None:
            first_stage_warmup_epochs = int(self.params["first_stage_warmup_epochs"])

        if self.params["save_res"]:
            with open("{}/params.txt".format(self.save_dir), "w") as f_params:
                f_params.write(str(self.params))

        if use_egm_init:
            self.training_history = []
            self.egm_init(
                data,
                egm_n_iter=egm_n_iter,
                egm_batches_per_eval=egm_batches_per_eval,
                batch_size=batch_size,
                verbose=verbose,
                evaluation_callback=evaluation_callback,
            )
            print("Initialize latent variables Z with e(V)...")
            data_z_init = self.e_net(data_v)
        else:
            self.training_history = []
            print("Random initialization of latent variables Z...")
            data_z_init = np.random.normal(
                0,
                1,
                size=(len(data_x), sum(self.params["z_dims"])),
            ).astype("float32")

        self.data_z = tf.Variable(data_z_init, name="Latent Variable", trainable=True)

        best_loss = np.inf
        best_state = None
        best_record = None
        selection_metric = str(
            self.params.get("fit_model_selection_metric", "mse_y")
        ).strip()
        restore_best_weights = bool(self.params.get("fit_restore_best_weights", False))
        show_batch_progress = bool(self.params.get("fit_use_progress_bar", False)) and bool(verbose)
        print("Iterative Updating Starts ...")
        for epoch in range(epochs + 1):
            sample_idx = np.random.choice(len(data_x), len(data_x), replace=False)
            include_outcome = epoch >= first_stage_warmup_epochs

            batch_iterator = range(0, len(data_x), batch_size)
            if show_batch_progress:
                batch_iterator = tqdm(
                    batch_iterator,
                    total=int(np.ceil(len(data_x) / batch_size)),
                    desc=f"Epoch {epoch}/{epochs}",
                    unit="batch",
                )
            for i in batch_iterator:
                    batch_idx = sample_idx[i : i + batch_size]
                    batch_z = tf.gather(self.data_z, batch_idx, axis=0)
                    batch_x = data_x[batch_idx, :]
                    batch_y = data_y[batch_idx, :]
                    batch_v = data_v[batch_idx, :]
                    batch_w = data_w[batch_idx, :]

                    loss_v, loss_mse_v = self.update_g_net(batch_z, batch_v)
                    loss_x, loss_mse_x = self.update_h_net(batch_z, batch_w, batch_x)

                    if include_outcome:
                        loss_y, loss_mse_y = self.update_f_net(
                            batch_z,
                            batch_w,
                            batch_y,
                            n_samples=int(self.params["iv_mc_samples"]),
                        )
                    else:
                        loss_y = tf.constant(0.0, dtype=tf.float32)
                        loss_mse_y = tf.constant(0.0, dtype=tf.float32)

                    loss_posterior_z = self.update_latent_variable_sgd(
                        batch_x,
                        batch_y,
                        batch_v,
                        batch_w,
                        batch_idx,
                        include_outcome=include_outcome,
                    )

                    loss_contents = (
                        "loss_px_wz: [%.4f], loss_mse_x: [%.4f], loss_py_iv: [%.4f], "
                        "loss_mse_y: [%.4f], loss_pv_z: [%.4f], loss_mse_v: [%.4f], "
                        "loss_posterior_z: [%.4f]"
                        % (
                            loss_x,
                            loss_mse_x,
                            loss_y,
                            loss_mse_y,
                            loss_v,
                            loss_mse_v,
                            loss_posterior_z,
                        )
                    )
                    if show_batch_progress:
                        batch_iterator.set_postfix_str(loss_contents)

            if epoch % epochs_per_eval == 0:
                causal_pre, mse_x, mse_y, mse_v = self.evaluate(
                    data=data, data_z=self.data_z
                )
                causal_pre = causal_pre.numpy()
                record = self._record_training_metrics(
                    stage="epoch_eval",
                    epoch=epoch,
                    mse_x=float(mse_x),
                    mse_y=float(mse_y),
                    mse_v=float(mse_v),
                    evaluation_callback=evaluation_callback,
                    include_outcome=include_outcome,
                )

                if verbose:
                    print(
                        "Epoch [%d/%d]: %s\n"
                        % (epoch, epochs, self._format_training_record(record))
                    )

                metric_name = selection_metric if selection_metric in record else "mse_y"
                metric_value = float(record[metric_name])

                if (
                    include_outcome
                    and epoch >= startoff
                    and metric_value < best_loss
                ):
                    best_loss = metric_value
                    best_state = self._snapshot_trainable_state()
                    best_record = dict(record)
                    self.best_causal_pre = causal_pre
                    self.best_epoch = epoch
                    self.best_metric_name = metric_name
                    self.best_metric_value = metric_value
                    if self.params["save_model"]:
                        ckpt_save_path = self.ckpt_manager.save(epoch)
                        print(
                            "Saving checkpoint for epoch {} at {}".format(
                                epoch, ckpt_save_path
                            )
                        )

                if self.params["save_res"]:
                    save_data(
                        "{}/causal_pre_at_{}.{}".format(
                            self.save_dir, epoch, save_format
                        ),
                        causal_pre,
                    )
        self.best_training_record = best_record
        if restore_best_weights and best_state is not None:
            self._restore_trainable_state(best_state)
            if hasattr(self, "data_z"):
                self.data_z.assign(self.e_net(data_v))
            print(
                "Restored best weights from epoch %d using %s=%.6f"
                % (
                    int(best_record["epoch"]),
                    self.best_metric_name,
                    self.best_metric_value,
                )
            )
        return self.training_history

    @tf.function
    def evaluate(self, data, data_z=None, nb_intervals=200):
        """Evaluate training reconstruction metrics and causal summaries.

        Parameters
        ----------
        data : tuple of np.ndarray
            Data tuple ``(x, y, v, w)``.
        data_z : tf.Tensor or np.ndarray or None, optional
            Optional latent matrix with shape ``(n, sum(z_dims))``. If omitted,
            the encoder output ``e(v)`` is used.
        nb_intervals : int, default=200
            Number of treatment values used for continuous-treatment ADRF
            summaries.

        Returns
        -------
        tuple
            ``(causal_summary, mse_x, mse_y, mse_v)`` where the first item is
            the ITE vector for binary treatment or ADRF curve for continuous
            treatment.
        """
        data_x, data_y, data_v, data_w = data
        if data_z is None:
            data_z = self.e_net(data_v)

        data_z0, data_z1, _ = self._split_z(data_z)
        data_v_pred = self.g_net(data_z)[:, : self.params["v_dim"]]
        data_x_pred = self._treatment_mean(data_z, data_w)
        data_y_pred = self._integrated_outcome_mean(
            data_z,
            data_w,
            n_samples=int(self.params["eval_mc_samples"]),
            sample_y=False,
        )

        mse_v = tf.reduce_mean((data_v - data_v_pred) ** 2)
        mse_x = tf.reduce_mean((data_x - data_x_pred) ** 2)
        mse_y = tf.reduce_mean((data_y - data_y_pred) ** 2)

        if self.params["binary_treatment"]:
            y_pred_pos = self.f_net(
                tf.concat([data_z0, data_z1, tf.ones((len(data_x), 1))], axis=-1)
            )[:, :1]
            y_pred_neg = self.f_net(
                tf.concat([data_z0, data_z1, tf.zeros((len(data_x), 1))], axis=-1)
            )[:, :1]
            ite_pre = y_pred_pos - y_pred_neg
            return ite_pre, mse_x, mse_y, mse_v

        x_min = tfp.stats.percentile(data_x, 5.0)
        x_max = tfp.stats.percentile(data_x, 95.0)
        x_values = tf.linspace(x_min, x_max, nb_intervals)

        def compute_dose_response(x):
            data_x_tile = tf.cast(tf.fill([tf.shape(data_x)[0], 1], x), tf.float32)
            y_pred = self.f_net(tf.concat([data_z0, data_z1, data_x_tile], axis=-1))[
                :, :1
            ]
            return tf.reduce_mean(y_pred)

        dose_response = tf.map_fn(
            compute_dose_response, x_values, fn_output_signature=tf.float32
        )
        return dose_response, mse_x, mse_y, mse_v

    def predict_structural(
        self,
        data_x,
        data_v,
        data_z=None,
        sample_y=False,
        latent_method=None,
        map_steps=None,
        map_lr=None,
        alpha=0.01,
        return_interval=False,
        n_mcmc=None,
        burn_in=None,
        q_sd=None,
        initial_q_sd=1.0,
        adaptive_sd=None,
        bs=None,
        verbose=0,
    ):
        """Predict the structural outcome using treatment and covariates only.

        This matches the DFIV/DeepIV evaluation target:
        ``f_struct(x, v) = E[Y | do(X=x), V=v]``.

        Parameters
        ----------
        data_x : np.ndarray
            Treatment values with shape ``(n, 1)``.
        data_v : np.ndarray
            Observed covariates with shape ``(n, v_dim)``.
        data_z : np.ndarray or None, optional
            Optional latent point estimate with shape ``(n, sum(z_dims))``.
            If provided, latent inference is skipped.
        sample_y : bool, default=False
            If ``True`` and latent posterior samples are used, include the
            outcome variance head in posterior predictive draws.
        latent_method : {"encoder", "map", "mcmc"} or None, optional
            Latent inference rule used when ``data_z`` is not supplied.
        map_steps, map_lr :
            MAP-specific hyper-parameters.
        alpha : float, default=0.01
            Significance level used for posterior intervals when
            ``latent_method="mcmc"`` and ``return_interval=True``.
        return_interval : bool, default=False
            If ``True`` and ``latent_method="mcmc"``, also return pointwise
            posterior intervals with shape ``(n, 2)``.
        n_mcmc, burn_in, q_sd, initial_q_sd, adaptive_sd, bs :
            MCMC-specific hyper-parameters for the covariate-only latent
            posterior.

        Returns
        -------
        np.ndarray or tuple
            Point prediction with shape ``(n, 1)``. If
            ``return_interval=True`` and ``latent_method="mcmc"``, returns
            ``(mean, interval)`` where ``interval`` has shape ``(n, 2)``.
        """
        data_x = self._to_2d_float32(data_x)
        data_v = self._to_2d_float32(data_v)
        if data_z is None and (latent_method or self.params["structural_latent_method"]) == "mcmc":
            posterior_z = self.infer_latent_from_covariates(
                data_v,
                method="mcmc",
                n_mcmc=n_mcmc,
                burn_in=burn_in,
                q_sd=q_sd,
                initial_q_sd=initial_q_sd,
                adaptive_sd=adaptive_sd,
                return_posterior=True,
                bs=bs,
                verbose=verbose,
            )
            outcome_draws = self.infer_outcomes_from_latent_posterior(
                posterior_z,
                tf.convert_to_tensor(data_x, dtype=tf.float32),
                sample_y=sample_y,
            ).numpy()
            outcome_mean = np.mean(outcome_draws, axis=0).reshape(-1, 1)
            if not return_interval:
                return outcome_mean.astype(np.float32)
            posterior_interval_upper = np.quantile(
                outcome_draws, 1 - alpha / 2, axis=0
            ).reshape(-1)
            posterior_interval_lower = np.quantile(
                outcome_draws, alpha / 2, axis=0
            ).reshape(-1)
            pos_int = np.stack(
                [posterior_interval_lower, posterior_interval_upper], axis=1
            ).astype(np.float32)
            return outcome_mean.astype(np.float32), pos_int

        if data_z is None:
            data_z = self.infer_latent_from_covariates(
                data_v,
                method=latent_method,
                map_steps=map_steps,
                map_lr=map_lr,
                verbose=verbose,
            )
        elif return_interval:
            raise ValueError(
                "`return_interval=True` is only supported with `latent_method='mcmc'`."
            )
        data_z = tf.convert_to_tensor(data_z, dtype=tf.float32)
        data_z0, data_z1, _ = self._split_z(data_z)
        outcome_output = self.f_net(tf.concat([data_z0, data_z1, data_x], axis=-1))
        mu_y = outcome_output[:, :1]
        if not sample_y:
            return mu_y.numpy()
        sigma_square_y = self._continuous_sigma(outcome_output, sigma_key="sigma_y")
        y_draw = tf.random.normal(
            tf.shape(mu_y), mean=mu_y, stddev=tf.sqrt(sigma_square_y)
        )
        return y_draw.numpy()

    def evaluate_structural_mse(
        self,
        data_x,
        data_v,
        data_y_true,
        data_z=None,
        latent_method=None,
        map_steps=None,
        map_lr=None,
        n_mcmc=None,
        burn_in=None,
        q_sd=None,
        initial_q_sd=1.0,
        adaptive_sd=None,
        bs=None,
        verbose=0,
    ):
        """Evaluate structural-function MSE for the DFIV/DeepIV benchmark.

        Parameters
        ----------
        data_x : np.ndarray
            Treatment grid with shape ``(n, 1)``.
        data_v : np.ndarray
            Observed covariates with shape ``(n, v_dim)``.
        data_y_true : np.ndarray
            Ground-truth structural outcomes with shape ``(n, 1)``.
        data_z : np.ndarray or None, optional
            Optional latent point estimate with shape ``(n, sum(z_dims))``.
        latent_method : {"encoder", "map", "mcmc"} or None, optional
            Latent inference rule when ``data_z`` is not provided.
        map_steps, map_lr :
            MAP-specific hyper-parameters.
        n_mcmc, burn_in, q_sd, initial_q_sd, adaptive_sd, bs :
            MCMC-specific hyper-parameters used when
            ``latent_method="mcmc"``.

        Returns
        -------
        float
            Mean squared error of the structural point predictor. When
            ``latent_method="mcmc"``, the point predictor is the posterior
            mean under ``p(z | v)``.
        """
        data_y_true = self._to_2d_float32(data_y_true)
        data_y_pred = self.predict_structural(
            data_x,
            data_v,
            data_z=data_z,
            latent_method=latent_method,
            map_steps=map_steps,
            map_lr=map_lr,
            n_mcmc=n_mcmc,
            burn_in=burn_in,
            q_sd=q_sd,
            initial_q_sd=initial_q_sd,
            adaptive_sd=adaptive_sd,
            bs=bs,
            verbose=verbose,
        )
        return float(np.mean((data_y_true - data_y_pred) ** 2))

    @tf.function
    def infer_outcomes_from_latent_posterior(
        self, data_posterior_z, data_x, sample_y=True, eps=1e-6
    ):
        """Project latent posterior draws to outcome posterior draws.

        Parameters
        ----------
        data_posterior_z : np.ndarray or tf.Tensor
            Latent posterior samples with shape ``(n_samples, n, sum(z_dims))``.
        data_x : tf.Tensor
            Treatment values with shape ``(n, 1)``.
        sample_y : bool, default=True
            Whether to sample from the outcome variance head.

        Returns
        -------
        tf.Tensor
            Outcome draws with shape ``(n_samples, n)``.
        """
        y_out_all = tf.map_fn(
            lambda z: self.f_net(
                tf.concat(
                    [
                        z[:, : self.params["z_dims"][0]],
                        z[
                            :,
                            self.params["z_dims"][0] : sum(
                                self.params["z_dims"][:2]
                            ),
                        ],
                        data_x,
                    ],
                    axis=-1,
                )
            ),
            data_posterior_z,
            fn_output_signature=tf.float32,
        )
        mu_y_all = y_out_all[:, :, 0]
        if "sigma_y" in self.params:
            sigma_square_y = tf.cast(self.params["sigma_y"] ** 2, tf.float32)
        else:
            sigma_square_y = tf.nn.softplus(y_out_all[:, :, 1]) + eps
        if sample_y:
            return tf.random.normal(
                shape=tf.shape(mu_y_all),
                mean=mu_y_all,
                stddev=tf.sqrt(sigma_square_y),
            )
        return mu_y_all

    def predict_outcome(
        self,
        data,
        alpha=0.01,
        n_mcmc=3000,
        burn_in=5000,
        x_values=None,
        q_sd=1.0,
        sample_y=False,
        bs=10000,
    ):
        """Predict observed outcomes from the partial posterior ``p(z | x, v, w)``.

        This is posterior prediction for observed subjects, not the structural
        benchmark used by DFIV/DeepIV.
        """
        assert 0 < alpha < 1, "alpha must lie in (0, 1)."
        data_x, _, data_v, data_w = self._parse_predict_data(data)
        target_x = self._prepare_target_x(data_x, x_values=x_values)

        n_test = len(data_x)
        bs = max(1, int(bs))
        outcome_mean = np.zeros(n_test, dtype=np.float32)
        posterior_interval_upper = np.zeros(n_test, dtype=np.float32)
        posterior_interval_lower = np.zeros(n_test, dtype=np.float32)

        print("MCMC Latent Variable Sampling ...")
        for start in range(0, n_test, bs):
            end = min(start + bs, n_test)
            batch_data = (
                data_x[start:end],
                data_v[start:end],
                data_w[start:end],
            )
            batch_target_x = target_x[start:end]
            batch_posterior_z = self.metropolis_hastings_sampler(
                batch_data, burn_in=burn_in, n_keep=n_mcmc, q_sd=q_sd
            )
            outcome_draws = self.infer_outcomes_from_latent_posterior(
                batch_posterior_z,
                tf.convert_to_tensor(batch_target_x, dtype=tf.float32),
                sample_y=sample_y,
            ).numpy()
            outcome_mean[start:end] = np.mean(outcome_draws, axis=0)
            posterior_interval_upper[start:end] = np.quantile(
                outcome_draws, 1 - alpha / 2, axis=0
            )
            posterior_interval_lower[start:end] = np.quantile(
                outcome_draws, alpha / 2, axis=0
            )

        pos_int = np.stack([posterior_interval_lower, posterior_interval_upper], axis=1)
        return outcome_mean, pos_int

    def predict(
        self,
        data,
        alpha=0.01,
        n_mcmc=3000,
        burn_in=5000,
        x_values=None,
        q_sd=1.0,
        sample_y=True,
        bs=10000,
    ):
        """Estimate causal effects from the partial posterior ``p(z | x, v, w)``.

        Returns ITEs for binary treatment and ADRF values for continuous
        treatment, mirroring :class:`CausalBGM`.
        """
        assert 0 < alpha < 1, "The significance level `alpha` must be in (0, 1)."
        if not self.params["binary_treatment"] and x_values is None:
            raise ValueError(
                "For continuous treatment, `x_values` must be provided."
            )

        if x_values is not None:
            if np.isscalar(x_values):
                x_values = np.array([x_values], dtype=np.float32)
            else:
                x_values = np.asarray(x_values, dtype=np.float32)

        data_x, _, data_v, data_w = self._parse_predict_data(data)
        n_test = len(data_x)
        bs = max(1, int(bs))

        print("MCMC Latent Variable Sampling ...")
        if self.params["binary_treatment"]:
            ite_mean = np.zeros(n_test, dtype=np.float32)
            posterior_interval_upper = np.zeros(n_test, dtype=np.float32)
            posterior_interval_lower = np.zeros(n_test, dtype=np.float32)

            for start in range(0, n_test, bs):
                end = min(start + bs, n_test)
                batch_data = (
                    data_x[start:end],
                    data_v[start:end],
                    data_w[start:end],
                )
                batch_posterior_z = self.metropolis_hastings_sampler(
                    batch_data, burn_in=burn_in, n_keep=n_mcmc, q_sd=q_sd
                )
                causal_effects = self.infer_from_latent_posterior(
                    batch_posterior_z, x_values=x_values, sample_y=sample_y
                ).numpy()
                ite_mean[start:end] = np.mean(causal_effects, axis=0)
                posterior_interval_upper[start:end] = np.quantile(
                    causal_effects, 1 - alpha / 2, axis=0
                )
                posterior_interval_lower[start:end] = np.quantile(
                    causal_effects, alpha / 2, axis=0
                )

            pos_int = np.stack([posterior_interval_lower, posterior_interval_upper], axis=1)
            return ite_mean, pos_int

        adrf_draw_sums = np.zeros((len(x_values), n_mcmc), dtype=np.float32)
        n_seen = 0
        for start in range(0, n_test, bs):
            end = min(start + bs, n_test)
            batch_data = (
                data_x[start:end],
                data_v[start:end],
                data_w[start:end],
            )
            batch_posterior_z = self.metropolis_hastings_sampler(
                batch_data, burn_in=burn_in, n_keep=n_mcmc, q_sd=q_sd
            )
            batch_effects = self.infer_from_latent_posterior(
                batch_posterior_z, x_values=x_values, sample_y=sample_y
            ).numpy()
            batch_n = end - start
            adrf_draw_sums += batch_effects * batch_n
            n_seen += batch_n

        causal_effects = adrf_draw_sums / float(n_seen)
        adrf = np.mean(causal_effects, axis=1)
        posterior_interval_upper = np.quantile(causal_effects, 1 - alpha / 2, axis=1)
        posterior_interval_lower = np.quantile(causal_effects, alpha / 2, axis=1)
        pos_int = np.stack([posterior_interval_lower, posterior_interval_upper], axis=1)
        return adrf, pos_int

    @tf.function
    def get_log_posterior(self, data_x, data_y, data_v, data_w, data_z, eps=1e-6):
        """Log quasi-posterior ``log p(z | x, y, v, w)`` up to a constant.

        Parameters
        ----------
        data_x, data_y : tf.Tensor
            Treatment and outcome tensors with shapes ``(n, 1)``.
        data_v : tf.Tensor
            Covariates with shape ``(n, v_dim)``.
        data_w : tf.Tensor
            Instruments with shape ``(n, w_dim)``.
        data_z : tf.Tensor
            Latent variables with shape ``(n, sum(z_dims))``.

        Returns
        -------
        tf.Tensor
            Pointwise log quasi-posterior values with shape ``(n,)``.
        """
        g_output = self.g_net(data_z)
        mu_v = g_output[:, : self.params["v_dim"]]
        sigma_square_v = self._continuous_sigma(g_output, sigma_key="sigma_v", eps=eps)
        loss_pv_z = self._gaussian_nll(
            data_v,
            mu_v,
            sigma_square_v,
            event_dim=self.params["v_dim"],
        )

        treatment_output = self._treatment_output(data_z, data_w)
        mu_x = treatment_output[:, :1]
        if self.params["binary_treatment"]:
            loss_px_z = tf.squeeze(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=data_x, logits=mu_x)
            )
        else:
            sigma_square_x = self._continuous_sigma(
                treatment_output, sigma_key="sigma_x", eps=eps
            )
            loss_px_z = self._gaussian_nll(data_x, mu_x, sigma_square_x, event_dim=1)

        loss_py_z = -self._integrated_outcome_log_prob(
            data_z,
            data_w,
            data_y,
            n_samples=int(self.params["iv_mc_samples"]),
            eps=eps,
        )
        loss_prior_z = tf.reduce_sum(data_z ** 2, axis=1) / 2.0
        return -(loss_pv_z + loss_px_z + loss_py_z + loss_prior_z)

    @tf.function
    def get_log_partial_posterior(self, data_x, data_v, data_w, data_z, eps=1e-6):
        """Log partial posterior ``log p(z | x, v, w)`` up to a constant.

        This posterior excludes the outcome term and is used for test-time
        latent inference when outcomes are unavailable.
        """
        g_output = self.g_net(data_z)
        mu_v = g_output[:, : self.params["v_dim"]]
        sigma_square_v = self._continuous_sigma(g_output, sigma_key="sigma_v", eps=eps)
        loss_pv_z = self._gaussian_nll(
            data_v,
            mu_v,
            sigma_square_v,
            event_dim=self.params["v_dim"],
        )

        treatment_output = self._treatment_output(data_z, data_w)
        mu_x = treatment_output[:, :1]
        if self.params["binary_treatment"]:
            loss_px_z = tf.squeeze(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=data_x, logits=mu_x)
            )
        else:
            sigma_square_x = self._continuous_sigma(
                treatment_output, sigma_key="sigma_x", eps=eps
            )
            loss_px_z = self._gaussian_nll(data_x, mu_x, sigma_square_x, event_dim=1)

        loss_prior_z = tf.reduce_sum(data_z ** 2, axis=1) / 2.0
        return -(loss_pv_z + loss_px_z + loss_prior_z)

    @tf.function
    def get_log_covariate_posterior(self, data_v, data_z, eps=1e-6):
        """Log covariate-only posterior ``log p(z | v)`` up to a constant.

        Parameters
        ----------
        data_v : tf.Tensor
            Covariates with shape ``(n, v_dim)``.
        data_z : tf.Tensor
            Latent variables with shape ``(n, sum(z_dims))``.

        Returns
        -------
        tf.Tensor
            Pointwise log posterior values proportional to
            ``log p(z) + log p(v | z)`` with shape ``(n,)``.
        """
        g_output = self.g_net(data_z)
        mu_v = g_output[:, : self.params["v_dim"]]
        sigma_square_v = self._continuous_sigma(g_output, sigma_key="sigma_v", eps=eps)
        loss_pv_z = self._gaussian_nll(
            data_v,
            mu_v,
            sigma_square_v,
            event_dim=self.params["v_dim"],
        )
        loss_prior_z = tf.reduce_sum(data_z ** 2, axis=1) / 2.0
        return -(loss_pv_z + loss_prior_z)

    def metropolis_hastings_sampler(
        self,
        data,
        initial_q_sd=1.0,
        q_sd=None,
        burn_in=5000,
        n_keep=3000,
        target_acceptance_rate=0.25,
        tolerance=0.05,
        adjustment_interval=50,
        adaptive_sd=None,
        window_size=100,
        condition_on_outcome=False,
    ):
        """Sample ``p(z | x, v, w)`` or ``p(z | x, y, v, w)`` by MH.

        Parameters
        ----------
        data : tuple
            If ``condition_on_outcome=False``, expects ``(x, v, w)``. Otherwise
            expects ``(x, y, v, w)``.
        initial_q_sd, q_sd, burn_in, n_keep, target_acceptance_rate, tolerance,
        adjustment_interval, adaptive_sd, window_size :
            Standard Metropolis-Hastings configuration parameters.
        condition_on_outcome : bool, default=False
            If ``True``, use the full quasi-posterior. Otherwise use the
            partial posterior that excludes ``y``.

        Returns
        -------
        np.ndarray
            Posterior samples with shape ``(n_keep, n, sum(z_dims))``.
        """
        if condition_on_outcome:
            data_x, data_y, data_v, data_w = self._parse_train_data(data)
        else:
            data_x, data_y, data_v, data_w = self._parse_predict_data(data)

        initial_state = np.random.normal(
            0,
            1,
            size=(len(data_x), sum(self.params["z_dims"])),
        ).astype("float32")
        if condition_on_outcome:
            log_posterior_fn = lambda state: self.get_log_posterior(
                data_x, data_y, data_v, data_w, state
            ).numpy()
        else:
            log_posterior_fn = lambda state: self.get_log_partial_posterior(
                data_x, data_v, data_w, state
            ).numpy()

        return self._run_metropolis_hastings(
            initial_state=initial_state,
            log_posterior_fn=log_posterior_fn,
            initial_q_sd=initial_q_sd,
            q_sd=q_sd,
            burn_in=burn_in,
            n_keep=n_keep,
            target_acceptance_rate=target_acceptance_rate,
            tolerance=tolerance,
            adjustment_interval=adjustment_interval,
            adaptive_sd=adaptive_sd,
            window_size=window_size,
        )

    def metropolis_hastings_covariate_sampler(
        self,
        data_v,
        initial_state=None,
        initial_q_sd=1.0,
        q_sd=None,
        burn_in=5000,
        n_keep=3000,
        target_acceptance_rate=0.25,
        tolerance=0.05,
        adjustment_interval=50,
        adaptive_sd=None,
        window_size=100,
    ):
        """Sample the covariate-only posterior ``p(z | v)`` by MH.

        Parameters
        ----------
        data_v : np.ndarray
            Covariates with shape ``(n, v_dim)``.
        initial_state : np.ndarray or None, optional
            Initial latent state with shape ``(n, sum(z_dims))``. If omitted,
            the encoder output ``e(v)`` is used.
        initial_q_sd, q_sd, burn_in, n_keep, target_acceptance_rate, tolerance,
        adjustment_interval, adaptive_sd, window_size :
            Standard Metropolis-Hastings configuration parameters.

        Returns
        -------
        np.ndarray
            Covariate-only posterior samples with shape
            ``(n_keep, n, sum(z_dims))``.
        """
        data_v = self._to_2d_float32(data_v)
        if initial_state is None:
            initial_state = self.e_net(data_v).numpy()
        else:
            initial_state = self._to_2d_float32(initial_state)

        log_posterior_fn = lambda state: self.get_log_covariate_posterior(
            data_v, state
        ).numpy()
        return self._run_metropolis_hastings(
            initial_state=initial_state,
            log_posterior_fn=log_posterior_fn,
            initial_q_sd=initial_q_sd,
            q_sd=q_sd,
            burn_in=burn_in,
            n_keep=n_keep,
            target_acceptance_rate=target_acceptance_rate,
            tolerance=tolerance,
            adjustment_interval=adjustment_interval,
            adaptive_sd=adaptive_sd,
            window_size=window_size,
        )
