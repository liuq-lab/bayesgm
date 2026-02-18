import datetime
import os

import dateutil.tz
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from bayesgm.datasets import Gaussian_sampler
from bayesgm.utils.data_io import save_data

from ..networks import BaseFullyConnectedNet, BayesianFullyConnectedNet, Discriminator
from .base import CausalBGM

class IdentifiableCausalBGM(CausalBGM):
    """Identifiable CausalBGM using nonlinear ICA theory (iVAE).

    Achieves identifiability under mild conditions by introducing an auxiliary
    variable :math:`U` and conditioning the latent prior on it:
    :math:`Z \\mid U \\sim \\mathcal{N}(\\mu(U), \\sigma^2(U) I)`.

    Inherits from :class:`CausalBGM`.

    Parameters
    ----------
    params : dict
        Same keys as :class:`CausalBGM`, plus optionally:

        - ``'n_segments'`` (int): Number of auxiliary-variable segments
          (default 10).
        - ``'prior_units'`` (list[int]): Hidden-layer sizes for the prior
          network (default ``[64]``).
    timestamp : str or None, optional
        Timestamp string for the run.
    random_seed : int or None, optional
        If provided, sets the global random seed for reproducibility.
    """

    def __init__(self, params, timestamp=None, random_seed=None):
        self.params = params
        self.timestamp = timestamp

        # Set random seed for reproducibility
        if random_seed is not None:
            tf.keras.utils.set_random_seed(random_seed)
            os.environ['TF_DETERMINISTIC_OPS'] = '1'
            tf.config.experimental.enable_op_determinism()

        # iVAE modification: Add default number of segments if not provided
        if 'n_segments' not in self.params:
            self.params['n_segments'] = 10 # Default value for auxiliary variable segments

        z_dim = sum(params['z_dims'])

        # Initialize networks (g, e, f, h)
        if self.params['use_bnn']:
            self.g_net = BayesianFullyConnectedNet(input_dim=z_dim, output_dim=params['v_dim'] + 1,
                                                 model_name='g_net', nb_units=params['g_units'])
            self.e_net = BayesianFullyConnectedNet(input_dim=params['v_dim'], output_dim=z_dim,
                                                 model_name='e_net', nb_units=params['e_units'])
            self.f_net = BayesianFullyConnectedNet(input_dim=params['z_dims'][0] + params['z_dims'][1] + 1,
                                                 output_dim=2, model_name='f_net', nb_units=params['f_units'])
            self.h_net = BayesianFullyConnectedNet(input_dim=params['z_dims'][0] + params['z_dims'][2],
                                                 output_dim=2, model_name='h_net', nb_units=params['h_units'])
            # iVAE modification: Define prior network p(z|u) using BNN
            self.prior_net = BayesianFullyConnectedNet(input_dim=self.params['n_segments'], output_dim=z_dim + 1,
                                                       model_name='prior_net', nb_units=params.get('prior_units', [64])) # Smaller net for prior typically sufficient
        else:
            self.g_net = BaseFullyConnectedNet(input_dim=z_dim, output_dim=params['v_dim'] + 1,
                                               model_name='g_net', nb_units=params['g_units'])
            self.e_net = BaseFullyConnectedNet(input_dim=params['v_dim'], output_dim=z_dim,
                                               model_name='e_net', nb_units=params['e_units'])
            self.f_net = BaseFullyConnectedNet(input_dim=params['z_dims'][0] + params['z_dims'][1] + 1,
                                               output_dim=2, model_name='f_net', nb_units=params['f_units'])
            self.h_net = BaseFullyConnectedNet(input_dim=params['z_dims'][0] + params['z_dims'][2],
                                               output_dim=2, model_name='h_net', nb_units=params['h_units'])
            # iVAE modification: Define prior network p(z|u) using standard NN
            self.prior_net = BaseFullyConnectedNet(input_dim=self.params['n_segments'], output_dim=z_dim + 1,
                                                   model_name='prior_net', nb_units=params.get('prior_units', [64]))

        self.dz_net = Discriminator(input_dim=z_dim, model_name='dz_net',
                                    nb_units=params['dz_units'])

        # Optimizers for pre-training and main training phase
        self.g_pre_optimizer = tf.keras.optimizers.Adam(params['lr'], beta_1=0.9, beta_2=0.99)
        self.d_pre_optimizer = tf.keras.optimizers.Adam(params['lr'], beta_1=0.9, beta_2=0.99)
        self.z_sampler = Gaussian_sampler(mean=np.zeros(z_dim), sd=1.0)

        self.g_optimizer = tf.keras.optimizers.Adam(params['lr_theta'], beta_1=0.9, beta_2=0.99)
        self.f_optimizer = tf.keras.optimizers.Adam(params['lr_theta'], beta_1=0.9, beta_2=0.99)
        self.h_optimizer = tf.keras.optimizers.Adam(params['lr_theta'], beta_1=0.9, beta_2=0.99)
        self.posterior_optimizer = tf.keras.optimizers.Adam(params['lr_z'], beta_1=0.9, beta_2=0.99)

        # iVAE modification: Add optimizer for the prior network parameters
        self.prior_optimizer = tf.keras.optimizers.Adam(params['lr_theta'], beta_1=0.9, beta_2=0.99)

        self.initialize_nets()

        # Checkpoint and results saving setup
        if self.timestamp is None:
            now = datetime.datetime.now(dateutil.tz.tzlocal())
            self.timestamp = now.strftime('%Y%m%d_%H%M%S')

        self.checkpoint_path = "{}/checkpoints/{}/{}".format(
            params['output_dir'], params['dataset'], self.timestamp)

        if self.params['save_model'] and not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

        self.save_dir = "{}/results/{}/{}".format(
            params['output_dir'], params['dataset'], self.timestamp)

        if self.params['save_res'] and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.ckpt = tf.train.Checkpoint(g_net=self.g_net,
                                        e_net=self.e_net,
                                        f_net=self.f_net,
                                        h_net=self.h_net,
                                        dz_net=self.dz_net,
                                        prior_net=self.prior_net, # iVAE modification: Add prior_net to checkpoint
                                        g_pre_optimizer=self.g_pre_optimizer,
                                        d_pre_optimizer=self.d_pre_optimizer,
                                        g_optimizer=self.g_optimizer,
                                        f_optimizer=self.f_optimizer,
                                        h_optimizer=self.h_optimizer,
                                        posterior_optimizer=self.posterior_optimizer,
                                        prior_optimizer=self.prior_optimizer) # iVAE modification: Add prior_optimizer to checkpoint

        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_path, max_to_keep=5)

        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print('Latest checkpoint restored!!')

    def initialize_nets(self, print_summary=False):
        """Initialize all the networks in IdentifiableCausalBGM."""
        self.g_net(np.zeros((1, sum(self.params['z_dims']))))
        self.f_net(np.zeros((1, self.params['z_dims'][0] + self.params['z_dims'][1] + 1)))
        self.h_net(np.zeros((1, self.params['z_dims'][0] + self.params['z_dims'][2])))
        self.prior_net(np.zeros((1, self.params['n_segments'])))

        if print_summary:
            print(self.g_net.summary())
            print(self.f_net.summary())
            print(self.h_net.summary())
            print(self.prior_net.summary()) # iVAE modification

    # iVAE modification: Update posterior of latent variables Z and prior network parameters
    @tf.function
    def update_latent_variable_sgd(self, data_x, data_y, data_v, data_z, data_u, eps=1e-6):
        with tf.GradientTape(persistent=True) as tape: # persistent=True to calculate multiple gradients

            data_z0 = data_z[:,:self.params['z_dims'][0]]
            data_z1 = data_z[:,self.params['z_dims'][0]:sum(self.params['z_dims'][:2])]
            data_z2 = data_z[:,sum(self.params['z_dims'][:2]):sum(self.params['z_dims'][:3])]

            # logp(v|z) for covariate model
            mu_v = self.g_net(data_z)[:,:self.params['v_dim']]
            if 'sigma_v' in self.params:
                sigma_square_v = self.params['sigma_v']**2
            else:
                sigma_square_v = tf.nn.softplus(self.g_net(data_z)[:,-1]) + eps

            loss_pv_z = tf.reduce_sum((data_v - mu_v)**2, axis=1)/(2*sigma_square_v) + \
                        self.params['v_dim'] * tf.math.log(sigma_square_v)/2
            loss_pv_z = tf.reduce_mean(loss_pv_z)

            # log(x|z) for treatment model
            mu_x = self.h_net(tf.concat([data_z0, data_z2], axis=-1))[:,:1]
            if 'sigma_x' in self.params:
                sigma_square_x = self.params['sigma_x']**2
            else:
                sigma_square_x = tf.nn.softplus(self.h_net(tf.concat([data_z0, data_z2], axis=-1))[:,-1]) + eps

            if self.params['binary_treatment']:
                loss_px_z = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=data_x,
                                                                                  logits=mu_x))
            else:
                loss_px_z = tf.reduce_sum((data_x - mu_x)**2, axis=1)/(2*sigma_square_x) + \
                            tf.math.log(sigma_square_x)/2
                loss_px_z = tf.reduce_mean(loss_px_z)

            # log(y|z,x) for outcome model
            mu_y = self.f_net(tf.concat([data_z0, data_z1, data_x], axis=-1))[:,:1]
            if 'sigma_y' in self.params:
                sigma_square_y = self.params['sigma_y']**2
            else:
                sigma_square_y = tf.nn.softplus(self.f_net(tf.concat([data_z0, data_z1, data_x], axis=-1))[:,-1]) + eps

            loss_py_zx = tf.reduce_sum((data_y - mu_y)**2, axis=1)/(2*sigma_square_y) + \
                         tf.math.log(sigma_square_y)/2
            loss_py_zx = tf.reduce_mean(loss_py_zx)

            # iVAE modification: Replace standard prior loss with conditional prior loss -log p(z|u)
            # Original prior loss: loss_prior_z =  tf.reduce_sum(data_z**2, axis=1)/2
            #                      loss_prior_z = tf.reduce_mean(loss_prior_z)

            # Calculate prior parameters P(Z|U) = N(mu(U), sigma^2(U)I)
            prior_output = self.prior_net(data_u)
            mu_z_prior = prior_output[:, :sum(self.params['z_dims'])]
            # Use scalar variance for all dimensions of z for simplicity
            sigma_square_z_prior = tf.nn.softplus(prior_output[:, -1:]) + eps # Shape (batch_size, 1)

            # Calculate negative log-likelihood for the conditional prior: -log P(Z|U)
            dim_z = sum(self.params['z_dims'])
            # Note: tf.squeeze converts shape (batch_size, 1) to (batch_size,) for element-wise division.
            loss_term1 = tf.reduce_sum((data_z - mu_z_prior)**2, axis=1) / (2.0 * tf.squeeze(sigma_square_z_prior))
            loss_term2 = dim_z * tf.math.log(tf.squeeze(sigma_square_z_prior)) / 2.0
            loss_prior_z = tf.reduce_mean(loss_term1 + loss_term2)

            if self.params['use_bnn']:
                loss_kl_prior = sum(self.prior_net.losses)
                loss_prior_z += loss_kl_prior * self.params.get('kl_weight', 1.0) # Add KL divergence for BNN prior network

            loss_postrior_z = loss_pv_z + loss_px_z + loss_py_zx + loss_prior_z

        # Calculate gradients for Z (E-step)
        posterior_gradients = tape.gradient(loss_postrior_z, [data_z])
        # Apply gradients to update Z
        self.posterior_optimizer.apply_gradients(zip(posterior_gradients, [data_z]))

        # Calculate gradients for prior network parameters (M-step for prior)
        prior_net_gradients = tape.gradient(loss_postrior_z, self.prior_net.trainable_variables)
        # Apply gradients to update prior network
        self.prior_optimizer.apply_gradients(zip(prior_net_gradients, self.prior_net.trainable_variables))
        
        del tape # release tape resources

        return loss_postrior_z

    def fit(self, data,
            batch_size=32, epochs=100, epochs_per_eval=5, startoff=0,
            use_egm_init=True, egm_n_iter=30000, egm_batches_per_eval=500,
            verbose=1, save_format='txt'):
        """Train the IdentifiableCausalBGM model on observed data.

        The training procedure consists of two phases:

        1. **EGM initialization** (optional) — warm-start by jointly training
           encoder and generator with adversarial losses to obtain a good
           starting point for the latent variables and model parameters.
           This phase is optional and can be skipped by setting
           ``use_egm_init`` to ``False``.
        2. **Iterative optimization** — generates an auxiliary variable
           :math:`U` internally and jointly optimizes latent variables,
           network parameters, and the conditional prior network.

        Parameters
        ----------
        data : tuple of np.ndarray
            A triplet ``(data_x, data_y, data_v)``.
        batch_size : int, default=32
            Mini-batch size.
        epochs : int, default=100
            Number of training epochs.
        epochs_per_eval : int, default=5
            Evaluate every this many epochs.
        startoff : int, default=0
            Only start tracking the best model after this many epochs.
        use_egm_init : bool, default=True
            Whether to run EGM initialization before iterative training.
        egm_n_iter : int, default=30000
            Number of EGM initialization iterations.
        egm_batches_per_eval : int, default=500
            Evaluate EGM every this many iterations.
        verbose : int, default=1
            Verbosity level.
        save_format : str, default='txt'
            File format for saving causal estimates.
        """

        data_x, data_y, data_v = data
        n_samples = len(data_x)

        # iVAE modification: Generate auxiliary variable U
        print(f"Generating auxiliary variable U for {self.params['n_segments']} segments.")
        n_segments = self.params['n_segments']
        segment_indices = np.random.randint(0, n_segments, size=n_samples)
        data_u = tf.keras.utils.to_categorical(segment_indices, num_classes=n_segments).astype('float32')

        if self.params['save_res']:
            f_params = open('{}/params.txt'.format(self.save_dir),'w')
            f_params.write(str(self.params))
            f_params.close()

        if use_egm_init:
            self.egm_init(data, egm_n_iter=egm_n_iter, egm_batches_per_eval=egm_batches_per_eval, batch_size=batch_size, verbose=verbose)
            print('Initialize latent variables Z with e(V)...')
            data_z_init = self.e_net(data_v)
        else:
            print('Random initialization of latent variables Z...')
            data_z_init = np.random.normal(0, 1, size=(n_samples, sum(self.params['z_dims']))).astype('float32')

        self.data_z = tf.Variable(data_z_init, name="Latent Variable", trainable=True)

        best_loss = np.inf
        print('Iterative Updating Starts ...')
        for epoch in range(epochs + 1):
            sample_idx = np.random.choice(n_samples, n_samples, replace=False)

            with tqdm(total=n_samples // batch_size, desc=f"Epoch {epoch}/{epochs}", unit="batch") as batch_bar:
                for i in range(0, n_samples - batch_size + 1, batch_size): # Skip the incomplete last batch
                    batch_idx = sample_idx[i:i + batch_size]
                    # Update model parameters of G, H, F with SGD
                    batch_z = tf.Variable(tf.gather(self.data_z, batch_idx, axis=0), name='batch_z', trainable=True)
                    batch_x = data_x[batch_idx, :]
                    batch_y = data_y[batch_idx, :]
                    batch_v = data_v[batch_idx, :]
                    batch_u = data_u[batch_idx, :] # iVAE modification: get batch for U

                    loss_v, loss_mse_v = self.update_g_net(batch_z, batch_v)
                    loss_x, loss_mse_x = self.update_h_net(batch_z, batch_x)
                    loss_y, loss_mse_y = self.update_f_net(batch_z, batch_x, batch_y)

                    # Update Z by maximizing a posterior or posterior mean, and update prior network parameters
                    loss_postrior_z = self.update_latent_variable_sgd(batch_x, batch_y, batch_v, batch_z, batch_u)

                    # Update data_z with updated batch_z
                    self.data_z.scatter_nd_update(
                        indices=tf.expand_dims(batch_idx, axis=1),
                        updates=batch_z
                    )

                    loss_contents = (
                        'loss_px_z: [%.4f], loss_mse_x: [%.4f], loss_py_z: [%.4f], '
                        'loss_mse_y: [%.4f], loss_pv_z: [%.4f], loss_mse_v: [%.4f], loss_postrior_z: [%.4f]'
                        % (loss_x, loss_mse_x, loss_y, loss_mse_y, loss_v, loss_mse_v, loss_postrior_z)
                    )
                    batch_bar.set_postfix_str(loss_contents)
                    batch_bar.update(1)

            if epoch % epochs_per_eval == 0:
                causal_pre, mse_x, mse_y, mse_v, data_x_pred, data_y_pred, data_v_pred = self.evaluate(data = data, data_z = self.data_z)
                causal_pre = causal_pre.numpy()

                if verbose:
                    print('Epoch [%d/%d]: MSE_x: %.4f, MSE_y: %.4f, MSE_v: %.4f\n' % (epoch, epochs, mse_x, mse_y, mse_v))

                if epoch >= startoff and mse_y < best_loss:
                    best_loss = mse_y
                    self.best_causal_pre = causal_pre
                    self.best_epoch = epoch
                    if self.params['save_model']:
                        ckpt_save_path = self.ckpt_manager.save(epoch)
                        print('Saving checkpoint for epoch {} at {}'.format(epoch, ckpt_save_path))
                if self.params['save_res']:
                    save_data('{}/causal_pre_at_{}.{}'.format(self.save_dir, epoch, save_format), causal_pre)

    def predict(self, data, alpha=0.01, n_mcmc=3000, x_values=None, q_sd=1.0, sample_y=True, bs=100):
        """Predict causal effects with posterior uncertainty via MCMC.

        Same interface as :meth:`CausalBGM.predict`.
        Internally generates a fresh auxiliary variable :math:`U` for the
        conditional prior during MCMC sampling.

        Parameters
        ----------
        data : tuple of np.ndarray
            A triplet ``(data_x, data_y, data_v)``.
        alpha : float, default=0.01
            Significance level for posterior intervals.
        n_mcmc : int, default=3000
            Number of posterior MCMC samples.
        x_values : array-like or None
            Treatment values for dose-response (continuous treatment).
        q_sd : float, default=1.0
            Proposal standard deviation for Metropolis-Hastings.
        sample_y : bool, default=True
            Whether to sample from the outcome variance model.
        bs : int, default=100
            Batch size for processing posterior samples.

        Returns
        -------
        effect : np.ndarray
            ITE (binary) or ADRF (continuous) point estimates.
        pos_int : np.ndarray
            Posterior intervals with shape ``(n, 2)`` or
            ``(len(x_values), 2)``.
        """
        assert 0 < alpha < 1, "The significance level 'alpha' must be greater than 0 and less than 1."

        if not self.params['binary_treatment']:
            if x_values is None:
                raise ValueError("For continuous treatment, 'x_values' must not be None.")

        if x_values is not None:
            if np.isscalar(x_values):
                x_values = np.array([x_values], dtype=float)
            else:
                x_values = np.array(x_values, dtype=float)

        causal_effects = []
        print('MCMC Latent Variable Sampling ...')
        # iVAE modification: Pass data to MCMC sampler to generate internal data_u
        data_posterior_z, data_u_mcmc = self.metropolis_hastings_sampler(data, n_keep=n_mcmc, q_sd=q_sd)

        # Iterate over the data_posterior_z in batches
        for i in range(0, data_posterior_z.shape[0], bs):
            batch_posterior_z = data_posterior_z[i:i + bs]
            # No need to pass data_u here as infer_from_latent_posterior only depends on Z, not U directly.
            # The influence of U is already captured in the sampled posterior of Z.
            causal_effect_batch = self.infer_from_latent_posterior(batch_posterior_z, x_values=x_values, sample_y=sample_y).numpy()
            causal_effects.append(causal_effect_batch)

        if self.params['binary_treatment']:
            causal_effects = np.concatenate(causal_effects, axis=0)
            ITE = np.mean(causal_effects, axis=0)
            posterior_interval_upper = np.quantile(causal_effects, 1-alpha/2, axis=0)
            posterior_interval_lower = np.quantile(causal_effects, alpha/2, axis=0)
            pos_int = np.stack([posterior_interval_lower, posterior_interval_upper], axis=1)
            return ITE, pos_int
        else:
            causal_effects = np.concatenate(causal_effects, axis=1)
            ADRF = np.mean(causal_effects, axis=1)
            posterior_interval_upper = np.quantile(causal_effects, 1-alpha/2, axis=1)
            posterior_interval_lower = np.quantile(causal_effects, alpha/2, axis=1)
            pos_int = np.stack([posterior_interval_lower, posterior_interval_upper], axis=1)
            return ADRF, pos_int

    # infer_from_latent_posterior function remains unchanged.
    # It calculates E[Y|do(x), z] = f(z0, z1, x). It doesn't need U because Z already contains all necessary information from U.
    @tf.function
    def infer_from_latent_posterior(self, data_posterior_z, x_values=None, sample_y=True, eps=1e-6):
        # ... function body as in original code ...
        data_z0 = data_posterior_z[:,:,:self.params['z_dims'][0]]
        data_z1 = data_posterior_z[:,:,self.params['z_dims'][0]:sum(self.params['z_dims'][:2])]
        data_z2 = data_posterior_z[:,:,sum(self.params['z_dims'][:2]):sum(self.params['z_dims'][:3])]

        if self.params['binary_treatment']:
            y_out_pos_all = tf.map_fn(
                lambda z: self.f_net(tf.concat([z[:, :self.params['z_dims'][0]],
                                                z[:, self.params['z_dims'][0]:sum(self.params['z_dims'][:2])],
                                                tf.ones([tf.shape(z)[0], 1])], axis=-1)),
                data_posterior_z,
                fn_output_signature=tf.float32
            )
            mu_y_pos_all = y_out_pos_all[:,:,0]
            if 'sigma_y' in self.params:
                sigma_square_y_pos = self.params['sigma_y']**2
            else:
                sigma_square_y_pos = tf.nn.softplus(y_out_pos_all[:,:,1]) + eps

            if sample_y:
                y_pred_pos_all = tf.random.normal(
                    shape=tf.shape(mu_y_pos_all), mean=mu_y_pos_all, stddev=tf.sqrt(sigma_square_y_pos)
                )
            else:
                y_pred_pos_all = mu_y_pos_all

            y_out_neg_all = tf.map_fn(
                lambda z: self.f_net(tf.concat([z[:, :self.params['z_dims'][0]],
                                                z[:, self.params['z_dims'][0]:sum(self.params['z_dims'][:2])],
                                                tf.zeros([tf.shape(z)[0], 1])], axis=-1)),
                data_posterior_z,
                fn_output_signature=tf.float32
            )
            mu_y_neg_all = y_out_neg_all[:,:,0]
            if 'sigma_y' in self.params:
                sigma_square_y_neg = self.params['sigma_y']**2
            else:
                sigma_square_y_neg = tf.nn.softplus(y_out_neg_all[:,:,1]) + eps

            if sample_y:
                y_pred_neg_all = tf.random.normal(
                    shape=tf.shape(mu_y_neg_all), mean=mu_y_neg_all, stddev=tf.sqrt(sigma_square_y_neg)
                )
            else:
                y_pred_neg_all = mu_y_neg_all

            ite_pred_all = y_pred_pos_all-y_pred_neg_all
            return ite_pred_all
        else:
            def compute_dose_response(x):
                data_x = tf.fill([tf.shape(data_posterior_z)[1], 1], x)
                data_x = tf.cast(data_x, tf.float32)
                y_out_all = tf.map_fn(
                    lambda z: self.f_net(tf.concat([z[:, :self.params['z_dims'][0]],
                                                    z[:, self.params['z_dims'][0]:sum(self.params['z_dims'][:2])],
                                                    data_x],axis=-1)),
                    data_posterior_z,
                    fn_output_signature=tf.float32
                )
                mu_y_all = y_out_all[:,:,0]
                if 'sigma_y' in self.params:
                    sigma_square_y = self.params['sigma_y']**2
                else:
                    sigma_square_y = tf.nn.softplus(y_out_all[:,:,1]) + eps

                if sample_y:
                    y_pred_all = tf.random.normal(
                        shape=tf.shape(mu_y_all), mean=mu_y_all, stddev=tf.sqrt(sigma_square_y)
                    )
                else:
                    y_pred_all = mu_y_all

                return tf.reduce_mean(y_pred_all, axis=1)

            dose_response = tf.map_fn(compute_dose_response, x_values, fn_output_signature=tf.float32)
            return dose_response

    # iVAE modification: Update get_log_posterior to accept data_u and calculate conditional prior likelihood
    @tf.function
    def get_log_posterior(self, data_x, data_y, data_v, data_z, data_u, eps=1e-6):
        """ Calculate log posterior log p(z|x,y,v,u) ~ log p(x,y,v|z) + log p(z|u) """
        data_z0 = data_z[:,:self.params['z_dims'][0]]
        data_z1 = data_z[:,self.params['z_dims'][0]:sum(self.params['z_dims'][:2])]
        data_z2 = data_z[:,sum(self.params['z_dims'][:2]):sum(self.params['z_dims'][:3])]

        # Likelihood term: log p(v|z) calculation (as negative loss)
        mu_v = self.g_net(data_z)[:,:self.params['v_dim']]
        if 'sigma_v' in self.params:
            sigma_square_v = self.params['sigma_v']**2
        else:
            sigma_square_v = tf.nn.softplus(self.g_net(data_z)[:,-1]) + eps
        loss_pv_z = tf.reduce_sum((data_v - mu_v)**2, axis=1)/(2*sigma_square_v) + \
                    self.params['v_dim'] * tf.math.log(sigma_square_v)/2

        # Likelihood term: log p(x|z) calculation (as negative loss)
        mu_x = self.h_net(tf.concat([data_z0, data_z2], axis=-1))[:,:1]
        if 'sigma_x' in self.params:
            sigma_square_x = self.params['sigma_x']**2
        else:
            sigma_square_x = tf.nn.softplus(self.h_net(tf.concat([data_z0, data_z2], axis=-1))[:,-1]) + eps

        if self.params['binary_treatment']:
            loss_px_z = tf.squeeze(tf.nn.sigmoid_cross_entropy_with_logits(labels=data_x,logits=mu_x))
        else:
            loss_px_z = tf.reduce_sum((data_x - mu_x)**2, axis=1)/(2*sigma_square_x) + \
                        tf.math.log(sigma_square_x)/2

        # Likelihood term: log p(y|z,x) calculation (as negative loss)
        mu_y = self.f_net(tf.concat([data_z0, data_z1, data_x], axis=-1))[:,:1]
        if 'sigma_y' in self.params:
            sigma_square_y = self.params['sigma_y']**2
        else:
            sigma_square_y = tf.nn.softplus(self.f_net(tf.concat([data_z0, data_z1, data_x], axis=-1))[:,-1]) + eps
        loss_py_zx = tf.reduce_sum((data_y - mu_y)**2, axis=1)/(2*sigma_square_y) + \
                     tf.math.log(sigma_square_y)/2

        # iVAE modification: Conditional prior term log p(z|u) calculation (as negative loss)
        # Original: loss_prior_z =  tf.reduce_sum(data_z**2, axis=1)/2
        prior_output = self.prior_net(data_u)
        mu_z_prior = prior_output[:, :sum(self.params['z_dims'])]
        sigma_square_z_prior = tf.nn.softplus(prior_output[:, -1:]) + eps # Shape (batch_size, 1)

        dim_z = sum(self.params['z_dims'])
        loss_term1 = tf.reduce_sum((data_z - mu_z_prior)**2, axis=1) / (2.0 * tf.squeeze(sigma_square_z_prior))
        loss_term2 = dim_z * tf.math.log(tf.squeeze(sigma_square_z_prior)) / 2.0
        loss_prior_z = loss_term1 + loss_term2
        # Note: We ignore BNN KL loss here as MCMC samples network parameters implicitly if BNN is used.

        loss_postrior_z = loss_pv_z + loss_px_z + loss_py_zx + loss_prior_z
        log_posterior = -loss_postrior_z
        return log_posterior

    # iVAE modification: Update MCMC sampler to generate and use data_u
    def metropolis_hastings_sampler(self, data, initial_q_sd=1.0, q_sd=None, burn_in=5000, n_keep=3000, target_acceptance_rate=0.25, tolerance=0.05, adjustment_interval=50, adaptive_sd=None, window_size=100):
        data_x, data_y, data_v = data
        n_samples = len(data_x)

        # iVAE modification: Generate auxiliary variable U for MCMC sampling.
        # Use the same logic as in fit() to ensure consistency.
        n_segments = self.params['n_segments']
        # Note: For test set prediction, ideally we should re-use segment assignments if known,
        # or randomly assign again. Random assignment here follows the spirit of treating U as noise.
        segment_indices = np.random.randint(0, n_segments, size=n_samples)
        data_u = tf.keras.utils.to_categorical(segment_indices, num_classes=n_segments).astype('float32')

        # Initialize the state of n chains
        current_state = np.random.normal(0, 1, size=(n_samples, sum(self.params['z_dims']))).astype('float32')

        samples = []
        counter = 0
        recent_acceptances = []

        if adaptive_sd is None:
            adaptive_sd = (q_sd is None or q_sd <= 0)
        if adaptive_sd:
            q_sd = initial_q_sd

        while len(samples) < n_keep:
            proposed_state = current_state + np.random.normal(0, q_sd, size=(n_samples, sum(self.params['z_dims']))).astype('float32')

            # iVAE modification: Pass data_u to get_log_posterior
            proposed_log_posterior = self.get_log_posterior(data_x, data_y, data_v, proposed_state, data_u)
            current_log_posterior = self.get_log_posterior(data_x, data_y, data_v, current_state, data_u)

            acceptance_ratio = np.exp(np.minimum(proposed_log_posterior - current_log_posterior, 0))
            indices = np.random.rand(n_samples) < acceptance_ratio
            current_state[indices] = proposed_state[indices]

            # Acceptance rate tracking and adaptation logic...
            recent_acceptances.append(indices)
            if len(recent_acceptances) > window_size:
                recent_acceptances = recent_acceptances[-window_size:]

            if adaptive_sd and counter < burn_in and counter % adjustment_interval == 0 and counter > 0:
                current_acceptance_rate = np.sum(recent_acceptances) / (len(recent_acceptances) * n_samples)
                print(f"Current MCMC Acceptance Rate: {current_acceptance_rate:.4f}")
                if current_acceptance_rate < target_acceptance_rate - tolerance:
                    q_sd *= 0.9
                elif current_acceptance_rate > target_acceptance_rate + tolerance:
                    q_sd *= 1.1
                # print(f"MCMC Proposal Standard Deviation (q_sd): {q_sd:.4f}") # Optional: for debugging

            if counter >= burn_in:
                samples.append(current_state.copy())

            counter += 1

        acceptance_rate = np.sum(recent_acceptances) / (len(recent_acceptances) * n_samples)
        print(f"Final MCMC Acceptance Rate: {acceptance_rate:.4f}")
        # Return samples and the corresponding data_u used for sampling (though data_u might not be needed by caller)
        return np.array(samples), data_u
