import datetime
import os

import dateutil.tz
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from bayesgm.datasets import Gaussian_sampler

from ..networks import (
    BaseFullyConnectedNet,
    BayesianFullyConnectedNet,
    Discriminator,
    MCMCFullyConnectedNet,
    run_mcmc_for_net,
)
from .base import CausalBGM

class FullMCMCCausalBGM(CausalBGM):
    """CausalBGM with full MCMC sampling for both individual latent variables
    and neural-network parameters.

    After calling :meth:`fit` (which uses SGD for both network weights and
    latent variables), invoke :meth:`run_mcmc_training` to draw posterior
    samples of all network weights via Hamiltonian Monte Carlo.  The
    :meth:`predict` method then marginalises over *both* latent-variable and
    weight uncertainty.

    Inherits from :class:`CausalBGM`.

    Parameters
    ----------
    params : dict
        Same keys as :class:`CausalBGM`.
    timestamp : str or None, optional
        Timestamp string for the run.
    random_seed : int or None, optional
        If provided, sets the global random seed for reproducibility.
    """

    def __init__(self, params, timestamp=None, random_seed=None):
        self.params = params
        self.timestamp = timestamp
        if random_seed is not None:
            tf.keras.utils.set_random_seed(random_seed)
            os.environ['TF_DETERMINISTIC_OPS'] = '1'
            tf.config.experimental.enable_op_determinism()
        if self.params['use_bnn']:
            self.g_net = MCMCFullyConnectedNet(input_dim=sum(params['z_dims']),output_dim = params['v_dim']+1, 
                                           model_name='g_net', nb_units=params['g_units'])
            self.e_net = BayesianFullyConnectedNet(input_dim=params['v_dim'],output_dim = sum(params['z_dims']), 
                                            model_name='e_net', nb_units=params['e_units'])
            self.f_net = MCMCFullyConnectedNet(input_dim=params['z_dims'][0]+params['z_dims'][1]+1,
                                           output_dim = 2, model_name='f_net', nb_units=params['f_units'])
            self.h_net = MCMCFullyConnectedNet(input_dim=params['z_dims'][0]+params['z_dims'][2],
                                           output_dim = 2, model_name='h_net', nb_units=params['h_units'])
        else:
            self.g_net = BaseFullyConnectedNet(input_dim=sum(params['z_dims']),output_dim = params['v_dim']+1, 
                                           model_name='g_net', nb_units=params['g_units'])
            self.e_net = BaseFullyConnectedNet(input_dim=params['v_dim'],output_dim = sum(params['z_dims']), 
                                            model_name='e_net', nb_units=params['e_units'])
            self.f_net = BaseFullyConnectedNet(input_dim=params['z_dims'][0]+params['z_dims'][1]+1,
                                           output_dim = 2, model_name='f_net', nb_units=params['f_units'])
            self.h_net = BaseFullyConnectedNet(input_dim=params['z_dims'][0]+params['z_dims'][2],
                                           output_dim = 2, model_name='h_net', nb_units=params['h_units'])

        self.dz_net = Discriminator(input_dim=sum(params['z_dims']),model_name='dz_net',
                                        nb_units=params['dz_units'])

        self.g_pre_optimizer = tf.keras.optimizers.Adam(params['lr'], beta_1=0.9, beta_2=0.99)
        self.d_pre_optimizer = tf.keras.optimizers.Adam(params['lr'], beta_1=0.9, beta_2=0.99)
        self.z_sampler = Gaussian_sampler(mean=np.zeros(sum(params['z_dims'])), sd=1.0)

        self.g_optimizer = tf.keras.optimizers.Adam(params['lr_theta'], beta_1=0.9, beta_2=0.99)
        self.f_optimizer = tf.keras.optimizers.Adam(params['lr_theta'], beta_1=0.9, beta_2=0.99)
        self.h_optimizer = tf.keras.optimizers.Adam(params['lr_theta'], beta_1=0.9, beta_2=0.99)
        self.posterior_optimizer = tf.keras.optimizers.Adam(params['lr_z'], beta_1=0.9, beta_2=0.99)
        
        self.initialize_nets()
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

        self.ckpt = tf.train.Checkpoint(g_net = self.g_net,
                                    e_net = self.e_net,
                                    f_net = self.f_net,
                                    h_net = self.h_net,
                                    dz_net = self.dz_net,
                                    g_pre_optimizer = self.g_pre_optimizer,
                                    d_pre_optimizer = self.d_pre_optimizer,
                                    g_optimizer = self.g_optimizer,
                                    f_optimizer = self.f_optimizer,
                                    h_optimizer = self.h_optimizer,
                                    posterior_optimizer = self.posterior_optimizer)
        
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_path, max_to_keep=5)                 

        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print ('Latest checkpoint restored!!') 
                    
    def run_mcmc_training(self, data, num_samples=2000, num_burnin=1000, eps=1e-6):
        """Draw posterior weight samples via Hamiltonian Monte Carlo.

        Runs HMC on the weights of ``g_net``, ``h_net``, and ``f_net``
        conditioned on the optimised latent variables from :meth:`fit`.
        Must be called **after** :meth:`fit`.

        Parameters
        ----------
        data : tuple of np.ndarray
            A triplet ``(data_x, data_y, data_v)``.
        num_samples : int, default=2000
            Number of HMC posterior samples to draw.
        num_burnin : int, default=1000
            Number of burn-in steps to discard.
        eps : float, default=1e-6
            Small constant added for numerical stability in the
            likelihood computation.
        """

        data_x, data_y, data_v = data
        data_z = self.data_z.numpy() # Use the optimized latent variables from fit()
        data_z0 = data_z[:, :self.params['z_dims'][0]]
        data_z1 = data_z[:, self.params['z_dims'][0]:sum(self.params['z_dims'][:2])]
        data_z2 = data_z[:, sum(self.params['z_dims'][:2]):sum(self.params['z_dims'][:3])]

        # --- MCMC for g_net (predicting V from Z) ---
        def g_net_likelihood(v_true, v_pred_out):
            mu_v = v_pred_out[:, :self.params['v_dim']]
            # By using `[-1:]` we keep the last dimension, changing the shape from (batch,) to (batch, 1)
            sigma_square_v = tf.nn.softplus(v_pred_out[:, -1:]) + eps # <--- Fixed
            log_prob = tf.reduce_sum(tfp.distributions.Normal(mu_v, tf.sqrt(sigma_square_v)).log_prob(v_true))
            return log_prob

        self.g_net_samples = run_mcmc_for_net(
            self.g_net, data_z, data_v, g_net_likelihood,
            self.g_net.get_weights(), num_samples, num_burnin
        )

        # --- MCMC for h_net (predicting X from Z) ---
        def h_net_likelihood(x_true, x_pred_out):
            mu_x = x_pred_out[:, :1]
            if self.params['binary_treatment']:
                dist = tfp.distributions.Bernoulli(logits=mu_x)
            else:
                sigma_square_x = tf.nn.softplus(x_pred_out[:, -1]) + eps
                dist = tfp.distributions.Normal(mu_x, tf.sqrt(sigma_square_x))
            return tf.reduce_sum(dist.log_prob(x_true))
        
        h_net_input = tf.concat([data_z0, data_z2], axis=-1)
        self.h_net_samples = run_mcmc_for_net(
            self.h_net, h_net_input, data_x, h_net_likelihood,
            self.h_net.get_weights(), num_samples, num_burnin
        )
        
        # --- MCMC for f_net (predicting Y from Z, X) ---
        def f_net_likelihood(y_true, y_pred_out):
            mu_y = y_pred_out[:, :1]
            sigma_square_y = tf.nn.softplus(y_pred_out[:, -1]) + eps
            log_prob = tf.reduce_sum(tfp.distributions.Normal(mu_y, tf.sqrt(sigma_square_y)).log_prob(y_true))
            return log_prob

        f_net_input = tf.concat([data_z0, data_z1, data_x], axis=-1)
        self.f_net_samples = run_mcmc_for_net(
            self.f_net, f_net_input, data_y, f_net_likelihood,
            self.f_net.get_weights(), num_samples, num_burnin
        )
                    
    # Predict with MCMC sampling
    def predict(self, data, alpha=0.01, n_mcmc=3000, x_values=None, q_sd=1.0, sample_y=True, bs=100):
        """Predict causal effects with full posterior uncertainty.

        Marginalises over **both** latent-variable and network-weight
        uncertainty.  :meth:`run_mcmc_training` must be called first to
        populate weight samples.

        Parameters
        ----------
        data : tuple of np.ndarray
            A triplet ``(data_x, data_y, data_v)``.
        alpha : float, default=0.01
            Significance level for posterior intervals.
        n_mcmc : int, default=3000
            Number of posterior MCMC samples for latent variables.
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
            Posterior intervals.
        """
        
        assert 0 < alpha < 1, "The significance level 'alpha' must be greater than 0 and less than 1."

        if not self.params['binary_treatment']:
            # Validate x_values for binary treatment
            if x_values is None:
                raise ValueError("For continous treatment, 'x_values' must not be None. Provide a list or a single treatment value.")

        if x_values is not None:
            if np.isscalar(x_values):
                # Convert scalar to 1D array
                x_values = np.array([x_values], dtype=float) 
            else:
                # Convert list to NumPy array
                x_values = np.array(x_values, dtype=float)

        # Initialize list to store causal effect samples
        causal_effects = []
        print('MCMC Latent Variable Sampling ...')
        data_posterior_z = self.metropolis_hastings_sampler(data, 
                                                            g_net_samples=self.g_net_samples,
                                                            h_net_samples=self.h_net_samples,
                                                            f_net_samples=self.f_net_samples,
                                                            n_keep=n_mcmc, 
                                                            q_sd=q_sd)
        print('Number of x_values:', len(x_values))
        print('Shape of NN weights by MCMC:', self.g_net_samples.shape, self.h_net_samples.shape, self.f_net_samples.shape)
        print('Shape of Latent Variable Z by MCMC:', data_posterior_z.shape)
        f_net_weights = self.f_net_samples
        # Randomly select one weight sample for each Z sample to create pairs
        num_z_samples = data_posterior_z.shape[0] #MCMC sample size for Z
        num_weight_samples = f_net_weights.shape[0] #MCMC sample size for weights
        # This creates a paired set of indices for efficient lookup
        paired_weight_indices = np.random.randint(0, num_weight_samples, size=num_z_samples)
        paired_f_net_weights = tf.gather(f_net_weights, paired_weight_indices)
        
      
        # Iterate over the data_posterior_z in batches
        for i in range(0, data_posterior_z.shape[0], bs):
            batch_posterior_z = data_posterior_z[i:i + bs]
            batch_weights = paired_f_net_weights[i:i + bs]
            
            causal_effect_batch = self.infer_from_latent_posterior(batch_posterior_z, 
                                                                   f_net_weights=batch_weights,
                                                                   x_values=x_values, 
                                                                   sample_y=sample_y).numpy()
            causal_effects.append(causal_effect_batch)
        
        # Estimate the posterior interval with user-specific significance level alpha
        print('Shape of causal effect:', np.array(causal_effects).shape)

        if self.params['binary_treatment']:
            # For binary treatment: Individual Treatment Effect (ITE)
            causal_effects = np.concatenate(causal_effects, axis=0)
            ITE = np.mean(causal_effects, axis=0)
            posterior_interval_upper = np.quantile(causal_effects, 1-alpha/2, axis=0)
            posterior_interval_lower = np.quantile(causal_effects, alpha/2, axis=0)
            pos_int = np.stack([posterior_interval_lower, posterior_interval_upper], axis=1)
            return ITE, pos_int
        else:
            # For continuous treatment: Average Dose Response Function (ADRF)
            causal_effects = np.concatenate(causal_effects, axis=0)
            ADRF = np.mean(causal_effects, axis=0)
            posterior_interval_upper = np.quantile(causal_effects, 1-alpha/2, axis=0)
            posterior_interval_lower = np.quantile(causal_effects, alpha/2, axis=0)
            pos_int = np.stack([posterior_interval_lower, posterior_interval_upper], axis=1)
            return ADRF, pos_int

        
    @tf.function
    def infer_from_latent_posterior(self, data_posterior_z, f_net_weights=None, x_values=None, sample_y=True, eps=1e-6):
        """Infer causal estimate on the test data and give estimation interval and posterior latent variables. ITE is estimated for binary treatment and ADRF is estimated for continous treatment.
        data_posterior_z: (np.ndarray): Posterior latent variables with shape (n_samples, n, p), where p is the dimension of Z.
        x_values: (list of floats or np.ndarray): Number of intervals for the dose response function.
        sample_y: (bool): consider the variance function in outcome generative model.
        return (np.ndarray): 
            ITE with shape (n_samples, n) containing all the MCMC samples.
            ADRF with shape (n_samples, len(x_values)) containing all the MCMC samples for each treatment value.
        """
        # Helper function to compute effect for a single paired (z_sample, weight_sample)
        def compute_effect(elems):
            z_sample, weight_sample = elems
            z0 = z_sample[:,:self.params['z_dims'][0]]
            z1 = z_sample[:,self.params['z_dims'][0]:sum(self.params['z_dims'][:2])]

            if self.params['binary_treatment']:
                # Predict outcome under treatment (x=1)
                input_pos = tf.concat([z0, z1, tf.ones([tf.shape(z_sample)[0], 1])], axis=-1)
                out_pos = self.f_net.call_with_weights(input_pos, weight_sample)
                mu_y_pos, sigma_y_pos = out_pos[:, :1], tf.nn.softplus(out_pos[:, 1:]) + eps

                # Predict outcome under control (x=0)
                input_neg = tf.concat([z0, z1, tf.zeros([tf.shape(z_sample)[0], 1])], axis=-1)
                out_neg = self.f_net.call_with_weights(input_neg, weight_sample)
                mu_y_neg, sigma_y_neg = out_neg[:, :1], tf.nn.softplus(out_neg[:, 1:]) + eps

                if sample_y: # Account for Aleatoric uncertainty
                    y_pred_pos = tf.random.normal(shape=tf.shape(mu_y_pos), mean=mu_y_pos, stddev=tf.sqrt(sigma_y_pos))
                    y_pred_neg = tf.random.normal(shape=tf.shape(mu_y_neg), mean=mu_y_neg, stddev=tf.sqrt(sigma_y_neg))
                else: # Use only the mean (epistemic + latent uncertainty only)
                    y_pred_pos, y_pred_neg = mu_y_pos, mu_y_neg

                # Return one sample of the ITE for each individual
                ite_pred = y_pred_pos - y_pred_neg
                return np.squeeze(ite_pred)
            else:
                # ADRF implementation would go here, mapping over x_values
                def compute_dose_response(x):
                    data_x_tile = tf.cast(tf.fill([tf.shape(z_sample)[0], 1], x), tf.float32)
                    y_out = self.f_net.call_with_weights(tf.concat([z0, z1, data_x_tile], axis=-1), weight_sample)
                    mu_y, sigma_y = y_out[:, :1], tf.nn.softplus(y_out[:, 1:]) + eps
                    if sample_y:
                        y_pred = tf.random.normal(shape=tf.shape(mu_y), mean=mu_y, stddev=tf.sqrt(sigma_y))
                    else:
                        y_pred = mu_y
                    return tf.reduce_mean(y_pred)

                return tf.map_fn(compute_dose_response, x_values, fn_output_signature=tf.float32)


        causal_effects = tf.map_fn(
            compute_effect,
            (data_posterior_z, f_net_weights),
            fn_output_signature=tf.float32 if self.params['binary_treatment'] else tf.TensorSpec(shape=(len(x_values),), dtype=tf.float32)
        )
            
        return causal_effects

    @tf.function
    def get_log_posterior(self, data_x, data_y, data_v, data_z, g_weights, h_weights, f_weights, eps=1e-6):
        """
        Calculate log posterior of Z for a GIVEN set of network weights.
        This version is stateless and graph-compatible.

        g_weights, h_weights, f_weights: Flattened tensors of weights for each network.
        """
        data_z0 = data_z[:, :self.params['z_dims'][0]]
        data_z1 = data_z[:, self.params['z_dims'][0]:sum(self.params['z_dims'][:2])]
        data_z2 = data_z[:, sum(self.params['z_dims'][:2]):sum(self.params['z_dims'][:3])]

        # logp(v|z) for covariate model
        g_net_output = self.g_net.call_with_weights(data_z, g_weights)
        mu_v = g_net_output[:, :self.params['v_dim']]
        sigma_square_v = tf.nn.softplus(g_net_output[:, -1]) + eps

        # logp(x|z) for treatment model
        h_net_input = tf.concat([data_z0, data_z2], axis=-1)
        h_net_output = self.h_net.call_with_weights(h_net_input, h_weights)
        mu_x = h_net_output[:, :1]

        # logp(y|z,x) for outcome model
        f_net_input = tf.concat([data_z0, data_z1, data_x], axis=-1)
        f_net_output = self.f_net.call_with_weights(f_net_input, f_weights)
        mu_y = f_net_output[:, :1]

        # --- Calculate Likelihood Losses (Negative Log-Likelihoods) ---

        loss_pv_z = tf.reduce_sum((data_v - mu_v)**2, axis=1) / (2 * sigma_square_v) + \
                    self.params['v_dim'] * tf.math.log(sigma_square_v) / 2

        if self.params['binary_treatment']:
            loss_px_z = tf.squeeze(tf.nn.sigmoid_cross_entropy_with_logits(labels=data_x, logits=mu_x))
        else:
            sigma_square_x = tf.nn.softplus(h_net_output[:, -1]) + eps
            loss_px_z = tf.reduce_sum((data_x - mu_x)**2, axis=1) / (2 * sigma_square_x) + \
                        tf.math.log(sigma_square_x) / 2

        sigma_square_y = tf.nn.softplus(f_net_output[:, -1]) + eps
        loss_py_zx = tf.reduce_sum((data_y - mu_y)**2, axis=1) / (2 * sigma_square_y) + \
                     tf.math.log(sigma_square_y) / 2

        # --- Calculate Prior Loss ---
        loss_prior_z = tf.reduce_sum(data_z**2, axis=1) / 2

        # --- Total Negative Log-Posterior ---
        loss_posterior_z = loss_pv_z + loss_px_z + loss_py_zx + loss_prior_z

        log_posterior = -loss_posterior_z
        return log_posterior


    def metropolis_hastings_sampler(self, data, g_net_samples, h_net_samples, f_net_samples, initial_q_sd = 1.0, q_sd = None, burn_in = 5000, n_keep = 3000, target_acceptance_rate=0.25, tolerance=0.05, adjustment_interval=50, adaptive_sd=None, window_size=100):
        """
        Samples from the posterior distribution P(Z|X,Y,V) using the Metropolis-Hastings algorithm with adaptive proposal adjustment.

        Args:
            data (tuple): Tuple containing data_x, data_y, data_v.
            q_sd (float or None): Fixed standard deviation for the proposal distribution. If None, `q_sd` will adapt.
            initial_q_sd (float): Initial standard deviation of the proposal distribution.
            burn_in (int): Number of samples for burn-in, set to 1000 as an initial estimate.
            n_keep (int): Number of samples retained after burn-in.
            target_acceptance_rate (float): Target acceptance rate for the Metropolis-Hastings algorithm.
            tolerance (float): Acceptable deviation from the target acceptance rate.
            adjustment_interval (int): Number of iterations between each adjustment of `q_sd`.
            window_size (int): The size of the sliding window for acceptance rate calculation.

        Returns:
            np.ndarray: Posterior samples with shape (n_keep, n, q), where q is the dimension of Z.
        """
        
        data_x, data_y, data_v = data

        # Initialize the state of n chains
        current_state = np.random.normal(0, 1, size = (len(data_x), sum(self.params['z_dims']))).astype('float32')

        # Initialize the list to store the samples
        samples = []
        counter = 0
        
        # Sliding window for acceptance tracking
        recent_acceptances = []

        num_weight_samples = f_net_samples.shape[0]
        
        # Determine if q_sd should be adaptive
        if adaptive_sd is None:
            adaptive_sd = (q_sd is None or q_sd <= 0)

        # Set the initial q_sd
        if adaptive_sd:
            q_sd = initial_q_sd
            
        # Run the Metropolis-Hastings algorithm
        while len(samples) < n_keep:
            # Propose a new state by sampling from a multivariate normal distribution
            proposed_state = current_state + np.random.normal(0, q_sd, size = (len(data_x), sum(self.params['z_dims']))).astype('float32')

            rand_idx = np.random.randint(0, num_weight_samples)
            g_w = g_net_samples[rand_idx]
            h_w = h_net_samples[rand_idx]
            f_w = f_net_samples[rand_idx]
        
            # Compute the acceptance ratio
            proposed_log_posterior = self.get_log_posterior(data_x, data_y, data_v, proposed_state, g_w, h_w, f_w)
            current_log_posterior  = self.get_log_posterior(data_x, data_y, data_v, current_state, g_w, h_w, f_w)
            #acceptance_ratio = np.exp(proposed_log_posterior-current_log_posterior)
            acceptance_ratio = np.exp(np.minimum(proposed_log_posterior - current_log_posterior, 0))
            # Accept or reject the proposed state
            indices = np.random.rand(len(data_x)) < acceptance_ratio
            current_state[indices] = proposed_state[indices]
            
            # Update the sliding window
            recent_acceptances.append(indices)
            if len(recent_acceptances) > window_size:
                # Keep only the most recent `window_size` elements
                recent_acceptances = recent_acceptances[-window_size:]
            
            # Adjust q_sd periodically during the burn-in phase
            if adaptive_sd and counter < burn_in and counter % adjustment_interval == 0 and counter > 0:
                # Calculate the current acceptance rate
                current_acceptance_rate = np.sum(recent_acceptances) / (len(recent_acceptances)*len(data_x))
                
                print(f"Current MCMC Acceptance Rate: {current_acceptance_rate:.4f}")
                
                # Adjust q_sd based on the acceptance rate
                if current_acceptance_rate < target_acceptance_rate - tolerance:
                    q_sd *= 0.9  # Decrease q_sd to increase acceptance rate
                elif current_acceptance_rate > target_acceptance_rate + tolerance:
                    q_sd *= 1.1  # Increase q_sd to decrease acceptance rate
                    
                print(f"MCMC Proposal Standard Deviation (q_sd): {q_sd:.4f}")

            # Append the current state to the list of samples
            if counter >= burn_in:
                samples.append(current_state.copy())
            
            counter += 1
            
        # Calculate the acceptance rate
        acceptance_rate = np.sum(recent_acceptances) / (len(recent_acceptances)*len(data_x))
        print(f"Final MCMC Acceptance Rate: {acceptance_rate:.4f}")
        #print(f"Final Proposal Standard Deviation (q_sd): {q_sd:.4f}")
        return np.array(samples)
