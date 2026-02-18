import datetime
import os

import dateutil.tz
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from bayesgm.datasets import Base_sampler, Gaussian_sampler

from ..networks import (
    Discriminator,
    MNISTDiscriminator,
    MNISTEncoderConv,
    MNISTGenerator,
)
from .base import BGM

class MNISTBGM(BGM):
    """BGM model for MNIST imaging data.

    Inherits from :class:`BGM` and overrides methods to use convolutional
    neural networks and a Bernoulli likelihood for binary image data of
    shape ``(28, 28, 1)``.

    Parameters
    ----------
    params : dict
        Configuration dictionary. Same keys as :class:`BGM`, with
        ``x_dim`` corresponding to the flattened image dimensionality
        (784 for MNIST).
    timestamp : str or None, optional
        Timestamp string for the run.  If ``None``, the current local time
        is used.
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

        # MNIST-specific networks
        self.g_net = MNISTGenerator(z_dim=params['z_dim'], filters=32,
                                    use_bnn=params['use_bnn'], name='g_net')
        self.e_net = MNISTEncoderConv(z_dim=params['z_dim'], filters=32, name='e_net')
        self.dz_net = Discriminator(input_dim=params['z_dim'], model_name='dz_net',
                                    nb_units=params['dz_units'])
        self.dx_net = MNISTDiscriminator(filters=64, name='dx_net')

        self.g_pre_optimizer = tf.keras.optimizers.Adam(params['lr'], beta_1=0.5, beta_2=0.9)
        self.d_pre_optimizer = tf.keras.optimizers.Adam(params['lr'], beta_1=0.5, beta_2=0.9)
        self.z_sampler = Gaussian_sampler(mean=np.zeros(params['z_dim']), sd=1.0)

        self.g_optimizer = tf.keras.optimizers.Adam(params['lr_theta'], beta_1=0.9, beta_2=0.99)
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
                                    dz_net = self.dz_net,
                                    dx_net = self.dx_net,
                                    g_pre_optimizer = self.g_pre_optimizer,
                                    d_pre_optimizer = self.d_pre_optimizer,
                                    g_optimizer = self.g_optimizer,
                                    posterior_optimizer = self.posterior_optimizer)
        
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.checkpoint_path, max_to_keep=100)                 

        if self.ckpt_manager.latest_checkpoint:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            print ('Latest checkpoint restored!!')

    # Update generative model for X
    @tf.function
    def update_g_net(self, data_z, data_x):
        """
        Updates the generative model g_net using Bernoulli log-likelihood for MNIST.
        Args:
            data_z: Tensor of shape (batch, z_dim), latent variable.
            data_x: Tensor of shape (batch, 28, 28, 1), observed MNIST data.
        Returns:
            loss_x: Scalar loss value for training g_net.
            loss_mse: Mean squared error between observed and predicted x.
        """
        with tf.GradientTape() as gen_tape:
            mu_x, sigma_square_x = self.g_net(data_z)
            x_logits = self.g_net.reparameterize(mu_x, sigma_square_x)

            # Convert logits to probabilities for MSE calculation
            x_probs = tf.nn.sigmoid(x_logits)
            loss_mse = tf.reduce_mean((data_x - x_probs)**2)
            
            # Bernoulli log-likelihood: -log p(x|z)
            # For Bernoulli: p(x) = x * p + (1-x) * (1-p) where p = sigmoid(logits)
            # log p(x) = x * log(p) + (1-x) * log(1-p)
            # log p(x) = x * logits - log(1 + exp(logits))  (using log-sum-exp trick)
            x_logits = tf.clip_by_value(x_logits, -10, 10)  # Prevent overflow
            log_px_z = tf.reduce_sum(
                data_x * x_logits - tf.nn.softplus(x_logits), 
                axis=[1, 2, 3]  # Sum over spatial dimensions
            )
            loss_x = -tf.reduce_mean(log_px_z)  # Negative log-likelihood
            
            if self.params['use_bnn']:
                loss_kl = sum(self.g_net.losses)
                loss_x += loss_kl * self.params['kl_weight']

        # Calculate the gradients for generators
        g_gradients = gen_tape.gradient(loss_x, self.g_net.trainable_variables)
        
        # Apply the gradients to the optimizer
        self.g_optimizer.apply_gradients(zip(g_gradients, self.g_net.trainable_variables))
        return loss_x, loss_mse
        
    # Update posterior of latent variables Z
    @tf.function
    def update_latent_variable_sgd(self, data_z, data_x):
        with tf.GradientTape() as tape:
            
            # logp(x|z) for Bernoulli model
            mu_x, sigma_square_x = self.g_net(data_z)
            x_logits = self.g_net.reparameterize(mu_x, sigma_square_x)
            
            # Bernoulli log-likelihood: -log p(x|z)
            x_logits = tf.clip_by_value(x_logits, -10, 10)  # Prevent overflow
            log_px_z = tf.reduce_sum(
                data_x * x_logits - tf.nn.softplus(x_logits), 
                axis=[1, 2, 3]  # Sum over spatial dimensions
            )
            loss_px_z = -tf.reduce_mean(log_px_z)  # Negative log-likelihood

            loss_prior_z =  tf.reduce_sum(data_z**2, axis=1)/2
            loss_prior_z = tf.reduce_mean(loss_prior_z)

            loss_postrior_z = loss_px_z + loss_prior_z

        # Calculate the gradients
        posterior_gradients = tape.gradient(loss_postrior_z, [data_z])
        
        # Apply the gradients to the optimizer
        self.posterior_optimizer.apply_gradients(zip(posterior_gradients, [data_z]))
        return loss_postrior_z
    
#################################### EGM initialization ###########################################
    @tf.function
    def train_disc_step(self, data_z, data_x):
        """train discriminators step for MNIST image data.
        Args:
            data_z: Latent tensor with shape [batch_size, z_dim].
            data_x: Image tensor with shape [batch_size, 28, 28, 1].
        Returns:
            returns various discriminator loss functions.
        """  
        epsilon_z = tf.random.uniform([],minval=0., maxval=1.)
        epsilon_x = tf.random.uniform([],minval=0., maxval=1.)
        with tf.GradientTape(persistent=True) as disc_tape:
            with tf.GradientTape() as gpz_tape:
                data_z_ = self.e_net(data_x)
                data_z_hat = data_z*epsilon_z + data_z_*(1-epsilon_z)
                data_dz_hat = self.dz_net(data_z_hat)
            with tf.GradientTape() as gpx_tape:
                mu_x_, sigma_square_x_ = self.g_net(data_z)
                x_logits_ = self.g_net.reparameterize(mu_x_, sigma_square_x_)
                data_x_ = tf.nn.sigmoid(x_logits_)
                data_x_hat = data_x*epsilon_x + data_x_*(1-epsilon_x)
                data_dx_hat = self.dx_net(data_x_hat)
            
            data_dx_ = self.dx_net(data_x_)
            data_dz_ = self.dz_net(data_z_)
            
            data_dx = self.dx_net(data_x)
            data_dz = self.dz_net(data_z)
            
            dz_loss = (tf.reduce_mean((0.9*tf.ones_like(data_dz) - data_dz)**2) \
                +tf.reduce_mean((0.1*tf.ones_like(data_dz_) - data_dz_)**2))/2.0
            dx_loss = (tf.reduce_mean((0.9*tf.ones_like(data_dx) - data_dx)**2) \
                +tf.reduce_mean((0.1*tf.ones_like(data_dx_) - data_dx_)**2))/2.0
            
            #gradient penalty for z
            grad_z = gpz_tape.gradient(data_dz_hat, data_z_hat)
            grad_norm_z = tf.sqrt(tf.reduce_sum(tf.square(grad_z), axis=1))#(bs,) 
            gpz_loss = tf.reduce_mean(tf.square(grad_norm_z - 1.0))
            
            #gradient penalty for x (spatial dimensions for images)
            grad_x = gpx_tape.gradient(data_dx_hat, data_x_hat)
            grad_norm_x = tf.sqrt(tf.reduce_sum(tf.square(grad_x), axis=[1, 2, 3]))#(bs,) 
            gpx_loss = tf.reduce_mean(tf.square(grad_norm_x - 1.0))
                
            d_loss = dx_loss + dz_loss + \
                    self.params['gamma']*(gpz_loss + gpx_loss)

        # Calculate the gradients for discriminators
        d_gradients = disc_tape.gradient(d_loss, self.dz_net.trainable_variables+self.dx_net.trainable_variables)
        
        # Apply the gradients to the optimizer
        self.d_pre_optimizer.apply_gradients(zip(d_gradients, self.dz_net.trainable_variables+self.dx_net.trainable_variables))
        
        return dz_loss, dx_loss, d_loss
    
    @tf.function
    def train_gen_step(self, data_z, data_x):
        """train generators step for MNIST image data.
        Args:
            data_z: Latent tensor with shape [batch_size, z_dim].
            data_x: Image tensor with shape [batch_size, 28, 28, 1].
        Returns:
            returns various generator loss functions.
        """  
        with tf.GradientTape(persistent=True) as gen_tape:
            mu_x_, sigma_square_x_ = self.g_net(data_z)
            x_logits_ = self.g_net.reparameterize(mu_x_, sigma_square_x_)
            data_x_ = tf.nn.sigmoid(x_logits_)
            reg_loss = tf.reduce_mean(tf.square(sigma_square_x_))
            data_z_ = self.e_net(data_x)

            data_z__= self.e_net(data_x_)
            mu_x__, sigma_square_x__ = self.g_net(data_z_)
            x_logits__ = self.g_net.reparameterize(mu_x__, sigma_square_x__)
            data_x__ = tf.nn.sigmoid(x_logits__)
            
            data_dx_ = self.dx_net(data_x_)
            data_dz_ = self.dz_net(data_z_)
            
            l2_loss_x = tf.reduce_mean((data_x - data_x__)**2)
            l2_loss_z = tf.reduce_mean((data_z - data_z__)**2)
            
            g_loss_adv = tf.reduce_mean((0.9*tf.ones_like(data_dx_)  - data_dx_)**2)
            e_loss_adv = tf.reduce_mean((0.9*tf.ones_like(data_dz_)  - data_dz_)**2)

            g_e_loss = g_loss_adv + e_loss_adv + 10 * (l2_loss_x + l2_loss_z) + self.params['alpha'] * reg_loss
                
        # Calculate the gradients for generators
        g_e_gradients = gen_tape.gradient(g_e_loss, self.g_net.trainable_variables+self.e_net.trainable_variables)
        
        # Apply the gradients to the optimizer
        self.g_pre_optimizer.apply_gradients(zip(g_e_gradients, self.g_net.trainable_variables+self.e_net.trainable_variables))

        return g_loss_adv, e_loss_adv, l2_loss_z, l2_loss_x, reg_loss, g_e_loss
    

    def egm_init(self, data, egm_n_iter=10000, batch_size=32, egm_batches_per_eval=500, verbose=1):
        
        self.data_sampler = Base_sampler(x=data,y=data,v=data, batch_size=batch_size, normalize=False)
        print('EGM Initialization Starts ...')
        for batch_iter in range(egm_n_iter+1):
            # Update model parameters of Discriminator
            for _ in range(self.params['g_d_freq']):
                batch_x,_,_ = self.data_sampler.next_batch()
                batch_z = self.z_sampler.get_batch(batch_size)
                dz_loss, dx_loss, d_loss = self.train_disc_step(batch_z, batch_x)

            # Update model parameters of G,E with SGD
            batch_x,_,_ = self.data_sampler.next_batch()
            batch_z = self.z_sampler.get_batch(batch_size)
            g_loss_adv, e_loss_adv, l2_loss_z, l2_loss_x, sigma_square_loss, g_e_loss = self.train_gen_step(batch_z, batch_x)
            if batch_iter % egm_batches_per_eval == 0:
                
                loss_contents = (
                    'EGM Initialization Iter [%d] : g_loss_adv[%.4f], e_loss_adv [%.4f], l2_loss_z [%.4f], l2_loss_x [%.4f], '
                    'sd^2_loss[%.4f], g_e_loss [%.4f], dz_loss [%.4f], dx_loss[%.4f], d_loss [%.4f]'
                    % (batch_iter, g_loss_adv, e_loss_adv, l2_loss_z, l2_loss_x, sigma_square_loss, g_e_loss, dz_loss, dx_loss, d_loss)
                )
                if verbose:
                    print(loss_contents)
                data_z_ = self.e_net(data)
                mu_x__, sigma_square_x__ = self.g_net(data_z_)
                x_logits__ = self.g_net.reparameterize(mu_x__, sigma_square_x__)
                data_x__ = tf.nn.sigmoid(x_logits__)
                MSE = tf.reduce_mean((data - data_x__)**2)
                data_gen = self.generate(nb_samples=5000)
                np.savez('%s/init_data_gen_at_%d.npz'%(self.save_dir, batch_iter),
                        data_gen=data_gen, z=data_z_, x_rec=data_x__)
                print('MSE_x', MSE.numpy())
                mse_x = self.evaluate(data = data)
                print('iter [%d/%d]: MSE_x: %.4f\n' % (batch_iter, egm_n_iter, mse_x))
                if self.params['save_model']:
                    base_path = self.checkpoint_path + f"/weights_at_egm_init_{batch_iter}"
                    self.e_net.save_weights(f"{base_path}_encoder.weights.h5")
                    self.g_net.save_weights(f"{base_path}_generator.weights.h5")
                    print('Saving checkpoint for egm_init at {}'.format(base_path))


        print('EGM Initialization Ends.')
#################################### EGM initialization #############################################

    def fit(self, data,
            batch_size=32, epochs=100, epochs_per_eval=5,
            use_egm_init=True, egm_n_iter=10000, egm_batches_per_eval=500, verbose=1):
        """Train the MNIST BGM model on image data.

        Parameters
        ----------
        data : np.ndarray
            MNIST image array with shape ``(n, 28, 28, 1)``, values in [0, 1].
        batch_size : int, default=32
            Mini-batch size.
        epochs : int, default=100
            Number of training epochs for the iterative phase.
        epochs_per_eval : int, default=5
            Evaluate and (optionally) save every this many epochs.
        use_egm_init : bool, default=True
            Whether to run EGM initialization before iterative training.
        egm_n_iter : int, default=10000
            Number of EGM initialization iterations.
        egm_batches_per_eval : int, default=500
            Evaluate EGM every this many iterations.
        verbose : int, default=1
            Verbosity level.  Set to 0 to suppress progress messages.
        """

        if self.params['save_res']:
            f_params = open('{}/params.txt'.format(self.save_dir),'w')
            f_params.write(str(self.params))
            f_params.close()
        
        if use_egm_init:
            self.egm_init(data, egm_n_iter=egm_n_iter, egm_batches_per_eval=egm_batches_per_eval, batch_size=batch_size, verbose=verbose)
            print('Initialize latent variables Z with e(V)...')
            data_z_init = self.e_net(data)
        else:
            print('Random initialization of latent variables Z...')
            data_z_init = np.random.normal(0, 1, size = (len(data), self.params['z_dim'])).astype('float32')

        self.data_z = tf.Variable(data_z_init, name="Latent Variable",trainable=True)

        self.history_loss = []
        print('Iterative Updating Starts ...')
        for epoch in range(epochs+1):
            sample_idx = np.random.choice(len(data), len(data), replace=False)
            
            # Create a progress bar for batches
            with tqdm(total=len(data) // batch_size, desc=f"Epoch {epoch}/{epochs}", unit="batch") as batch_bar:
                for i in range(0,len(data) - batch_size + 1,batch_size): ## Skip the incomplete last batch
                    batch_idx = sample_idx[i:i+batch_size]
                    # Update model parameters of G, H, F with SGD
                    batch_z = tf.Variable(tf.gather(self.data_z, batch_idx, axis = 0), name='batch_z', trainable=True)
                    batch_x = data[batch_idx,:]
                    loss_x, loss_mse_x = self.update_g_net(batch_z, batch_x)

                    # Update Z by maximizing a posterior or posterior mean
                    loss_postrior_z = self.update_latent_variable_sgd(batch_z, batch_x)

                    # Update data_z with updated batch_z
                    self.data_z.scatter_nd_update(
                        indices=tf.expand_dims(batch_idx, axis=1),
                        updates=batch_z                             
                    )
                    
                    # Update the progress bar with the current loss information
                    loss_contents = (
                        'loss_x: [%.4f], loss_mse_x: [%.4f], loss_postrior_z: [%.4f]'
                        % (loss_x, loss_mse_x, loss_postrior_z)
                    )
                    batch_bar.set_postfix_str(loss_contents)
                    batch_bar.update(1)
            
            # Evaluate the full training data and print metrics for the epoch
            if epoch % epochs_per_eval == 0:
                mse_x = self.evaluate(data = data, data_z = self.data_z)
                self.history_loss.append(mse_x)

                if verbose:
                    print('Epoch [%d/%d]: MSE_x: %.4f\n' % (epoch, epochs, mse_x))

                if self.params['save_model']:
                    base_path = self.checkpoint_path + f"/weights_at_{epoch}"
                    self.g_net.save_weights(f"{base_path}_generator.weights.h5")
                    print('Saving checkpoint for epoch {} at {}'.format(epoch, base_path))
                        
                data_gen = self.generate(nb_samples=5000)
                if self.params['save_res']:
                    np.savez('%s/data_gen_at_%d.npz'%(self.save_dir, epoch),
                            gen=data_gen,
                            z=self.data_z.numpy()
                            )

    @tf.function
    def evaluate(self, data, data_z=None):
        """Compute the mean squared error between observed and reconstructed MNIST images.

        Parameters
        ----------
        data : np.ndarray or tf.Tensor
            Observed images with shape ``(n, 28, 28, 1)``.
        data_z : tf.Tensor or None, optional
            Latent variables with shape ``(n, z_dim)``.
            If ``None``, the encoder is used to infer them.

        Returns
        -------
        mse_x : tf.Tensor
            Scalar mean squared error.
        """
        if data_z is None:
            data_z = self.e_net(data, training=False)

        mu_x, sigma_square_x = self.g_net(data_z, training=False)
        x_logits = self.g_net.reparameterize(mu_x, sigma_square_x)
        data_x_pred = tf.nn.sigmoid(x_logits)

        mse_x = tf.reduce_mean((data-data_x_pred)**2)
        return mse_x

    @tf.function
    def generate(self, nb_samples=1000):
        """Generate synthetic MNIST images from the trained model.

        Samples latent codes from the standard normal prior and decodes them
        through the convolutional generator.

        Parameters
        ----------
        nb_samples : int, default=1000
            Number of images to generate.

        Returns
        -------
        data_x_pred : tf.Tensor
            Generated images with shape ``(nb_samples, 28, 28, 1)``,
            pixel values in [0, 1].
        """

        data_z = tf.random.normal(shape=(nb_samples, self.params['z_dim']), mean=0.0, stddev=1.0)

        mu_x, sigma_square_x = self.g_net(data_z, training=False)
        x_logits = self.g_net.reparameterize(mu_x, sigma_square_x)
        data_x_pred = tf.nn.sigmoid(x_logits)

        return data_x_pred

    @tf.function
    def predict_on_posteriors(self, data_posterior_z):
        n_mcmc = tf.shape(data_posterior_z)[0]
        n_samples = tf.shape(data_posterior_z)[1]

        # Flatten data
        data_posterior_z_flat = tf.reshape(data_posterior_z, [-1, self.params['z_dim']])  # (n_mcmc * n_samples, z_dim)
        mu_x_flat, sigma_square_x_flat = self.g_net(data_posterior_z_flat)  # (n_mcmc*n_samples, 28, 28, 1)
        x_logits_flat = self.g_net.reparameterize(mu_x_flat, sigma_square_x_flat)

        data_x_pred_flat = tf.nn.sigmoid(x_logits_flat)
        data_x_pred = tf.reshape(data_x_pred_flat, [n_mcmc, n_samples, 28, 28, 1])

        return data_x_pred

    def predict(self, data, alpha=0.05, return_samples=False, bs=100, n_mcmc=5000, burn_in=5000, step_size=0.01, num_leapfrog_steps=10, seed=42):
        """
        Predict the posterior distribution of P(x2|x1) for MNIST images.

        Parameters
        ----------
        data : np.ndarray or tf.Tensor
            Observed data with shape ``(n, 28, 28, 1)``.
            Missing pixels should be encoded as ``np.nan``.
        alpha : float, default=0.05
            Significance level for prediction intervals.
        return_samples : bool, default=False
            If ``False``, return imputed images with shape ``(n, 28, 28, 1)``.
            If ``True``, return posterior samples with shape
            ``(n_mcmc, n, 28, 28, 1)``.
        bs : int, default=100
            Batch size for posterior prediction.
        n_mcmc : int, default=5000
            Number of retained MCMC samples.
        burn_in : int, default=5000
            Number of burn-in iterations.
        step_size : float, default=0.01
            HMC step size.
        num_leapfrog_steps : int, default=10
            Number of leapfrog steps in HMC.
        seed : int, default=42
            Random seed.

        Returns
        -------
        data_x_pred : np.ndarray
            Imputed images if ``return_samples=False`` with shape
            ``(n, 28, 28, 1)``. Posterior predictive samples if
            ``return_samples=True`` with shape ``(n_mcmc, n, 28, 28, 1)``.
        pred_interval : np.ndarray or list[np.ndarray]
            Prediction intervals on missing pixels.
            For a shared missing pattern, shape is
            ``(n, n_missing_pixels, 2)``.
            Otherwise, this is a per-sample list where element ``i`` has shape
            ``(n_missing_pixels_i, 2)``.
        """
        assert 0 < alpha < 1, "The significance level 'alpha' must be greater than 0 and less than 1."

        if not isinstance(data, tf.Tensor):
            data_tf = tf.convert_to_tensor(data, dtype=tf.float32)
        else:
            data_tf = tf.cast(data, tf.float32)

        # Shape: (n, 28, 28, 1)
        n_data_samples = data_tf.shape[0]

        # Boolean mask of missingness (True where NaN)
        is_nan_tf = tf.math.is_nan(data_tf)
        # Observed mask (True where not NaN)
        is_obs_tf = tf.logical_not(is_nan_tf)

        # We'll still feed some numeric value at missing locations; they are ignored via indices.
        data_clean_tf = tf.where(is_nan_tf,
                                 tf.zeros_like(data_tf),
                                 data_tf)

        # Flatten observed mask to build per-sample index lists
        is_obs_flat_tf = tf.reshape(is_obs_tf, [n_data_samples, -1])
        is_obs_flat_np = is_obs_flat_tf.numpy()

        # Build ind_x1 as list-of-lists of observed pixel indices
        ind_x1_list = [
            np.where(row)[0].tolist()
            for row in is_obs_flat_np
        ]
        
        data_posterior_z = self.tfp_mcmc_sampler(
            data=data_clean_tf,
            ind_x1=ind_x1_list,
            n_mcmc=n_mcmc,
            burn_in=burn_in,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps,
            seed=seed
        )
        # data_posterior_z: (n_mcmc, n_data_samples, z_dim)
        data_x_pred_all = []
        
        # Loop over data dimension in batches
        for i in range(0, n_data_samples, bs):
            batch_posterior_z = data_posterior_z[:, i:i + bs, :]  # (n_mcmc, bs_i, z_dim)
            data_x_batch_pred = self.predict_on_posteriors(batch_posterior_z)
            # Expected shape: (n_mcmc, bs_i, 28, 28, 1)
            data_x_batch_pred = data_x_batch_pred.numpy()
            data_x_pred_all.append(data_x_batch_pred)

        # Concatenate along data dimension
        data_x_pred_all = np.concatenate(data_x_pred_all, axis=1)
        # Shape: (n_mcmc, n_data_samples, 28, 28, 1)
        
        data_np = data_tf.numpy()
        miss_mask_full = np.isnan(data_np).astype(np.float32)
        obs_mask_full = 1.0 - miss_mask_full
        data_obs_np = np.nan_to_num(data_np, nan=0.0)

        # Compute prediction intervals on missing pixels only
        n_mcmc_samples = data_x_pred_all.shape[0]
        flat_pred = data_x_pred_all.reshape(n_mcmc_samples,
                                            n_data_samples,
                                            -1)           # (n_mcmc, n, 784)

        miss_mask_flat = miss_mask_full.reshape(n_data_samples, -1).astype(bool)

        # Check if all samples share the same missing pattern
        same_pattern = np.all(miss_mask_flat == miss_mask_flat[0])

        if same_pattern:
            # Common missing pattern across all samples
            miss_idx = np.where(miss_mask_flat[0])[0]  # (N_missed_pixels,)
            if miss_idx.size == 0:
                # No missing pixels at all
                pred_interval = np.zeros((n_data_samples, 0, 2), dtype=np.float32)
            else:
                # Gather only missing pixel samples
                pix_samples = flat_pred[:, :, miss_idx]   # (n_mcmc, n, N_missed_pixels)
                lower = np.quantile(pix_samples, alpha / 2.0, axis=0)        # (n, N_missed_pixels)
                upper = np.quantile(pix_samples, 1.0 - alpha / 2.0, axis=0)  # (n, N_missed_pixels)
                pred_interval = np.stack([lower, upper], axis=-1)            # (n, N_missed_pixels, 2)
        else:
            # Different missing patterns; return a list of per-sample intervals
            pred_interval = []
            for i in range(n_data_samples):
                miss_idx_i = np.where(miss_mask_flat[i])[0]  # (N_missed_pixels_i,)
                if miss_idx_i.size == 0:
                    # This sample has no missing pixels
                    pred_interval.append(np.zeros((0, 2), dtype=np.float32))
                    continue
                pix_samples_i = flat_pred[:, i, miss_idx_i]  # (n_mcmc, N_missed_pixels_i)
                lower_i = np.quantile(pix_samples_i, alpha / 2.0, axis=0)        # (N_missed_pixels_i,)
                upper_i = np.quantile(pix_samples_i, 1.0 - alpha / 2.0, axis=0)  # (N_missed_pixels_i,)
                intervals_i = np.stack([lower_i, upper_i], axis=-1)              # (N_missed_pixels_i, 2)
                pred_interval.append(intervals_i)

        if return_samples:
            return data_x_pred_all, pred_interval
        else:
            # Return single imputed dataset: posterior mean across MCMC samples
            # Shape: (n, 28, 28, 1) — observed pixels intact, missing filled with posterior means
            data_imputed = np.mean(data_x_pred_all, axis=0)
            # Ensure observed pixels are exactly the original (avoid floating-point drift)
            data_imputed = miss_mask_full * data_imputed + obs_mask_full * data_obs_np
            return data_imputed, pred_interval

    @tf.function
    def get_log_posterior(self, data_z, data_x, ind_x1=None, obs_mask=None):
        """
        Calculate log posterior using Bernoulli likelihood for MNIST images.
        data_z: (tf.Tensor): Input data with shape (n, z_dim).
        data_x: (tf.Tensor): (n, 28, 28, 1) or (n, 784) full images;
                     missing pixels are ignored via indices/mask.
        ind_x1:  None, or int32 Tensor of shape (n, K_max) with pixel indices
                 for each sample (padded where obs_mask == 0).
        obs_mask: None, or float32 Tensor of shape (n, K_max),
                  1 for real observed indices, 0 for padding.
        return (tf.Tensor): Log posterior with shape (n, ).
        """

        mu_x, sigma_square_x = self.g_net(data_z)
        x_logits = self.g_net.reparameterize(mu_x, sigma_square_x)
        # Clip logits to prevent overflow
        x_logits = tf.clip_by_value(x_logits, -10, 10)

        # Flatten both tensors for indexing
        batch_size = tf.shape(data_x)[0]
        data_x_flat = tf.reshape(data_x, [batch_size, -1])  # (n, 784)
        x_logits_flat = tf.reshape(x_logits, [batch_size, -1])  # (n, 784)

        # Bernoulli likelihood term log p(x_obs | z)
        if ind_x1 is None:
            ll_term = data_x_flat * x_logits_flat - tf.nn.softplus(x_logits_flat)
            log_px_z = tf.reduce_sum(ll_term, axis=1)
        else:
            # Gather observed pixels per-sample
            # ind_x1: (n, K_max), obs_mask: (n, K_max)
            data_x_cond   = tf.gather(data_x_flat, ind_x1, batch_dims=1)   # (n, K_max)
            x_logits_cond = tf.gather(x_logits_flat, ind_x1, batch_dims=1) # (n, K_max)

            ll_term = data_x_cond * x_logits_cond - tf.nn.softplus(x_logits_cond)  # (n, K_max)

            if obs_mask is not None:
                ll_term = ll_term * obs_mask  # zero out padded positions

            log_px_z = tf.reduce_sum(ll_term, axis=1)  # (n,)
        
        log_prior_z = -0.5 * tf.reduce_sum(data_z**2, axis=1)
        return log_prior_z + log_px_z
