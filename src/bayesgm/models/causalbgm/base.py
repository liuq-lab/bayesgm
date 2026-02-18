import tensorflow as tf
import tensorflow_probability as tfp
from ..networks import BaseFullyConnectedNet, Discriminator, BayesianFullyConnectedNet
import numpy as np
from bayesgm.datasets import Gaussian_sampler
from bayesgm.utils.data_io import save_data
import dateutil.tz
import datetime
import os
from tqdm import tqdm

class CausalBGM(object):
    """Causal Bayesian Generative Model (CausalBGM) for causal inference.

    CausalBGM learns a latent-variable generative model for causal inference
    with treatment :math:`X`, outcome :math:`Y`, and high-dimensional covariates
    :math:`V`.  The latent variable :math:`Z` is partitioned into
    :math:`(Z_0, Z_1, Z_2, Z_3)` to disentangle confounding, outcome-specific,
    treatment-specific, and residual variation.

    Parameters
    ----------
    params : dict
        Configuration dictionary.  Required keys:

        - ``'v_dim'`` (int): Dimension of covariates :math:`V`.
        - ``'z_dims'`` (list[int]): Dimensions ``[z0, z1, z2, z3]`` of the
          four latent sub-vectors.
        - ``'binary_treatment'`` (bool): ``True`` for binary treatment,
          ``False`` for continuous.
        - ``'dataset'`` (str): Dataset name (used for checkpoint paths).
        - ``'output_dir'`` (str): Root directory for outputs.

        Optional keys (with defaults):

        - ``'use_bnn'`` (bool): Whether to use Bayesian neural networks. Default ``True``.
        - ``'g_units'`` (list[int]): Hidden-layer sizes for the generator network. Default ``[64, 64, 64, 64, 64]``.
        - ``'e_units'`` (list[int]): Hidden-layer sizes for the encoder network. Default ``[64, 64, 64, 64, 64]``.
        - ``'f_units'`` (list[int]): Hidden-layer sizes for the outcome network. Default ``[64, 32, 8]``.
        - ``'h_units'`` (list[int]): Hidden-layer sizes for the treatment network. Default ``[64, 32, 8]``.
        - ``'dz_units'`` (list[int]): Hidden-layer sizes for the latent discriminator. Default ``[64, 32, 8]``.
        - ``'lr'`` (float): Learning rate for EGM pre-training. Default ``0.0002``.
        - ``'lr_theta'`` (float): Learning rate for network parameters. Default ``0.0001``.
        - ``'lr_z'`` (float): Learning rate for latent-variable updates. Default ``0.0001``.
        - ``'g_d_freq'`` (int): Discriminator-to-generator update ratio. Default ``5``.
        - ``'save_model'`` (bool): Whether to save model checkpoints. Default ``False``.
        - ``'save_res'`` (bool): Whether to save results. Default ``True``.
        - ``'kl_weight'`` (float): KL-divergence weight when ``use_bnn`` is True. Default ``0.0001``.
    timestamp : str or None, optional
        Timestamp string for the run.  If ``None``, the current local time
        is used.
    random_seed : int or None, optional
        If provided, sets the global random seed for reproducibility.
    """

    def __init__(self, params, timestamp=None, random_seed=None):
        super(CausalBGM, self).__init__()
        self.params = params
        self.timestamp = timestamp
        if random_seed is not None:
            tf.keras.utils.set_random_seed(random_seed)
            os.environ['TF_DETERMINISTIC_OPS'] = '1'
            tf.config.experimental.enable_op_determinism()
        if self.params['use_bnn']:
            self.g_net = BayesianFullyConnectedNet(input_dim=sum(params['z_dims']),output_dim = params['v_dim']+1, 
                                           model_name='g_net', nb_units=params['g_units'])
            self.e_net = BayesianFullyConnectedNet(input_dim=params['v_dim'],output_dim = sum(params['z_dims']), 
                                            model_name='e_net', nb_units=params['e_units'])
            self.f_net = BayesianFullyConnectedNet(input_dim=params['z_dims'][0]+params['z_dims'][1]+1,
                                           output_dim = 2, model_name='f_net', nb_units=params['f_units'])
            self.h_net = BayesianFullyConnectedNet(input_dim=params['z_dims'][0]+params['z_dims'][2],
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

    def get_config(self):
        """Return the configuration of the CausalBGM model.

        Returns
        -------
        dict
            A dictionary with key ``"params"`` containing the full
            configuration dictionary passed at construction time.
        """

        return {
                "params": self.params,
        }

    def initialize_nets(self, print_summary = False):
        """Initialize all the networks in CausalBGM."""

        self.g_net(np.zeros((1, sum(self.params['z_dims']))))
        self.f_net(np.zeros((1, self.params['z_dims'][0]+self.params['z_dims'][1]+1)))
        self.h_net(np.zeros((1, self.params['z_dims'][0]+self.params['z_dims'][2])))
        if print_summary:
            print(self.g_net.summary())
            print(self.f_net.summary())    
            print(self.h_net.summary()) 

    # Update generative model for covariates V
    @tf.function
    def update_g_net(self, data_z, data_v, eps=1e-6):
        with tf.GradientTape() as gen_tape:
            g_net_output = self.g_net(data_z)
            mu_v = g_net_output[:,:self.params['v_dim']]
            if 'sigma_v' in self.params:
                sigma_square_v = self.params['sigma_v']**2
            else:
                sigma_square_v = tf.nn.softplus(g_net_output[:,-1]) + eps
            #loss = -log(p(x|z))
            loss_mse = tf.reduce_mean((data_v - mu_v)**2)
            loss_v = tf.reduce_sum((data_v - mu_v)**2, axis=1)/(2*sigma_square_v) + \
                    self.params['v_dim'] * tf.math.log(sigma_square_v)/2
            loss_v = tf.reduce_mean(loss_v)
            
            if self.params['use_bnn']:
                loss_kl = sum(self.g_net.losses)
                loss_v += loss_kl * self.params['kl_weight']

        # Calculate the gradients for generators and discriminators
        g_gradients = gen_tape.gradient(loss_v, self.g_net.trainable_variables)
        
        # Apply the gradients to the optimizer
        self.g_optimizer.apply_gradients(zip(g_gradients, self.g_net.trainable_variables))
        return loss_v, loss_mse
    
    # Update generative model for treatment X
    @tf.function
    def update_h_net(self, data_z, data_x, eps=1e-6):
        with tf.GradientTape() as gen_tape:
            data_z0 = data_z[:,:self.params['z_dims'][0]]
            data_z2 = data_z[:,sum(self.params['z_dims'][:2]):sum(self.params['z_dims'][:3])]
            h_net_output = self.h_net(tf.concat([data_z0, data_z2], axis=-1))
            mu_x = h_net_output[:,:1]
            if self.params['binary_treatment']:
                loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=data_x, 
                                                       logits=mu_x))
                loss_x =  loss
            else:
                if 'sigma_x' in self.params:
                    sigma_square_x = self.params['sigma_x']**2
                else:
                    sigma_square_x = tf.nn.softplus(h_net_output[:,-1]) + eps
                #loss = -log(p(x|z))
                loss = tf.reduce_mean((data_x - mu_x)**2)
                loss_x = tf.reduce_sum((data_x - mu_x)**2, axis=1)/(2*sigma_square_x) + \
                        tf.math.log(sigma_square_x)/2
                loss_x = tf.reduce_mean(loss_x)

            if self.params['use_bnn']:
                loss_kl = sum(self.h_net.losses)
                loss_x += loss_kl * self.params['kl_weight']
                
        # Calculate the gradients for generators and discriminators
        h_gradients = gen_tape.gradient(loss_x, self.h_net.trainable_variables)
        
        # Apply the gradients to the optimizer
        self.h_optimizer.apply_gradients(zip(h_gradients, self.h_net.trainable_variables))
        return loss_x, loss
    
    # Update generative model for outcome Y
    @tf.function
    def update_f_net(self, data_z, data_x, data_y, eps=1e-6):
        with tf.GradientTape() as gen_tape:
            data_z0 = data_z[:,:self.params['z_dims'][0]]
            data_z1 = data_z[:,self.params['z_dims'][0]:sum(self.params['z_dims'][:2])]
            f_net_output = self.f_net(tf.concat([data_z0, data_z1, data_x], axis=-1))
            mu_y = f_net_output[:,:1]
            if 'sigma_y' in self.params:
                sigma_square_y = self.params['sigma_y']**2
            else:
                sigma_square_y = tf.nn.softplus(f_net_output[:,-1]) + eps
            #loss = -log(p(y|z,x))
            loss_mse = tf.reduce_mean((data_y - mu_y)**2)
            loss_y = tf.reduce_sum((data_y - mu_y)**2, axis=1)/(2*sigma_square_y) + \
                    tf.math.log(sigma_square_y)/2
            loss_y = tf.reduce_mean(loss_y)
            
            if self.params['use_bnn']:
                loss_kl = sum(self.f_net.losses)
                loss_y += loss_kl * self.params['kl_weight']

        # Calculate the gradients for generators and discriminators
        f_gradients = gen_tape.gradient(loss_y, self.f_net.trainable_variables)
        
        # Apply the gradients to the optimizer
        self.f_optimizer.apply_gradients(zip(f_gradients, self.f_net.trainable_variables))
        return loss_y, loss_mse
    
    # Update posterior of latent variables Z
    @tf.function
    def update_latent_variable_sgd(self, data_x, data_y, data_v, data_z, eps=1e-6):
        with tf.GradientTape() as tape:
            
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

            loss_prior_z =  tf.reduce_sum(data_z**2, axis=1)/2
            loss_prior_z = tf.reduce_mean(loss_prior_z)

            loss_postrior_z = loss_pv_z + loss_px_z + loss_py_zx + loss_prior_z
            #loss_postrior_z = loss_postrior_z/self.params['v_dim']

        # Calculate the gradients
        posterior_gradients = tape.gradient(loss_postrior_z, [data_z])
        # Apply the gradients to the optimizer
        self.posterior_optimizer.apply_gradients(zip(posterior_gradients, [data_z]))
        return loss_postrior_z
    
#################################### EGM initialization ###########################################
    @tf.function
    def train_disc_step(self, data_z, data_v):
        epsilon_z = tf.random.uniform([],minval=0., maxval=1.)
        with tf.GradientTape(persistent=True) as disc_tape:
            with tf.GradientTape() as gp_tape:
                data_z_ = self.e_net(data_v)
                data_z_hat = data_z*epsilon_z + data_z_*(1-epsilon_z)
                data_dz_hat = self.dz_net(data_z_hat)

            data_dz_ = self.dz_net(data_z_)
            data_dz = self.dz_net(data_z)
            dz_loss = -tf.reduce_mean(data_dz) + tf.reduce_mean(data_dz_)

            # Calculate gradient penalty 
            grad_z = gp_tape.gradient(data_dz_hat, data_z_hat)
            grad_norm_z = tf.sqrt(tf.reduce_sum(tf.square(grad_z), axis=1))
            gpz_loss = tf.reduce_mean(tf.square(grad_norm_z - 1.0))
            
            d_loss = dz_loss + 10 * gpz_loss

        # Calculate the gradients for generators and discriminators
        d_gradients = disc_tape.gradient(d_loss, self.dz_net.trainable_variables)
        
        # Apply the gradients to the optimizer
        self.d_pre_optimizer.apply_gradients(zip(d_gradients, self.dz_net.trainable_variables))
        return dz_loss, d_loss
    
    @tf.function
    def train_gen_step(self, data_z, data_v, data_x, data_y):
        with tf.GradientTape(persistent=True) as gen_tape:
            sigma_square_loss = 0
            data_v_ = self.g_net(data_z)[:,:self.params['v_dim']]
            sigma_square_loss += tf.reduce_mean(tf.square(self.g_net(data_z)[:,-1]))
            data_z_ = self.e_net(data_v)
            
            data_z0 = data_z_[:,:self.params['z_dims'][0]]
            data_z1 = data_z_[:,self.params['z_dims'][0]:sum(self.params['z_dims'][:2])]
            data_z2 = data_z_[:,sum(self.params['z_dims'][:2]):sum(self.params['z_dims'][:3])]

            data_z__= self.e_net(data_v_)
            data_v__ = self.g_net(data_z_)[:,:self.params['v_dim']]
            
            data_dz_ = self.dz_net(data_z_)
            
            l2_loss_v = tf.reduce_mean((data_v - data_v__)**2)
            l2_loss_z = tf.reduce_mean((data_z - data_z__)**2)
            
            e_loss_adv = -tf.reduce_mean(data_dz_)

            data_y_ = self.f_net(tf.concat([data_z0, data_z1, data_x], axis=-1))[:,:1]
            sigma_square_loss += tf.reduce_mean(
                tf.square(self.f_net(tf.concat([data_z0, data_z1, data_x], axis=-1))[:,-1]))
            data_x_ = self.h_net(tf.concat([data_z0, data_z2], axis=-1))[:,:1]
            sigma_square_loss += tf.reduce_mean(
                tf.square(self.h_net(tf.concat([data_z0, data_z2], axis=-1))[:,-1]))

            if self.params['binary_treatment']:
                l2_loss_x = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=data_x, 
                                                       logits=data_x_))
            else:
                l2_loss_x = tf.reduce_mean((data_x_ - data_x)**2)
            l2_loss_y = tf.reduce_mean((data_y_ - data_y)**2)
            g_e_loss = e_loss_adv+(l2_loss_v + self.params['use_z_rec']*l2_loss_z) \
                        + (l2_loss_x+l2_loss_y) + 0.001 * sigma_square_loss

        # Calculate the gradients for generators and discriminators
        g_e_gradients = gen_tape.gradient(g_e_loss, self.g_net.trainable_variables+self.e_net.trainable_variables+\
                                        self.f_net.trainable_variables+self.h_net.trainable_variables)
        
        # Apply the gradients to the optimizer
        self.g_pre_optimizer.apply_gradients(zip(g_e_gradients, self.g_net.trainable_variables+self.e_net.trainable_variables+\
                                            self.f_net.trainable_variables+self.h_net.trainable_variables))
        return e_loss_adv, l2_loss_v, l2_loss_z, l2_loss_x, l2_loss_y, g_e_loss
    

    def egm_init(self, data, egm_n_iter=10000, batch_size=32, egm_batches_per_eval=500, verbose=1):
        """Run Encoding Generative Modeling (EGM) initialization.

        Initializes network parameters by jointly fitting the generator,
        encoder, and discriminator before the main iterative optimization.
        This is optional and can be skipped by setting `use_egm_init` to False in :meth:`fit`.

        Parameters
        ----------
        data : tuple of np.ndarray
            A triplet ``(data_x, data_y, data_v)``.
        egm_n_iter : int, default=10000
            Number of EGM mini-batch iterations.
        batch_size : int, default=32
            Mini-batch size.
        egm_batches_per_eval : int, default=500
            Evaluate and log every this many iterations.
        verbose : int, default=1
            Verbosity level (0 = silent).
        """
        data_x, data_y, data_v = data
        
        print('EGM Initialization Starts ...')
        for batch_iter in range(egm_n_iter+1):
            # Update model parameters of Discriminator
            for _ in range(self.params['g_d_freq']):
                batch_idx = np.random.choice(len(data_x), batch_size, replace=False)
                batch_z = self.z_sampler.get_batch(batch_size)
                batch_v = data_v[batch_idx,:]
                dz_loss, d_loss = self.train_disc_step(batch_z, batch_v)

            # Update model parameters of G, H, F with SGD
            batch_z = self.z_sampler.get_batch(batch_size)
            batch_idx = np.random.choice(len(data_x), batch_size, replace=False)
            batch_x = data_x[batch_idx,:]
            batch_y = data_y[batch_idx,:]
            batch_v = data_v[batch_idx,:]
            e_loss_adv, l2_loss_v, l2_loss_z, l2_loss_x, l2_loss_y, g_e_loss = self.train_gen_step(batch_z, batch_v, batch_x, batch_y)
            if batch_iter % egm_batches_per_eval == 0:
                
                loss_contents = (
                    'EGM Initialization Iter [%d] : e_loss_adv [%.4f], l2_loss_v [%.4f], l2_loss_z [%.4f], '
                    'l2_loss_x [%.4f], l2_loss_y [%.4f], g_e_loss [%.4f], dz_loss [%.4f], d_loss [%.4f]'
                    % (batch_iter, e_loss_adv, l2_loss_v, l2_loss_z, l2_loss_x, l2_loss_y, g_e_loss, dz_loss, d_loss)
                )
                if verbose:
                    print(loss_contents)
                if self.params['save_res']:
                    causal_pre, mse_x, mse_y, mse_v, data_x_pred, data_y_pred, data_v_pred = self.evaluate(data = data)
                    causal_pre = causal_pre.numpy()
                    save_data('{}/causal_pre_egm_init_iter-{}.txt'.format(self.save_dir, batch_iter), causal_pre)
        print('EGM Initialization Ends.')
#################################### EGM initialization #############################################

    def fit(self, data,
            batch_size=32, epochs=100, epochs_per_eval=5, startoff=0,
            use_egm_init=True, egm_n_iter=30000, egm_batches_per_eval=500, save_format='txt', verbose=1):
        """Train the CausalBGM model on observed data.

        The training procedure consists of two phases:

        1. **EGM initialization** (optional) — warm-start by jointly training encoder and
           generator with adversarial losses to obtain a good starting point
           for the latent variables and model parameters. This phase is optional and can be skipped by setting `use_egm_init` to False.
        2. **Iterative optimization** — alternates between updating the
           generator network parameters :math:`\\theta` and the per-sample
           latent variables :math:`Z` via SGD.
           
        Parameters
        ----------
        data : tuple of np.ndarray
            A triplet ``(data_x, data_y, data_v)`` where

            - ``data_x`` has shape ``(n, 1)`` — treatment variable.
            - ``data_y`` has shape ``(n, 1)`` — outcome variable.
            - ``data_v`` has shape ``(n, v_dim)`` — covariates.
        batch_size : int, default=32
            Mini-batch size.
        epochs : int, default=100
            Number of training epochs for the iterative phase.
        epochs_per_eval : int, default=5
            Evaluate and (optionally) save every this many epochs.
        startoff : int, default=0
            Only start tracking the best model after this many epochs.
        use_egm_init : bool, default=True
            Whether to run EGM initialization before iterative training.
        egm_n_iter : int, default=30000
            Number of EGM initialization iterations.
        egm_batches_per_eval : int, default=500
            Evaluate EGM every this many iterations.
        save_format : str, default='txt'
            File format for saving causal estimates (``'txt'`` or ``'npy'``).
        verbose : int, default=1
            Verbosity level.
        """
        
        data_x, data_y, data_v = data
        
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
            data_z_init = np.random.normal(0, 1, size = (len(data_x), sum(self.params['z_dims']))).astype('float32')

        self.data_z = tf.Variable(data_z_init, name="Latent Variable",trainable=True)
        
        best_loss = np.inf
        print('Iterative Updating Starts ...')
        for epoch in range(epochs+1):
            sample_idx = np.random.choice(len(data_x), len(data_x), replace=False)
            
            # Create a progress bar for batches
            with tqdm(total=len(data_x) // batch_size, desc=f"Epoch {epoch}/{epochs}", unit="batch") as batch_bar:
                for i in range(0,len(data_x) - batch_size + 1,batch_size): ## Skip the incomplete last batch
                    batch_idx = sample_idx[i:i+batch_size]
                    # Update model parameters of G, H, F with SGD
                    batch_z = tf.Variable(tf.gather(self.data_z, batch_idx, axis = 0), name='batch_z', trainable=True)
                    batch_x = data_x[batch_idx,:]
                    batch_y = data_y[batch_idx,:]
                    batch_v = data_v[batch_idx,:]
                    loss_v, loss_mse_v = self.update_g_net(batch_z, batch_v)
                    loss_x, loss_mse_x = self.update_h_net(batch_z, batch_x)
                    loss_y, loss_mse_y = self.update_f_net(batch_z, batch_x, batch_y)

                    # Update Z by maximizing a posterior or posterior mean
                    loss_postrior_z = self.update_latent_variable_sgd(batch_x, batch_y, batch_v, batch_z)

                    # Update data_z with updated batch_z
                    self.data_z.scatter_nd_update(
                        indices=tf.expand_dims(batch_idx, axis=1),
                        updates=batch_z                             
                    )
                    
                    # Update the progress bar with the current loss information
                    loss_contents = (
                        'loss_px_z: [%.4f], loss_mse_x: [%.4f], loss_py_z: [%.4f], '
                        'loss_mse_y: [%.4f], loss_pv_z: [%.4f], loss_mse_v: [%.4f], loss_postrior_z: [%.4f]'
                        % (loss_x, loss_mse_x, loss_y, loss_mse_y, loss_v, loss_mse_v, loss_postrior_z)
                    )
                    batch_bar.set_postfix_str(loss_contents)
                    batch_bar.update(1)
            
            # Evaluate the full training data and print metrics for the epoch
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

    @tf.function
    def evaluate(self, data, data_z=None, nb_intervals=200):
        """Evaluate the model on observed data and compute causal estimates.

        For **binary treatment**, the Individual Treatment Effect (ITE) is
        returned.  For **continuous treatment**, the Average Dose-Response
        Function (ADRF) is computed over a grid of treatment values.

        Parameters
        ----------
        data : tuple of np.ndarray
            A triplet ``(data_x, data_y, data_v)``.
        data_z : tf.Tensor or None, optional
            Latent variables.  If ``None``, inferred via the encoder.
        nb_intervals : int, default=200
            Number of grid points for the dose-response curve
            (continuous treatment only).

        Returns
        -------
        causal_estimate : tf.Tensor
            ITE with shape ``(n, 1)`` (binary) or ADRF with shape
            ``(nb_intervals,)`` (continuous).
        mse_x : tf.Tensor
            MSE for the treatment model.
        mse_y : tf.Tensor
            MSE for the outcome model.
        mse_v : tf.Tensor
            MSE for the covariate model.
        data_x_pred : tf.Tensor
            Predicted treatment values.
        data_y_pred : tf.Tensor
            Predicted outcome values.
        data_v_pred : tf.Tensor
            Predicted covariates.
        """
        data_x, data_y, data_v = data
        if data_z is None:
            data_z = self.e_net(data_v)
        data_z0 = data_z[:,:self.params['z_dims'][0]]
        data_z1 = data_z[:,self.params['z_dims'][0]:sum(self.params['z_dims'][:2])]
        data_z2 = data_z[:,sum(self.params['z_dims'][:2]):sum(self.params['z_dims'][:3])]
        data_v_pred = self.g_net(data_z)[:,:self.params['v_dim']]
        data_y_pred = self.f_net(tf.concat([data_z0, data_z1, data_x], axis=-1))[:,:1]
        data_x_pred = self.h_net(tf.concat([data_z0, data_z2], axis=-1))[:,:1]
        if self.params['binary_treatment']:
            data_x_pred = tf.sigmoid(data_x_pred)
        mse_v = tf.reduce_mean((data_v-data_v_pred)**2)
        mse_x = tf.reduce_mean((data_x-data_x_pred)**2)
        mse_y = tf.reduce_mean((data_y-data_y_pred)**2)
        
        if self.params['binary_treatment']:
            # Individual treatment effect (ITE) && average treatment effect (ATE)
            y_pred_pos = self.f_net(tf.concat([data_z0, data_z1, np.ones((len(data_x),1))], axis=-1))[:,:1]
            y_pred_neg = self.f_net(tf.concat([data_z0, data_z1, np.zeros((len(data_x),1))], axis=-1))[:,:1]
            ite_pre = y_pred_pos-y_pred_neg
            return ite_pre, mse_x, mse_y, mse_v, data_x_pred, data_y_pred, data_v_pred
        else:
            # Average dose response function (ADRF)
            # Compute 5% and 95% quantiles from data_x
            x_min = tfp.stats.percentile(data_x, 5.0)
            x_max = tfp.stats.percentile(data_x, 95.0)
            x_values = tf.linspace(x_min, x_max, nb_intervals)
            
            def compute_dose_response(x):
                data_x_tile = tf.fill([tf.shape(data_x)[0], 1], x)
                data_x_tile = tf.cast(data_x_tile, tf.float32)
                y_pred = self.f_net(tf.concat([data_z0, data_z1, data_x_tile], axis=-1))[:, :1]
                return tf.reduce_mean(y_pred)
        
            dose_response = tf.map_fn(compute_dose_response, x_values, fn_output_signature=tf.float32)
            
            return dose_response, mse_x, mse_y, mse_v, data_x_pred, data_y_pred, data_v_pred

    # Predict with MCMC sampling
    def predict(self, data, alpha=0.01, n_mcmc=3000, x_values=None, q_sd=1.0, sample_y=True, bs=3000):
        """
        Evaluate the model on the test data and provide both point estimates and posterior intervals for causal effects.
        - For binary treatment, the Individual Treatment Effect (ITE) is estimated.
        - For continuous treatment, the Average Dose Response Function (ADRF) is estimated.

        Parameters
        ----------
        data : list
            Input data containing [data_x, data_y, data_v].
        alpha : float, default=0.01
            Significance level for the posterior interval.
        n_mcmc : int, default=3000
            Number of posterior MCMC samples to draw.
        x_values : list of floats or np.ndarray
            Treatment values for dose-response prediction.
            Required for continuous treatment.
        q_sd : float, default=1.0
            Proposal standard deviation used by the Metropolis-Hastings sampler.
        sample_y : bool, default=True
            Whether to sample from the outcome variance model.
        bs : int, default=3000
            Batch size for processing posterior samples.

        Returns
        -------
        effect : np.ndarray
            Binary treatment: point estimate of ITE with shape ``(n,)``.
            Continuous treatment: ADRF estimate with shape ``(len(x_values),)``.
        pos_int : np.ndarray
            Posterior interval with shape ``(n, 2)`` for ITE, or
            ``(len(x_values), 2)`` for ADRF.
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
        for i in range(0, data[0].shape[0], bs):
            batch_data = (data[0][i:i + bs], data[1][i:i + bs], data[2][i:i + bs])
            batch_posterior_z = self.metropolis_hastings_sampler(batch_data , n_keep=n_mcmc, q_sd=q_sd)
            causal_effect_batch = self.infer_from_latent_posterior(batch_posterior_z, x_values=x_values, sample_y=sample_y).numpy()
            causal_effects.append(causal_effect_batch)
        causal_effects = np.concatenate(causal_effects, axis=-1)
        
        # Estimate the posterior interval with user-specific significance level alpha

        if self.params['binary_treatment']:
            # For binary treatment: Individual Treatment Effect (ITE), causal effect has shape (n_mcmc, n)
            ITE = np.mean(causal_effects, axis=0)
            posterior_interval_upper = np.quantile(causal_effects, 1-alpha/2, axis=0)
            posterior_interval_lower = np.quantile(causal_effects, alpha/2, axis=0)
            pos_int = np.stack([posterior_interval_lower, posterior_interval_upper], axis=1)
            return ITE, pos_int
        else:
            # For continuous treatment: Average Dose Response Function (ADRF), causal effect has shape (len(x_values), n_mcmc, n)
            causal_effects = np.mean(causal_effects, axis=-1)
            ADRF = np.mean(causal_effects, axis=1)
            posterior_interval_upper = np.quantile(causal_effects, 1-alpha/2, axis=1)
            posterior_interval_lower = np.quantile(causal_effects, alpha/2, axis=1)
            pos_int = np.stack([posterior_interval_lower, posterior_interval_upper], axis=1)
            return ADRF, pos_int

        
    @tf.function
    def infer_from_latent_posterior(self, data_posterior_z, x_values=None, sample_y=True, eps=1e-6):
        """Infer causal estimate on the test data and give estimation interval and posterior latent variables. ITE is estimated for binary treatment and ADRF is estimated for continous treatment.
        data_posterior_z: (np.ndarray): Posterior latent variables with shape (n_samples, n, p), where p is the dimension of Z.
        x_values: (list of floats or np.ndarray): Number of intervals for the dose response function.
        sample_y: (bool): consider the variance function in outcome generative model.
        return (np.ndarray): 
            ITE with shape (n_samples, n) containing all the MCMC samples.
            ADRF with shape (len(x_values), n_samples, n) containing all the MCMC samples for each treatment value.
        """

        # Extract the components of Z for X,Y
        data_z0 = data_posterior_z[:,:,:self.params['z_dims'][0]]
        data_z1 = data_posterior_z[:,:,self.params['z_dims'][0]:sum(self.params['z_dims'][:2])]
        data_z2 = data_posterior_z[:,:,sum(self.params['z_dims'][:2]):sum(self.params['z_dims'][:3])]

        if self.params['binary_treatment']:
            
            # Extract mean and sigma^2 of positive samples both with shape (n_keep, n_test)
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
            
            # Extract mean and sigma^2 of negative samples both with shape (n_keep, n_test)
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
                    
                return y_pred_all
            
            dose_response = tf.map_fn(compute_dose_response, x_values, fn_output_signature=tf.float32)
            
            return dose_response

    @tf.function
    def get_log_posterior(self, data_x, data_y, data_v, data_z, eps=1e-6):
        """
        Calculate log posterior.
        data_x: (np.ndarray): Input data with shape (n, 1), where p is the dimension of X.
        data_y: (np.ndarray): Input data with shape (n, 1), where q is the dimension of Y.
        data_v: (np.ndarray): Input data with shape (n, p), where r is the dimension of V.
        data_z: (np.ndarray): Input data with shape (n, q), where q is the dimension of Z.
        return (np.ndarray): Log posterior with shape (n, ).
        """
        data_z0 = data_z[:,:self.params['z_dims'][0]]
        data_z1 = data_z[:,self.params['z_dims'][0]:sum(self.params['z_dims'][:2])]
        data_z2 = data_z[:,sum(self.params['z_dims'][:2]):sum(self.params['z_dims'][:3])]

        mu_v = self.g_net(data_z)[:,:self.params['v_dim']]
        if 'sigma_v' in self.params:
            sigma_square_v = self.params['sigma_v']**2
        else:
            sigma_square_v = tf.nn.softplus(self.g_net(data_z)[:,-1]) + eps

        mu_x = self.h_net(tf.concat([data_z0, data_z2], axis=-1))[:,:1]
        if 'sigma_x' in self.params:
            sigma_square_x = self.params['sigma_x']**2
        else:
            sigma_square_x = tf.nn.softplus(self.h_net(tf.concat([data_z0, data_z2], axis=-1))[:,-1]) + eps

        mu_y = self.f_net(tf.concat([data_z0, data_z1, data_x], axis=-1))[:,:1]
        if 'sigma_y' in self.params:
            sigma_square_y = self.params['sigma_y']**2
        else:
            sigma_square_y = tf.nn.softplus(self.f_net(tf.concat([data_z0, data_z1, data_x], axis=-1))[:,-1]) + eps

        loss_pv_z = tf.reduce_sum((data_v - mu_v)**2, axis=1)/(2*sigma_square_v) + \
                self.params['v_dim'] * tf.math.log(sigma_square_v)/2
        
        if self.params['binary_treatment']:
            loss_px_z = tf.squeeze(tf.nn.sigmoid_cross_entropy_with_logits(labels=data_x,logits=mu_x))
        else:
            loss_px_z = tf.reduce_sum((data_x - mu_x)**2, axis=1)/(2*sigma_square_x) + \
                    tf.math.log(sigma_square_x)/2

        loss_py_zx = tf.reduce_sum((data_y - mu_y)**2, axis=1)/(2*sigma_square_y) + \
                tf.math.log(sigma_square_y)/2

        loss_prior_z =  tf.reduce_sum(data_z**2, axis=1)/2

        loss_postrior_z = loss_pv_z + loss_px_z + loss_py_zx + loss_prior_z

        log_posterior = -loss_postrior_z
        return log_posterior


    def metropolis_hastings_sampler(self, data, initial_q_sd = 1.0, q_sd = None, burn_in = 5000, n_keep = 3000, target_acceptance_rate=0.25, tolerance=0.05, adjustment_interval=50, adaptive_sd=None, window_size=100):
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

            # Compute the acceptance ratio
            proposed_log_posterior = self.get_log_posterior(data_x, data_y, data_v, proposed_state)
            current_log_posterior  = self.get_log_posterior(data_x, data_y, data_v, current_state)
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
