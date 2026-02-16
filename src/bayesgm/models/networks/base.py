import tensorflow as tf
import tensorflow_probability as tfp

class BaseFullyConnectedNet(tf.keras.Model):
    """Basic multi-layer perceptron (MLP) with optional batch normalization.
    """
    def __init__(self, input_dim, output_dim, model_name, nb_units=[256, 256, 256], batchnorm=False):  
        super(BaseFullyConnectedNet, self).__init__()
        self.input_layer = tf.keras.layers.Input((input_dim,))
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_name = model_name
        self.nb_units = nb_units
        self.batchnorm = batchnorm
        self.all_layers = []
        """ Builds the FC stacks. """
        for i in range(len(nb_units) + 1):
            units = self.output_dim if i == len(nb_units) else self.nb_units[i]
            fc_layer = tf.keras.layers.Dense(
                units = units,
                activation = None,
                kernel_regularizer = tf.keras.regularizers.L2(1e-4),
                bias_regularizer = tf.keras.regularizers.L2(1e-4)
            )   
            norm_layer = tf.keras.layers.BatchNormalization()
            self.all_layers.append([fc_layer, norm_layer])
        
        self.out = self.call(self.input_layer)

    def call(self, inputs, training=True):
        """ Return the output of the Generator.
        Args:
            inputs: tensor with shape [batch_size, input_dim]
        Returns:
            Output of Generator.
            float32 tensor with shape [batch_size, output_dim]
        """
        for i, layers in enumerate(self.all_layers[:-1]):
            # Run inputs through the sublayers.
            fc_layer, norm_layer = layers
            with tf.name_scope("%s_layer_%d" % (self.model_name, i+1)):
                x = fc_layer(inputs) if i==0 else fc_layer(x)
                if self.batchnorm:
                    x = norm_layer(x)
                x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        fc_layer, norm_layer = self.all_layers[-1]
        with tf.name_scope("%s_layer_ouput" % self.model_name):
            output = fc_layer(x)
            # No activation func at last layer
            #x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        return output

class BaseVariationalNet(tf.keras.Model):
    """Standard (non-Bayesian) variational network with diagonal covariance.
    Outputs both a mean and a variance for each output dimension, enabling
    reparameterized sampling.
    """
    def __init__(self, input_dim, output_dim, model_name, nb_units=[256, 256, 256]):
        """
        Initializes the model layers.

        Args:
            input_dim (int): The dimension of the input features.
            output_dim (int): The dimension of the output.
            model_name (str): A name for the model, used for scoping.
            nb_units (list): A list of integers specifying the number of units in each hidden layer.
        """
        super(BaseVariationalNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_name = model_name
        self.nb_units = nb_units
        self.all_layers = []
        
        # Batch normalization layer to stabilize inputs
        self.norm_layer = tf.keras.layers.BatchNormalization()
        
        # Define standard Dense layers for each hidden layer
        for i in range(len(nb_units)):
            dense_layer = tf.keras.layers.Dense(
                units=self.nb_units[i],
                activation=None  # Activation will be applied separately
            )
            self.all_layers.append(dense_layer)
            
        # Output layer for the mean prediction
        self.mean_layer = tf.keras.layers.Dense(
                units=self.output_dim,
                activation=None  # Linear activation for regression output
            )
            
        # Output layer for the variance prediction
        self.var_layer = tf.keras.layers.Dense(
                units=self.output_dim,
                activation=None  # Linear activation
            )
            
    def call(self, inputs, eps=1e-6, training=True):
        """ Return the output of the Bayesian network. """
        x = self.norm_layer(inputs, training=training)
        for i, bayesian_layer in enumerate(self.all_layers):
            with tf.name_scope("%s_layer_%d" % (self.model_name, i+1)):
                x = bayesian_layer(x)
                x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        
        # Final layer without activation
        with tf.name_scope("%s_layer_output" % self.model_name):
            mean = self.mean_layer(x)
            var = self.var_layer(x)
            var = tf.nn.softplus(var) + eps
        return mean, var
    
    def reparameterize(self, mean, var):
        # Sample from a standard normal distribution
        eps = tf.random.normal(shape=tf.shape(mean))
        # Return the reparameterized sample
        return eps * tf.sqrt(var) + mean

class BaseVariationalLowRankNet(tf.keras.Model):
    """Standard (non-Bayesian) variational network with low-rank covariance.
    
    Outputs a mean, diagonal variance, and a low-rank factor U so that the
    covariance is Σ(z) = diag(var) + U Uᵀ.  BayesianVariationalLowRankNet
    is the Bayesian counterpart of this class.
    """
    def __init__(self, input_dim, output_dim, model_name, nb_units=[256, 256, 256], rank=2):
        """
        Args:
            input_dim (int): Dimension of input features.
            output_dim (int): Dimension of output features.
            model_name (str): Name of the model.
            nb_units (list): Number of units per hidden layer.
            rank (int): Rank of the low-rank covariance matrix.
        """
        super(BaseVariationalLowRankNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_name = model_name
        self.nb_units = nb_units
        self.rank = rank

        self.all_layers = []
        self.norm_layer = tf.keras.layers.BatchNormalization()

        # Define regular Dense layers for each fully connected layer
        for i in range(len(nb_units)):
            dense_layer = tf.keras.layers.Dense(
                units=self.nb_units[i],
                activation=None,
                kernel_regularizer=tf.keras.regularizers.L2(1e-4),
                bias_regularizer=tf.keras.regularizers.L2(1e-4)
            )
            self.all_layers.append(dense_layer)

        # Output layers
        self.mean_layer = tf.keras.layers.Dense(
            units=self.output_dim,
            activation=None,
            kernel_regularizer=tf.keras.regularizers.L2(1e-4),
            bias_regularizer=tf.keras.regularizers.L2(1e-4)
        )

        # Variance layer: Outputs per-dimension variance
        self.var_layer = tf.keras.layers.Dense(
            units=self.output_dim,  # Per-dimension variance
            activation=None,
            kernel_regularizer=tf.keras.regularizers.L2(1e-4),
            bias_regularizer=tf.keras.regularizers.L2(1e-4)
        )

        # Low-rank factor layer: Outputs (batch, output_dim, rank)
        self.low_rank_layer = tf.keras.layers.Dense(
            units=self.output_dim * self.rank,
            activation=None,
            kernel_regularizer=tf.keras.regularizers.L2(1e-4),
            bias_regularizer=tf.keras.regularizers.L2(1e-4)
        )

    def call(self, inputs, training=True):
        """ Return the output of the FCN network. """
        x = self.norm_layer(inputs)
        for i, dense_layer in enumerate(self.all_layers):
            with tf.name_scope(f"{self.model_name}_layer_{i+1}"):
                x = dense_layer(x)
                x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

        # Compute mean
        with tf.name_scope(f"{self.model_name}_layer_output"):
            mean = self.mean_layer(x)  # Shape: (batch, p)

            # Per-dimension variance
            var_raw = self.var_layer(x)  # Shape: (batch, p)
            var_diag = tf.nn.softplus(var_raw) + 1e-6  # Ensure positive variance

            # Low-rank matrix U(z)
            U_flat = self.low_rank_layer(x)  # Shape: (batch, p * rank)
            U = tf.reshape(U_flat, [-1, self.output_dim, self.rank])  # Shape: (batch, p, rank)
        
        return mean, var_diag, U

    def reparameterize(self, mean, var_diag, U):
        """
        Performs reparameterization using the low-rank structure:
        z = μ + D^(1/2) * ε₁ + U * ε₂
        """
        batch_size = tf.shape(mean)[0]

        # Sample ε₁ ~ N(0, I_p)
        eps1 = tf.random.normal(shape=(batch_size, self.output_dim))

        # Sample ε₂ ~ N(0, I_r)
        eps2 = tf.random.normal(shape=(batch_size, self.rank))

        # Diagonal component
        diag_sample = tf.sqrt(var_diag) * eps1  # (batch, output_dim)

        # Low-rank component: batch matrix multiply U @ eps2[i]
        eps2_expanded = tf.expand_dims(eps2, -1)  # (batch, rank, 1)
        low_rank_sample = tf.matmul(U, eps2_expanded)  # (batch, output_dim, 1)
        low_rank_sample = tf.squeeze(low_rank_sample, -1)  # (batch, output_dim)

        # Final reparameterized sample
        return mean + diag_sample + low_rank_sample
    
    def compute_covariance_inverse(self, var_diag, U):
        """
        Computes the inverse of the covariance matrix Σ(z) = diag(var_diag) + UU^T
        using the Woodbury identity.
        Args:
            var_diag: Tensor of shape (batch, p), diagonal variance.
            U: Tensor of shape (batch, p, rank), low-rank component.
        Returns:
            Sigma_inv: Tensor of shape (batch, p, p), inverse of covariance matrix.
        """
        # D_inv = diag(1/var_diag)
        D_inv = tf.linalg.diag(1.0 / var_diag)  # Shape: (batch, p, p)

        # Compute U^T D^-1: Each column of U is divided by var_diag (broadcasting)
        U_T_D_inv = tf.transpose(U, perm=[0, 2, 1]) / tf.expand_dims(var_diag, axis=1)  # Shape: (batch, rank, p)

        # Compute middle term: M = I + U^T D^-1 U
        M = tf.matmul(U_T_D_inv, U)  # Shape: (batch, rank, rank)
        M_inv = tf.linalg.inv(tf.eye(self.rank) + M)  # Shape: (batch, rank, rank)

        # Compute Σ^{-1} using Woodbury identity: D^-1 - D^-1 U M^-1 U^T D^-1
        Sigma_inv = D_inv - tf.matmul(tf.transpose(U_T_D_inv, perm=[0, 2, 1]), tf.matmul(M_inv, U_T_D_inv))

        return Sigma_inv

    def compute_log_det(self, var_diag, U):
        """
        Computes the log determinant of Σ(z) = diag(var_diag) + UU^T
        using Sylvester's determinant theorem.

        Args:
            var_diag: Tensor of shape (batch, p), diagonal variance.
            U: Tensor of shape (batch, p, rank), low-rank component.

        Returns:
            log_det: Tensor of shape (batch,), log determinant of Σ(z).
        """
        # log(det(D)) = sum(log(diagonal elements))
        log_det_D = tf.reduce_sum(tf.math.log(var_diag), axis=-1)  # Shape: (batch,)

        # Compute M = I + U^T D^-1 U
        U_T_D_inv = tf.transpose(U, perm=[0, 2, 1]) / tf.expand_dims(var_diag, axis=1)  # Shape: (batch, rank, p)
        M = tf.matmul(U_T_D_inv, U)  # Shape: (batch, rank, rank)

        # log(det(I + M))
        log_det_M = tf.linalg.logdet(tf.eye(self.rank) + M)  # Shape: (batch,)

        # Apply Sylvester's theorem: log(det(Σ)) = log(det(D)) + log(det(I + M))
        log_det = log_det_D + log_det_M

        return log_det

    def transfer_weights_from_bayesian(self, bayesian_model, method='mean'):
        """
        Transfers weights from a BayesianVariationalLowRankNet to this model.

        This function can operate in two modes, controlled by the `method` argument:
        - 'mean': Extracts the posterior mean of the Bayesian weights and assigns them.
                  This creates a single deterministic model representing the "average"
                  function learned by the BNN.
        - 'sample': Draws a single random sample from the posterior weight distributions
                    and assigns it. This is useful for creating ensemble members.

        Args:
            bayesian_model: BayesianVariationalLowRankNet model with trained weights.
            method (str): The transfer method. Must be either 'mean' or 'sample'.
                          Defaults to 'mean'.
        """
        if method not in ['mean', 'sample']:
            raise ValueError(f"Invalid method '{method}'. Must be either 'mean' or 'sample'.")

        print(f"Starting weight transfer using the '{method}' method...")

        # Transfer weights from hidden layers
        for i, (fcn_layer, bayesian_layer) in enumerate(zip(self.all_layers, bayesian_model.all_layers)):
            if method == 'mean':
                kernel_weights = bayesian_layer.kernel_posterior.mean()
                bias_weights = bayesian_layer.bias_posterior.mean()
            else:  # method == 'sample'
                kernel_weights = bayesian_layer.kernel_posterior.sample()
                bias_weights = bayesian_layer.bias_posterior.sample()
            
            fcn_layer.kernel.assign(kernel_weights)
            fcn_layer.bias.assign(bias_weights)
            print(f"Transferred weights for hidden layer {i+1}")

        # Define a helper function for brevity
        def transfer_output_layer(fcn_layer, bayesian_layer, layer_name):
            if method == 'mean':
                kernel_weights = bayesian_layer.kernel_posterior.mean()
                bias_weights = bayesian_layer.bias_posterior.mean()
            else: # method == 'sample'
                kernel_weights = bayesian_layer.kernel_posterior.sample()
                bias_weights = bayesian_layer.bias_posterior.sample()
            fcn_layer.kernel.assign(kernel_weights)
            fcn_layer.bias.assign(bias_weights)
            print(f"Transferred weights for {layer_name} layer")

        # Transfer weights from output layers
        transfer_output_layer(self.mean_layer, bayesian_model.mean_layer, "mean")
        transfer_output_layer(self.var_layer, bayesian_model.var_layer, "variance")
        transfer_output_layer(self.low_rank_layer, bayesian_model.low_rank_layer, "low-rank")

        # Batch normalization parameters are not distributions, so we transfer them directly.
        if hasattr(bayesian_model.norm_layer, 'moving_mean') and bayesian_model.norm_layer.moving_mean is not None:
            self.norm_layer.moving_mean.assign(bayesian_model.norm_layer.moving_mean)
            self.norm_layer.moving_variance.assign(bayesian_model.norm_layer.moving_variance)
            self.norm_layer.gamma.assign(bayesian_model.norm_layer.gamma)
            self.norm_layer.beta.assign(bayesian_model.norm_layer.beta)
            print("Transferred batch normalization parameters")

        print(f"Weight transfer using '{method}' method completed successfully!")

class Discriminator(tf.keras.Model):
    """Fully connected discriminator network.
    
    Outputs a scalar logit for each input using tanh activations
    and optional batch normalization.
    """
    def __init__(self, input_dim, model_name, nb_units=[256, 256], batchnorm=True):  
        super(Discriminator, self).__init__()
        self.input_layer = tf.keras.layers.Input((input_dim,))
        self.input_dim = input_dim
        self.model_name = model_name
        self.nb_units = nb_units
        self.batchnorm = batchnorm
        self.all_layers = []
        """Builds the FC stacks."""
        for i in range(len(self.nb_units)+1):
            units = 1 if i == len(self.nb_units) else self.nb_units[i]
            fc_layer = tf.keras.layers.Dense(
                units = units,
                activation = None
            )
            norm_layer = tf.keras.layers.BatchNormalization()

            self.all_layers.append([fc_layer, norm_layer])
        self.out = self.call(self.input_layer)

    def call(self, inputs, training=True):
        """Return the output of the Discriminator network.
        Args:
            inputs: tensor with shape [batch_size, input_dim]
        Returns:
            Output of Discriminator.
            float32 tensor with shape [batch_size, 1]
        """
            
        for i, layers in enumerate(self.all_layers[:-1]):
            # Run inputs through the sublayers.
            fc_layer, norm_layer = layers
            with tf.name_scope("%s_layer_%d" % (self.model_name,i+1)):
                x = fc_layer(inputs) if i==0 else fc_layer(x)
                if self.batchnorm:
                    x = norm_layer(x)
                x = tf.keras.activations.tanh(x)
                #x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        fc_layer, norm_layer = self.all_layers[-1]
        with tf.name_scope("%s_layer_ouput" % self.model_name):
            output = fc_layer(x)
        return output
    
class MCMCFullyConnectedNet(BaseFullyConnectedNet):
    """
    A fully connected network intended for use in FullMCMCCausalBGM.
    It is structurally identical to BaseFullyConnectedNet, but we add helper
    methods to compute the log prior and to perform a forward pass with an
    explicit set of weights.
    """
    def __init__(self, *args, **kwargs):
        super(MCMCFullyConnectedNet, self).__init__(*args, **kwargs)

    @tf.function
    def call_with_weights(self, inputs, flattened_weights):
        """
        Performs a forward pass using a provided set of flattened weights
        in a stateless, @tf.function-compatible manner.
        """
        # Unflatten the weights tensor into a list of tensors with the correct shapes
        # for each layer's kernel and bias.
        weight_shapes = [w.shape for w in self.trainable_variables]
        unflattened_weights = []
        start_idx = 0
        for shape in weight_shapes:
            size = tf.reduce_prod(shape)
            w = tf.reshape(flattened_weights[start_idx : start_idx + size], shape)
            unflattened_weights.append(w)
            start_idx += size

        # Perform the forward pass manually using the unflattened weights.
        x = inputs
        weight_idx = 0
        # Iterate through hidden layers
        for i, (fc_layer, norm_layer) in enumerate(self.all_layers[:-1]):
            kernel = unflattened_weights[weight_idx]
            bias = unflattened_weights[weight_idx + 1]
            weight_idx += 2
            
            x = tf.matmul(x, kernel) + bias
            if self.batchnorm:
                # Batch norm layers have non-trainable state (moving mean/variance)
                # but calling them like this is generally graph-compatible.
                x = norm_layer(x, training=False) # Use inference mode for stability
            x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        
        # Process the final output layer
        final_kernel = unflattened_weights[weight_idx]
        final_bias = unflattened_weights[weight_idx + 1]
        output = tf.matmul(x, final_kernel) + final_bias
        
        return output

    @tf.function
    def log_prior(self, flattened_weights):
        """Calculates the log prior probability of the weights (L2 regularizer)."""
        # Standard normal prior
        prior_dist = tfp.distributions.Normal(loc=0., scale=1.)
        return tf.reduce_sum(prior_dist.log_prob(flattened_weights))

def run_mcmc_for_net(net, x_train, y_train, likelihood_fn, initial_state, num_samples=1000, num_burnin_steps=500):
    """
    Runs Hamiltonian Monte Carlo to sample weights for a given network.

    Args:
        net: An instance of MCMCFullyConnectedNet.
        x_train: Training input data.
        y_train: Training target data.
        likelihood_fn: A function(y_true, y_pred) that returns the log-likelihood.
        initial_state: A list of tensors representing the initial weights.
        num_samples: Number of samples to return.
        num_burnin_steps: Number of burn-in steps.

    Returns:
        A tensor of weight samples.
    """
    
    # Flatten the initial state for the sampler
    flat_initial_state = tf.concat([tf.reshape(w, [-1]) for w in initial_state], axis=0)

    # Define the target log probability function (posterior)
    def target_log_prob_fn(weights):
        # Log Prior P(theta)
        log_prior = net.log_prior(weights)
        
        # Log Likelihood P(D|theta)
        y_pred = net.call_with_weights(x_train, weights)
        log_likelihood = likelihood_fn(y_train, y_pred)

        return log_prior + log_likelihood

    # Set up the HMC sampler
    hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob_fn,
        step_size=0.01,
        num_leapfrog_steps=3
    )
    
    # Use an adaptive step size for better performance
    adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
        inner_kernel=hmc_kernel,
        num_adaptation_steps=int(num_burnin_steps * 0.8)
    )

    # Run the chain
    @tf.function
    def run_chain():
        samples, kernel_results = tfp.mcmc.sample_chain(
            num_results=num_samples,
            num_burnin_steps=num_burnin_steps,
            current_state=flat_initial_state,
            kernel=adaptive_hmc,
            trace_fn=lambda _, pkr: pkr.inner_results.is_accepted
        )
        return samples, kernel_results

    print(f"Running HMC for {net.model_name}...")
    samples, is_accepted = run_chain()
    acceptance_rate = tf.reduce_mean(tf.cast(is_accepted, dtype=tf.float32))
    print(f"HMC for {net.model_name} finished. Acceptance rate: {acceptance_rate:.4f}")
    
    return samples
