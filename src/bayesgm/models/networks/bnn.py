import tensorflow as tf
import tensorflow_probability as tfp

class BayesianFullyConnectedNet(tf.keras.Model):
    """Bayesian version of BaseFullyConnectedNet.
    """
    def __init__(self, input_dim, output_dim, model_name, nb_units=[256, 256, 256]):
        super(BayesianFullyConnectedNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_name = model_name
        self.nb_units = nb_units
        self.all_layers = []
        
        self.norm_layer = tf.keras.layers.BatchNormalization()

        kernel_prior_fn = lambda dtype, shape, name, trainable, add_variable_fn: tfp.distributions.Independent(
                    tfp.distributions.Normal(loc=tf.zeros(shape, dtype=dtype), scale=0.1),
                    reinterpreted_batch_ndims=len(shape)
                )

        # Define Bayesian layers for each fully connected layer
        for i in range(len(nb_units) + 1):
            units = self.output_dim if i == len(nb_units) else self.nb_units[i]
            bayesian_layer = tfp.layers.DenseFlipout(
                units=units,
                activation=None,
                kernel_prior_fn=kernel_prior_fn
            )
            self.all_layers.append(bayesian_layer)
            
    def call(self, inputs, training=True):
        """ Return the output of the Bayesian network. """
        x = self.norm_layer(inputs)
        for i, bayesian_layer in enumerate(self.all_layers[:-1]):
            with tf.name_scope("%s_layer_%d" % (self.model_name, i+1)):
                x = bayesian_layer(x)
                x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        
        # Final layer without activation
        bayesian_layer = self.all_layers[-1]
        with tf.name_scope("%s_layer_output" % self.model_name):
            output = bayesian_layer(x)
        #kl_divergence = sum(self.losses)
        return output#, kl_divergence
    
class BayesianVariationalNet(tf.keras.Model):
    """Bayesian version of BaseVariationalNet.
    """
    def __init__(self, input_dim, output_dim, model_name, nb_units=[256, 256, 256]):
        super(BayesianVariationalNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_name = model_name
        self.nb_units = nb_units
        self.all_layers = []
        
        self.norm_layer = tf.keras.layers.BatchNormalization()

        kernel_prior_fn = lambda dtype, shape, name, trainable, add_variable_fn: tfp.distributions.Independent(
                    tfp.distributions.Normal(loc=tf.zeros(shape, dtype=dtype), scale=0.1),
                    reinterpreted_batch_ndims=len(shape)
                )

        # Define Bayesian layers for each fully connected layer
        for i in range(len(nb_units)):
            #units = self.output_dim if i == len(nb_units) else self.nb_units[i]
            bayesian_layer = tfp.layers.DenseFlipout(
                units=self.nb_units[i],
                activation=None,
                kernel_prior_fn=kernel_prior_fn,
                bias_prior_fn=kernel_prior_fn
            )
            self.all_layers.append(bayesian_layer)
        self.mean_layer = tfp.layers.DenseFlipout(
                units=self.output_dim,
                activation=None,
                kernel_prior_fn=kernel_prior_fn,
                bias_prior_fn=kernel_prior_fn
            )
        self.var_layer = tfp.layers.DenseFlipout(
                units=self.output_dim,
                #units=1,
                activation=None,
                kernel_prior_fn=kernel_prior_fn,
                bias_prior_fn=kernel_prior_fn
            )
            
    def call(self, inputs, eps=1e-6, training=True):
        """ Return the output of the Bayesian network. """
        x = self.norm_layer(inputs, training=training)
        for i, bayesian_layer in enumerate(self.all_layers):
            with tf.name_scope("%s_layer_%d" % (self.model_name, i + 1)):
                x = bayesian_layer(x, training=training)
                x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
                
        # Final layer without activation
        with tf.name_scope("%s_layer_output" % self.model_name):
            mean = self.mean_layer(x, training=training)
            var = self.var_layer(x, training=training)
            var = tf.nn.softplus(var) + eps
        return mean, var
    
    def reparameterize(self, mean, var):
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.sqrt(var) + mean

class BayesianVariationalLowRankNet(tf.keras.Model):
    """Bayesian version of BaseVariationalLowRankNet.
    
    Uses DenseFlipout layers with a low-rank-plus-diagonal covariance
    parameterization: Σ(z) = diag(var) + U Uᵀ, where U has a given rank.
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
        super(BayesianVariationalLowRankNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model_name = model_name
        self.nb_units = nb_units
        self.rank = rank 

        self.all_layers = []
        self.norm_layer = tf.keras.layers.BatchNormalization()

        def custom_prior_fn(dtype, shape, name, trainable, add_variable_fn):
            """
            Custom prior function that returns an Independent Normal distribution
            with mean 0 and a small standard deviation (0.1).

            Args:
                dtype: The data type for the distribution parameters.
                shape: The shape of the weight tensor.
                name: A name for the variable (unused here).
                trainable: Whether the variables are trainable (unused here).
                add_variable_fn: A function for adding variables (unused here).

            Returns:
                A tfp.distributions.Independent distribution representing the prior.
            """
            # Create a Normal distribution with mean 0 and scale 0.1
            prior_dist = tfp.distributions.Normal(
                loc=tf.zeros(shape, dtype=dtype),
                scale=tf.ones(shape, dtype=dtype)
            )
            # Wrap it as an Independent distribution, with the appropriate number of reinterpreted dimensions
            return tfp.distributions.Independent(prior_dist, reinterpreted_batch_ndims=len(shape))
        
        kernel_prior_fn = lambda dtype, shape, name, trainable, add_variable_fn: tfp.distributions.Independent(
            tfp.distributions.Normal(loc=tf.zeros(shape, dtype=dtype), scale=0.1),
            reinterpreted_batch_ndims=len(shape)
        )

        # Define Bayesian layers for each fully connected layer
        for i in range(len(nb_units)):
            bayesian_layer = tfp.layers.DenseFlipout(
                units=self.nb_units[i],
                kernel_prior_fn=kernel_prior_fn,
                activation=None
            )
            self.all_layers.append(bayesian_layer)

        # Output layers
        self.mean_layer = tfp.layers.DenseFlipout(
            units=self.output_dim,
            kernel_prior_fn=kernel_prior_fn,
            activation=None
        )

        # Variance layer: Outputs per-dimension variance
        self.var_layer = tfp.layers.DenseFlipout(
            units=self.output_dim,  # Per-dimension variance
            kernel_prior_fn=kernel_prior_fn,
            activation=None
        )

        # Low-rank factor layer: Outputs (batch, output_dim, rank)
        self.low_rank_layer = tfp.layers.DenseFlipout(
            units=self.output_dim * self.rank,
            kernel_prior_fn=kernel_prior_fn,
            activation=None
        )

    def call(self, inputs, training=True):
        """ Return the output of the Bayesian network. """
        x = self.norm_layer(inputs)
        for i, bayesian_layer in enumerate(self.all_layers):
            with tf.name_scope(f"{self.model_name}_layer_{i+1}"):
                x = bayesian_layer(x)
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
        eps2 = tf.random.normal(shape=(batch_size, self.rank)) # (batch, rank)

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

        # Compute U^T D^-1: Each column of U is divided by sqrt(var_diag) (broadcasting)
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

