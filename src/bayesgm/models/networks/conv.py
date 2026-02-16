import tensorflow as tf
import tensorflow_probability as tfp
tfl = tf.keras.layers
tfpl = tfp.layers

class MNISTEncoderConv(tf.keras.Model):
    """Convolutional encoder used to infer latent variables from MNIST images."""

    def __init__(self, z_dim: int = 10, filters: int = 32, name: str = "mnist_encoder_conv"):
        super().__init__(name=name)

        self.net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
                tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), strides=(2, 2), padding="same", use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(0.2),
                # (B, 14, 14, 32)
                tf.keras.layers.Conv2D(filters * 2, kernel_size=(3, 3), strides=(2, 2), padding="same", use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(0.2),
                # (B, 7, 7, 64)
                tf.keras.layers.Conv2D(filters * 4, kernel_size=(3, 3), strides=(1, 1), padding="same", use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.LeakyReLU(0.2),
                # (B, 7, 7, 128)
                tf.keras.layers.Flatten(),
                # (B, 128 * 7 * 7)
                tf.keras.layers.Dense(256),
                tf.keras.layers.LeakyReLU(0.2),
                # (B, 256)
                tf.keras.layers.Dense(z_dim),
            ]
        )

    def call(self, x, training: bool = True):  # pylint: disable=arguments-differ
        return self.net(x, training=training)


class MNISTGenerator(tf.keras.Model):
    """Generator that maps latent variables to MNIST-like observations."""

    def __init__(
        self,
        z_dim: int = 10,
        filters: int = 32,
        use_bnn: bool = False,
        kl_weight: float = 1.0 / 60000,
        eps: float = 1e-6,
        name: str = "mnist_generator",
    ):
        super().__init__(name=name)
        self.eps = eps
        self.kl_weight = kl_weight

        tfl = tf.keras.layers
        tfpl = tfp.layers

        if use_bnn:
            def kl_fn():
                return lambda q, p, _: self.kl_weight * tfp.distributions.kl_divergence(q, p)

            self.fc = tf.keras.Sequential(
                [
                    tf.keras.layers.InputLayer(input_shape=(z_dim,)),
                    tfpl.DenseFlipout(7 * 7 * filters * 4),
                    tf.keras.layers.LeakyReLU(0.2),
                    tf.keras.layers.Reshape((7, 7, filters * 4)),
                ]
            )
            self.up = tf.keras.Sequential(
                [
                    tfl.UpSampling2D(size=2, interpolation="nearest"),
                    tfpl.Convolution2DFlipout(
                        filters * 2,
                        kernel_size=3,
                        padding="same",
                        activation=None,
                        use_bias=False,
                        kernel_divergence_fn=kl_fn(),
                    ),
                    tfl.BatchNormalization(),
                    tfl.LeakyReLU(0.2),
                    tfl.UpSampling2D(size=2, interpolation="nearest"),
                    tfpl.Convolution2DFlipout(
                        filters,
                        kernel_size=3,
                        padding="same",
                        activation=None,
                        use_bias=False,
                        kernel_divergence_fn=kl_fn(),
                    ),
                    tfl.BatchNormalization(),
                    tfl.LeakyReLU(0.2),
                    tfpl.Convolution2DFlipout(
                        filters,
                        kernel_size=3,
                        padding="same",
                        activation=None,
                        use_bias=False,
                        kernel_divergence_fn=kl_fn(),
                    ),
                    tfl.BatchNormalization(),
                    tfl.LeakyReLU(0.2),
                ]
            )
            self.mean_head = tfpl.Convolution2DFlipout(
                1,
                1,
                padding="same",
                activation=None,
                kernel_divergence_fn=kl_fn(),
                name="x_mean_logits",
            )
            self.var_head = tfpl.Convolution2DFlipout(
                1,
                1,
                padding="same",
                activation=None,
                kernel_divergence_fn=kl_fn(),
                name="x_var_raw",
            )
        else:
            self.fc = tf.keras.Sequential(
                [
                    tf.keras.layers.InputLayer(input_shape=(z_dim,)),
                    tf.keras.layers.Dense(7 * 7 * filters * 4),
                    tf.keras.layers.LeakyReLU(0.2),
                    tf.keras.layers.Reshape((7, 7, filters * 4)),
                ]
            )
            self.up = tf.keras.Sequential(
                [
                    tf.keras.layers.Conv2DTranspose(filters * 2, 3, strides=2, padding="same", use_bias=False),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.LeakyReLU(0.2),
                    tf.keras.layers.Conv2DTranspose(filters, 3, strides=2, padding="same", use_bias=False),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.LeakyReLU(0.2),
                    tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.LeakyReLU(0.2),
                ]
            )
            self.mean_head = tf.keras.layers.Conv2D(1, 1, padding="same", name="x_mean")
            self.var_head = tf.keras.layers.Conv2D(
                1,
                1,
                padding="same",
                name="x_var"
                #bias_initializer=tf.keras.initializers.Constant(-3.0),
            )

    def call(self, z, training: bool = True):
        h = self.fc(z, training=training)
        h = self.up(h, training=training)

        x_mean = self.mean_head(h)
        x_var_raw = self.var_head(h)
        # Ensure positive variance using softplus with numerical stability
        x_var = tf.nn.softplus(x_var_raw) + self.eps
        return x_mean, x_var

    @staticmethod
    def reparameterize(mean, var):
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.sqrt(var) + mean


class MNISTDiscriminator(tf.keras.Model):
    """Discriminator tailored for MNIST sized inputs."""

    def __init__(self, filters: int = 64, dropout: float = 0.3, name: str = "mnist_discriminator"):
        super().__init__(name=name)

        self.blocks = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(filters, 5, strides=2, padding="same", use_bias=True),
                tf.keras.layers.LeakyReLU(0.2),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Conv2D(filters * 2, 5, strides=2, padding="same", use_bias=True),
                tf.keras.layers.LeakyReLU(0.2),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Conv2D(filters * 4, 3, strides=2, padding="same", use_bias=True),
                tf.keras.layers.LeakyReLU(0.2),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128),
                tf.keras.layers.LeakyReLU(0.2),
            ]
        )
        self.logit = tf.keras.layers.Dense(1, activation=None)

    def call(self, inputs, training: bool = True):  # pylint: disable=arguments-differ
        x = inputs
        if x.shape.rank == 2 and x.shape[-1] == 28 * 28:
            x = tf.reshape(x, [-1, 28, 28, 1])
        x = self.blocks(x, training=training)
        return self.logit(x)
