from .base import (
    BaseFullyConnectedNet,
    BaseVariationalNet,
    BaseVariationalLowRankNet,
    Discriminator,
    MCMCFullyConnectedNet,
    run_mcmc_for_net,
)
from .bnn import (
    BayesianFullyConnectedNet,
    BayesianVariationalNet,
    BayesianVariationalLowRankNet,
)
from .conv import MNISTEncoderConv, MNISTGenerator, MNISTDiscriminator

__all__ = [
    # Basic MLP
    "BaseFullyConnectedNet",
    # Bayesian version of BaseFullyConnectedNet
    "BayesianFullyConnectedNet",
    # Variational net (diagonal covariance)
    "BaseVariationalNet",
    # Bayesian version of BaseVariationalNet
    "BayesianVariationalNet",
    # Variational net (low-rank covariance)
    "BaseVariationalLowRankNet",
    # Bayesian version of BaseVariationalLowRankNet
    "BayesianVariationalLowRankNet",
    # Discriminator
    "Discriminator",
    # MCMC-compatible MLP (inherits BaseFullyConnectedNet)
    "MCMCFullyConnectedNet",
    "run_mcmc_for_net",
    # CNN modules for MNIST
    "MNISTEncoderConv",
    "MNISTGenerator",
    "MNISTDiscriminator",
]
