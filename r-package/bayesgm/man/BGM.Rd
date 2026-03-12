\name{BGM}
\alias{BGM}
\title{Create a BGM Model Wrapper}
\description{
Creates an R wrapper around the Python \code{bayesgm.models.BGM} class.
The returned object is an R6 object with \code{$fit()} and \code{$predict()}
methods.
}
\usage{
BGM(..., params = NULL, random_seed = NULL)
}
\arguments{
  \item{...}{
    Named model configuration parameters passed through to the Python
    \code{BGM} constructor. Typical examples include \code{dataset},
    \code{output_dir}, \code{z_dim}, and the network unit vectors.
  }
  \item{params}{
    An optional named list of model configuration parameters. Use this instead
    of \code{...} when programmatically constructing the parameter list.
  }
  \item{random_seed}{
    Optional integer random seed forwarded to the Python constructor.
  }
}
\details{
The constructor stores model parameters and delays creation of the Python
object until \code{$fit()} is called, because the observed dimension
\code{x_dim} is inferred from the supplied data matrix.

In R, object methods are called with \code{$}, not Python-style dots. The
intended workflow is:

\preformatted{
model <- BGM(dataset = "Sim_heteroskedastic", output_dir = tempdir())
model <- model$fit(data = matrix(rnorm(64 * 6), ncol = 6), epochs = 0, use_egm_init = FALSE)
res <- model$predict(data = cbind(matrix(rnorm(8 * 5), ncol = 5), NA_real_), n_mcmc = 5, burn_in = 10)
}
}
\value{
An R6 object with methods:
\itemize{
  \item \code{$fit(data, ...)} to train the model.
  \item \code{$predict(data, ...)} to impute or sample missing values.
  \item \code{$get_params()} to inspect the stored parameter list.
  \item \code{$get_py_model()} to access the underlying Python object.
}
}
\examples{
\dontrun{
library(bayesgm)

model <- BGM(
  dataset = "DemoBGM",
  output_dir = tempdir(),
  use_bnn = FALSE,
  z_dim = 3L
)

model <- model$fit(
  data = matrix(rnorm(64 * 6), ncol = 6),
  epochs = 0,
  use_egm_init = FALSE
)

res <- model$predict(
  data = cbind(matrix(rnorm(8 * 5), ncol = 5), NA_real_),
  n_mcmc = 5,
  burn_in = 10
)
}
}
