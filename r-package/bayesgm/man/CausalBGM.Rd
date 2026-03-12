\name{CausalBGM}
\alias{CausalBGM}
\title{Create a CausalBGM Model Wrapper}
\description{
Creates an R wrapper around the Python \code{bayesgm.models.CausalBGM} class.
The returned object is an R6 object with \code{$fit()} and \code{$predict()}
methods.
}
\usage{
CausalBGM(..., params = NULL, random_seed = NULL)
}
\arguments{
  \item{...}{
    Named model configuration parameters passed through to the Python
    \code{CausalBGM} constructor. Typical examples include
    \code{dataset}, \code{output_dir}, \code{binary_treatment},
    \code{z_dims}, and the network unit vectors.
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
object until \code{$fit()} is called, because the covariate dimension
\code{v_dim} is inferred from the supplied \code{v} matrix.

In R, object methods are called with \code{$}, not Python-style dots. The
intended workflow is:

\preformatted{
model <- CausalBGM(binary_treatment = TRUE, output_dir = tempdir())
model <- model$fit(x = x, y = y, v = v, epochs = 0, use_egm_init = FALSE)
res <- model$predict(x = x, y = y, v = v, n_mcmc = 5, burn_in = 10)
}
}
\value{
An R6 object with methods:
\itemize{
  \item \code{$fit(x, y, v, ...)} to train the model.
  \item \code{$predict(x, y, v, ...)} to generate causal effect estimates.
  \item \code{$get_params()} to inspect the stored parameter list.
  \item \code{$get_py_model()} to access the underlying Python object.
}
}
\examples{
\dontrun{
library(bayesgm)

model <- CausalBGM(
  binary_treatment = TRUE,
  dataset = "Demo",
  output_dir = tempdir(),
  use_bnn = FALSE,
  z_dims = c(1L, 1L, 1L, 1L)
)

model <- model$fit(
  x = matrix(rbinom(64, 1, 0.5), ncol = 1),
  y = matrix(rnorm(64), ncol = 1),
  v = matrix(rnorm(64 * 4), ncol = 4),
  epochs = 0,
  use_egm_init = FALSE
)

res <- model$predict(
  x = matrix(rbinom(64, 1, 0.5), ncol = 1),
  y = matrix(rnorm(64), ncol = 1),
  v = matrix(rnorm(64 * 4), ncol = 4),
  n_mcmc = 5,
  burn_in = 10
)
}
}
