\name{load_sim_heteroskedastic}
\alias{load_sim_heteroskedastic}
\title{Load the Python Sim_heteroskedastic Workflow Data Split}
\description{
Runs the Python \code{simulate_z_hetero(...)} data-generation pipeline
through \pkg{reticulate}, applies the same
\code{sklearn.model_selection.train_test_split(...)} call used in
\code{src/main.py}, and returns the train/test matrices needed by the
\code{BGM} vignette.
}
\usage{
load_sim_heteroskedastic(
  n = 20000L,
  z_dim = 10L,
  x_dim = 100L,
  seed = 42L,
  test_size = 0.1,
  split_seed = 123L
)
}
\arguments{
  \item{n}{
    Number of samples passed to \code{simulate_z_hetero()}.
  }
  \item{z_dim}{
    Latent dimension \code{k} passed to \code{simulate_z_hetero()}.
  }
  \item{x_dim}{
    Total observed dimension for the BGM model, including the response
    column. The simulator is called with \code{d = x_dim - 1}.
  }
  \item{seed}{
    Random seed passed to \code{simulate_z_hetero()}.
  }
  \item{test_size}{
    Test-set fraction passed to \code{train_test_split()}.
  }
  \item{split_seed}{
    Random state passed to \code{train_test_split()}.
  }
}
\value{
A list with components \code{X_train}, \code{X_test}, \code{Y_train},
\code{Y_test}, \code{data_train}, and \code{data_test}. The returned
\code{data_test} matrix has the response column replaced with \code{NA},
matching the BGM prediction workflow in Python.
}
\examples{
\dontrun{
configure_bayesgm(
  python = "/Users/ql339/anaconda3/envs/bgm/bin/python",
  pythonpath = "/path/to/bayesgm/src"
)

sim_data <- load_sim_heteroskedastic(n = 2000, z_dim = 5, x_dim = 20)
str(sim_data, max.level = 1)
}
}
