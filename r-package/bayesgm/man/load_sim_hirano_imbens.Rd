\name{load_sim_hirano_imbens}
\alias{load_sim_hirano_imbens}
\title{Load the Python Sim_Hirano_Imbens Sampler Output}
\description{
Runs the Python \code{Sim_Hirano_Imbens_sampler(...).load_all()} pipeline
through \pkg{reticulate} and returns the resulting treatment, outcome, and
covariate arrays as R objects.
}
\usage{
load_sim_hirano_imbens(N = 1000L, v_dim = 20L, seed = 0L, batch_size = 32L)
}
\arguments{
  \item{N}{
    Sample size passed to the Python sampler.
  }
  \item{v_dim}{
    Number of covariates passed to the Python sampler.
  }
  \item{seed}{
    Random seed passed to the Python sampler.
  }
  \item{batch_size}{
    Batch size passed to the Python sampler.
  }
}
\value{
A list with components \code{x}, \code{y}, and \code{v}.
}
\examples{
\dontrun{
configure_bayesgm(
  python = "/Users/ql339/anaconda3/envs/bgm/bin/python",
  pythonpath = "/path/to/bayesgm/src"
)

sim_data <- load_sim_hirano_imbens(N = 1000, v_dim = 20)
str(sim_data)
}
}
