\name{install_bayesgm_python}
\alias{install_bayesgm_python}
\title{Install the Python bayesgm Package for reticulate}
\description{
Installs the Python \code{bayesgm} package into a reticulate-managed Python
environment.
}
\usage{
install_bayesgm_python(
  envname = NULL,
  method = c("auto", "virtualenv", "conda"),
  local_path = NULL,
  python = NULL
)
}
\arguments{
  \item{envname}{
    Optional reticulate environment name.
  }
  \item{method}{
    Installation method. \code{"conda"} uses conda; all other values use pip.
  }
  \item{local_path}{
    Optional local Python package path. For this repository, point this to the
    Python package root, which is the \code{src/} directory containing
    \code{setup.py}.
  }
  \item{python}{
    Optional Python interpreter path to activate before installation.
  }
}
\value{
Invisibly returns \code{TRUE}.
}
\examples{
\dontrun{
install_bayesgm_python()

install_bayesgm_python(
  python = "/Users/ql339/anaconda3/envs/bgm/bin/python",
  local_path = "/path/to/bayesgm/src"
)
}
}
