\name{configure_bayesgm}
\alias{configure_bayesgm}
\title{Configure reticulate for Python bayesgm}
\description{
Configures the Python interpreter and optional Python module search path used
by the R package.
}
\usage{
configure_bayesgm(
  python = Sys.getenv("RETICULATE_PYTHON", unset = ""),
  pythonpath = Sys.getenv("BAYESGM_PYTHONPATH", unset = ""),
  required = FALSE
)
}
\arguments{
  \item{python}{
    Optional path to the Python interpreter that should be used by
    \pkg{reticulate}.
  }
  \item{pythonpath}{
    Optional path inserted into \code{sys.path} before importing Python
    modules. This is useful during development, for example when pointing to
    the repository's \code{src/} directory.
  }
  \item{required}{
    Passed to \code{reticulate::use_python()}.
  }
}
\value{
Invisibly returns \code{TRUE}.
}
\examples{
\dontrun{
configure_bayesgm(
  python = "/Users/ql339/anaconda3/envs/bgm/bin/python",
  pythonpath = "/path/to/bayesgm/src"
)
}
}
