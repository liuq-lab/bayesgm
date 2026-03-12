\name{bayesgm_available}
\alias{bayesgm_available}
\title{Check Whether the Python bayesgm Module is Available}
\description{
Checks whether the Python \code{bayesgm} module can be imported by the current
reticulate session.
}
\usage{
bayesgm_available(configure = TRUE)
}
\arguments{
  \item{configure}{
    If \code{TRUE}, call \code{\link{configure_bayesgm}()} first.
  }
}
\value{
Logical scalar indicating whether the Python module is importable.
}
\examples{
bayesgm_available()
}
