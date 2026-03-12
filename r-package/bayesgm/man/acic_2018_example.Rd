\name{acic_2018_example}
\alias{acic_2018_example}
\docType{data}
\title{ACIC 2018 Binary-Treatment Example Dataset}
\description{
A bundled ACIC 2018 semi-synthetic binary-treatment example used by the
\pkg{bayesgm} vignettes. The dataset is derived from the ACIC 2018 covariate
table together with the factual and counterfactual files for
\code{ufid = "629e3d2c63914e45b227cc913c09cebe"}.
}
\usage{
acic_2018_example
}
\format{
A list with the following components:
\describe{
  \item{\code{ufid}}{Character scalar giving the ACIC configuration id.}
  \item{\code{sample_id}}{Character vector of sample identifiers.}
  \item{\code{x}}{Numeric treatment matrix with shape \code{(1000, 1)}.}
  \item{\code{y}}{Numeric outcome matrix with shape \code{(1000, 1)}.}
  \item{\code{v}}{Numeric covariate matrix with shape \code{(1000, 177)}, standardised to match the Python sampler output.}
  \item{\code{y0_true}}{Numeric vector of ground-truth potential outcomes under control.}
  \item{\code{y1_true}}{Numeric vector of ground-truth potential outcomes under treatment.}
  \item{\code{ite_true}}{Numeric vector of ground-truth individual treatment effects.}
  \item{\code{ate_true}}{Numeric scalar equal to \code{mean(ite_true)}.}
}
}
\source{
Derived from the ACIC 2018 competition data:
\code{x.csv},
\code{scaling/factuals/629e3d2c63914e45b227cc913c09cebe.csv}, and
\code{scaling/counterfactuals/629e3d2c63914e45b227cc913c09cebe_cf.csv}.
}
\examples{
data("acic_2018_example", package = "bayesgm")

str(acic_2018_example, max.level = 1)
}
