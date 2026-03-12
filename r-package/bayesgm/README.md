# bayesgm R package

This package provides an R interface to the Python `bayesgm` toolkit through
`reticulate`. It currently exposes R wrappers for both `CausalBGM` and `BGM`.

Typical causal usage:

```r
library(bayesgm)

configure_bayesgm(
  python = "/Users/ql339/anaconda3/envs/bgm/bin/python",
  pythonpath = "/path/to/bayesgm/src"
)

model <- CausalBGM(
  binary_treatment = TRUE,
  dataset = "Demo",
  output_dir = tempdir()
)

model <- model$fit(x = x, y = y, v = v, epochs = 0, use_egm_init = FALSE)
res <- model$predict(x = x, y = y, v = v, n_mcmc = 5, burn_in = 10)
```

The R wrapper uses idiomatic R method syntax, so model methods are called with
`$fit()` and `$predict()`.

Typical generative-model usage:

```r
library(bayesgm)

configure_bayesgm(
  python = "/Users/ql339/anaconda3/envs/bgm/bin/python",
  pythonpath = "/path/to/bayesgm/src"
)

model <- BGM(
  dataset = "Sim_heteroskedastic",
  output_dir = tempdir()
)

model <- model$fit(data = matrix(rnorm(128 * 6), ncol = 6), epochs = 0, use_egm_init = FALSE)
res <- model$predict(data = cbind(matrix(rnorm(16 * 5), ncol = 5), NA_real_), n_mcmc = 5, burn_in = 10)
```
