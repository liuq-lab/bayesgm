test_that("CausalBGM wrapper fits and predicts for binary treatment", {
  python <- Sys.getenv("RETICULATE_PYTHON", unset = "")
  pythonpath <- Sys.getenv("BAYESGM_PYTHONPATH", unset = "")

  if (!nzchar(python) || !nzchar(pythonpath)) {
    skip("RETICULATE_PYTHON and BAYESGM_PYTHONPATH are required for the integration test.")
  }

  withr::local_envvar(c(
    RETICULATE_PYTHON = python,
    BAYESGM_PYTHONPATH = pythonpath
  ))

  skip_if_not(bayesgm_available())

  set.seed(123)
  n <- 64
  v <- matrix(rnorm(n * 4), ncol = 4)
  x <- matrix(rbinom(n, 1, 0.5), ncol = 1)
  y <- matrix(1.5 * x[, 1] + 0.7 * v[, 1] - 0.3 * v[, 2] + rnorm(n, sd = 0.25), ncol = 1)

  model <- CausalBGM(
    binary_treatment = TRUE,
    dataset = "RBinary",
    output_dir = tempdir(),
    save_model = FALSE,
    save_res = FALSE,
    use_bnn = FALSE,
    z_dims = c(1L, 1L, 1L, 1L),
    g_units = c(8L, 8L),
    e_units = c(8L, 8L),
    f_units = c(8L, 4L),
    h_units = c(8L, 4L),
    dz_units = c(8L, 4L)
  )

  model <- model$fit(
    x = x,
    y = y,
    v = v,
    epochs = 0,
    epochs_per_eval = 1,
    batch_size = 32,
    use_egm_init = FALSE,
    egm_n_iter = 0,
    egm_batches_per_eval = 1,
    verbose = 0
  )

  res <- model$predict(
    x = x,
    y = y,
    v = v,
    n_mcmc = 5,
    burn_in = 10,
    q_sd = 0.5
  )

  expect_length(res$effect, n)
  expect_equal(dim(res$interval), c(n, 2))
})

test_that("CausalBGM wrapper fits and predicts for continuous treatment", {
  python <- Sys.getenv("RETICULATE_PYTHON", unset = "")
  pythonpath <- Sys.getenv("BAYESGM_PYTHONPATH", unset = "")

  if (!nzchar(python) || !nzchar(pythonpath)) {
    skip("RETICULATE_PYTHON and BAYESGM_PYTHONPATH are required for the integration test.")
  }

  withr::local_envvar(c(
    RETICULATE_PYTHON = python,
    BAYESGM_PYTHONPATH = pythonpath
  ))

  skip_if_not(bayesgm_available())

  set.seed(456)
  n <- 64
  v <- matrix(rnorm(n * 4), ncol = 4)
  x <- matrix(rexp(n, rate = 1), ncol = 1)
  y <- matrix(x[, 1] + 0.5 * v[, 1] + rnorm(n, sd = 0.3), ncol = 1)

  model <- CausalBGM(
    binary_treatment = FALSE,
    dataset = "RContinuous",
    output_dir = tempdir(),
    save_model = FALSE,
    save_res = FALSE,
    use_bnn = FALSE,
    z_dims = c(1L, 1L, 1L, 1L),
    g_units = c(8L, 8L),
    e_units = c(8L, 8L),
    f_units = c(8L, 4L),
    h_units = c(8L, 4L),
    dz_units = c(8L, 4L)
  )

  model <- model$fit(
    x = x,
    y = y,
    v = v,
    epochs = 0,
    epochs_per_eval = 1,
    batch_size = 32,
    use_egm_init = FALSE,
    egm_n_iter = 0,
    egm_batches_per_eval = 1,
    verbose = 0
  )

  res <- model$predict(
    x = x,
    y = y,
    v = v,
    x_values = c(0, 1, 2),
    n_mcmc = 5,
    burn_in = 10,
    q_sd = 0.5
  )

  expect_length(res$effect, 3)
  expect_equal(dim(res$interval), c(3, 2))
})
