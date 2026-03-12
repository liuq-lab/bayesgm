test_that("BGM wrapper fits and predicts with missing data", {
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
  skip_if_not(reticulate::py_module_available("sklearn"))

  sim_data <- load_sim_heteroskedastic(
    n = 128L,
    z_dim = 3L,
    x_dim = 8L,
    seed = 42L,
    test_size = 0.25,
    split_seed = 123L
  )

  model <- BGM(
    dataset = "RSimHetero",
    output_dir = tempdir(),
    save_model = FALSE,
    save_res = FALSE,
    use_bnn = FALSE,
    z_dim = 3L,
    g_units = c(8L, 8L),
    e_units = c(8L, 8L),
    dz_units = c(8L, 4L),
    dx_units = c(8L, 4L)
  )

  model <- model$fit(
    data = sim_data$data_train,
    batch_size = 32L,
    epochs = 0L,
    epochs_per_eval = 1L,
    use_egm_init = FALSE,
    egm_n_iter = 0L,
    egm_batches_per_eval = 1L,
    verbose = 0L
  )

  res <- model$predict(
    data = sim_data$data_test,
    alpha = 0.05,
    bs = 16L,
    n_mcmc = 5L,
    burn_in = 10L,
    step_size = 0.01,
    num_leapfrog_steps = 3L,
    seed = 42L
  )

  expect_equal(dim(res$data), dim(sim_data$data_test))
  expect_equal(dim(res$interval), c(nrow(sim_data$data_test), 1, 2))
  expect_false(anyNA(res$data[, ncol(res$data)]))
})
