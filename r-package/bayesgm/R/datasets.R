.py_triplet_to_r <- function(py_triplet) {
  list(
    x = reticulate::py_to_r(py_triplet[[1]]),
    y = reticulate::py_to_r(py_triplet[[2]]),
    v = reticulate::py_to_r(py_triplet[[3]])
  )
}

load_sim_hirano_imbens <- function(N = 1000L,
                                   v_dim = 20L,
                                   seed = 0L,
                                   batch_size = 32L) {
  sampler <- .import_bayesgm_datasets()$Sim_Hirano_Imbens_sampler(
    batch_size = as.integer(batch_size),
    N = as.integer(N),
    v_dim = as.integer(v_dim),
    seed = as.integer(seed)
  )

  .py_triplet_to_r(sampler$load_all())
}

load_sim_heteroskedastic <- function(n = 20000L,
                                     z_dim = 10L,
                                     x_dim = 100L,
                                     seed = 42L,
                                     test_size = 0.1,
                                     split_seed = 123L) {
  if (x_dim < 2L) {
    stop("`x_dim` must be at least 2 so the last column can hold the response.", call. = FALSE)
  }

  datasets_module <- .import_bayesgm_datasets()
  model_selection <- tryCatch(
    reticulate::import("sklearn.model_selection", delay_load = FALSE),
    error = function(e) {
      stop(
        paste(
          "The Python module 'sklearn.model_selection' is required for load_sim_heteroskedastic().",
          "Install scikit-learn in the configured Python environment."
        ),
        call. = FALSE
      )
    }
  )

  sim_data <- datasets_module$simulate_z_hetero(
    n = as.integer(n),
    k = as.integer(z_dim),
    d = as.integer(x_dim - 1L),
    seed = as.integer(seed)
  )

  split <- model_selection$train_test_split(
    sim_data[[1]],
    sim_data[[2]],
    test_size = test_size,
    random_state = as.integer(split_seed)
  )

  X_train <- as.matrix(reticulate::py_to_r(split[[1]]))
  X_test <- as.matrix(reticulate::py_to_r(split[[2]]))
  Y_train <- matrix(as.numeric(reticulate::py_to_r(split[[3]])), ncol = 1L)
  Y_test <- matrix(as.numeric(reticulate::py_to_r(split[[4]])), ncol = 1L)

  data_train <- cbind(X_train, Y_train)
  data_test <- cbind(X_test, matrix(NA_real_, nrow = nrow(X_test), ncol = 1L))

  list(
    X_train = X_train,
    X_test = X_test,
    Y_train = Y_train,
    Y_test = Y_test,
    data_train = data_train,
    data_test = data_test
  )
}
