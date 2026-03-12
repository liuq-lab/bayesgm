.default_bgm_params <- function() {
  list(
    dataset = "MyData",
    output_dir = ".",
    save_res = TRUE,
    save_model = TRUE,
    use_bnn = FALSE,
    rank = 2L,
    z_dim = 10L,
    lr_theta = 0.005,
    lr_z = 0.005,
    g_units = c(64L, 64L, 64L, 64L, 64L),
    kl_weight = 0.00005,
    lr = 0.001,
    g_d_freq = 1L,
    use_z_rec = TRUE,
    alpha = 0.0,
    gamma = 0.0,
    e_units = c(64L, 64L, 64L, 64L, 64L),
    dz_units = c(64L, 32L, 8L),
    dx_units = c(64L, 32L, 8L)
  )
}

BayesgmBGM <- R6Class(
  "BayesgmBGM",
  public = list(
    params = NULL,
    random_seed = NULL,
    py_model = NULL,

    initialize = function(params = NULL, random_seed = NULL) {
      defaults <- .default_bgm_params()
      params <- .drop_nulls(params %||% list())

      self$params <- utils::modifyList(defaults, params)
      self$random_seed <- random_seed
      self$py_model <- NULL
    },

    fit = function(data,
                   batch_size = 32L,
                   epochs = 100L,
                   epochs_per_eval = 5L,
                   use_egm_init = TRUE,
                   egm_n_iter = 20000L,
                   egm_batches_per_eval = 500L,
                   verbose = 1L) {
      data_np <- .to_numpy_matrix(data, "data")

      if (is.null(self$py_model)) {
        params <- self$params
        params$x_dim <- as.integer(dim(data_np)[2])
        py_class <- .import_bayesgm_models()$BGM
        self$py_model <- py_class(params = params, random_seed = self$random_seed)
      }

      self$py_model$fit(
        data = data_np,
        batch_size = as.integer(batch_size),
        epochs = as.integer(epochs),
        epochs_per_eval = as.integer(epochs_per_eval),
        use_egm_init = use_egm_init,
        egm_n_iter = as.integer(egm_n_iter),
        egm_batches_per_eval = as.integer(egm_batches_per_eval),
        verbose = as.integer(verbose)
      )

      invisible(self)
    },

    predict = function(data,
                       alpha = 0.05,
                       return_samples = FALSE,
                       bs = 100L,
                       n_mcmc = 5000L,
                       burn_in = 5000L,
                       step_size = 0.01,
                       num_leapfrog_steps = 10L,
                       seed = 42L) {
      if (is.null(self$py_model)) {
        stop("The model is not fitted yet. Call `$fit()` before `$predict()`.", call. = FALSE)
      }

      data_np <- .to_numpy_matrix(data, "data")

      res <- self$py_model$predict(
        data = data_np,
        alpha = alpha,
        return_samples = return_samples,
        bs = as.integer(bs),
        n_mcmc = as.integer(n_mcmc),
        burn_in = as.integer(burn_in),
        step_size = step_size,
        num_leapfrog_steps = as.integer(num_leapfrog_steps),
        seed = as.integer(seed)
      )

      list(
        data = reticulate::py_to_r(res[[1]]),
        interval = reticulate::py_to_r(res[[2]])
      )
    },

    get_params = function() {
      self$params
    },

    get_py_model = function() {
      self$py_model
    }
  )
)

BGM <- function(..., params = NULL, random_seed = NULL) {
  extra_params <- list(...)
  if (!is.null(params) && length(extra_params) > 0L) {
    stop("Use either named arguments or `params`, not both.", call. = FALSE)
  }

  if (is.null(params)) {
    params <- extra_params
  }

  BayesgmBGM$new(params = params, random_seed = random_seed)
}
