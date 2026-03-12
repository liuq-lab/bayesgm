.default_causalbgm_params <- function() {
  list(
    dataset = "MyData",
    output_dir = ".",
    save_res = TRUE,
    save_model = FALSE,
    binary_treatment = TRUE,
    use_bnn = TRUE,
    z_dims = c(3L, 3L, 6L, 6L),
    lr_theta = 0.0001,
    lr_z = 0.0001,
    g_units = c(64L, 64L, 64L, 64L, 64L),
    f_units = c(64L, 32L, 8L),
    h_units = c(64L, 32L, 8L),
    kl_weight = 0.0001,
    lr = 0.0001,
    g_d_freq = 5L,
    use_z_rec = TRUE,
    e_units = c(64L, 64L, 64L, 64L, 64L),
    dz_units = c(64L, 32L, 8L)
  )
}

BayesgmCausalBGM <- R6Class(
  "BayesgmCausalBGM",
  public = list(
    params = NULL,
    random_seed = NULL,
    py_model = NULL,

    initialize = function(params = NULL, random_seed = NULL) {
      defaults <- .default_causalbgm_params()
      params <- .drop_nulls(params %||% list())

      self$params <- utils::modifyList(defaults, params)
      self$random_seed <- random_seed
      self$py_model <- NULL
    },

    fit = function(x,
                   y,
                   v,
                   epochs = 100L,
                   epochs_per_eval = 5L,
                   batch_size = 32L,
                   startoff = 0L,
                   use_egm_init = TRUE,
                   egm_n_iter = 30000L,
                   egm_batches_per_eval = 500L,
                   save_format = "txt",
                   verbose = 1L) {
      triplet <- .prepare_triplet(x, y, v)

      if (is.null(self$py_model)) {
        params <- self$params
        params$v_dim <- as.integer(dim(triplet$v)[2])
        py_class <- .import_bayesgm_models()$CausalBGM
        self$py_model <- py_class(params = params, random_seed = self$random_seed)
      }

      self$py_model$fit(
        data = list(triplet$x, triplet$y, triplet$v),
        epochs = as.integer(epochs),
        epochs_per_eval = as.integer(epochs_per_eval),
        batch_size = as.integer(batch_size),
        startoff = as.integer(startoff),
        use_egm_init = use_egm_init,
        egm_n_iter = as.integer(egm_n_iter),
        egm_batches_per_eval = as.integer(egm_batches_per_eval),
        save_format = save_format,
        verbose = as.integer(verbose)
      )

      invisible(self)
    },

    predict = function(x,
                       y,
                       v,
                       alpha = 0.01,
                       n_mcmc = 3000L,
                       burn_in = 5000L,
                       x_values = NULL,
                       q_sd = 1.0,
                       sample_y = TRUE,
                       bs = 10000L) {
      if (is.null(self$py_model)) {
        stop("The model is not fitted yet. Call `$fit()` before `$predict()`.", call. = FALSE)
      }

      triplet <- .prepare_triplet(x, y, v)
      x_values_py <- if (is.null(x_values)) NULL else reticulate::np_array(as.numeric(x_values), dtype = "float32")

      res <- self$py_model$predict(
        data = list(triplet$x, triplet$y, triplet$v),
        alpha = alpha,
        n_mcmc = as.integer(n_mcmc),
        burn_in = as.integer(burn_in),
        x_values = x_values_py,
        q_sd = q_sd,
        sample_y = sample_y,
        bs = as.integer(bs)
      )

      list(
        effect = reticulate::py_to_r(res[[1]]),
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

`%||%` <- function(x, y) {
  if (is.null(x)) y else x
}

CausalBGM <- function(..., params = NULL, random_seed = NULL) {
  extra_params <- list(...)
  if (!is.null(params) && length(extra_params) > 0L) {
    stop("Use either named arguments or `params`, not both.", call. = FALSE)
  }

  if (is.null(params)) {
    params <- extra_params
  }

  BayesgmCausalBGM$new(params = params, random_seed = random_seed)
}
