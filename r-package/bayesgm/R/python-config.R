.bayesgm_state <- new.env(parent = emptyenv())
.bayesgm_state$models_module <- NULL
.bayesgm_state$datasets_module <- NULL

.nzchar_or_null <- function(x) {
  if (is.null(x)) {
    return(NULL)
  }
  if (!is.character(x) || length(x) != 1L || !nzchar(x)) {
    return(NULL)
  }
  x
}

configure_bayesgm <- function(python = Sys.getenv("RETICULATE_PYTHON", unset = ""),
                              pythonpath = Sys.getenv("BAYESGM_PYTHONPATH", unset = ""),
                              required = FALSE) {
  python <- .nzchar_or_null(python)
  pythonpath <- .nzchar_or_null(pythonpath)

  if (!is.null(python)) {
    reticulate::use_python(python, required = required)
  }

  if (!is.null(pythonpath)) {
    pythonpath <- normalizePath(pythonpath, winslash = "/", mustWork = FALSE)
    sys <- reticulate::import("sys", delay_load = FALSE, convert = FALSE)
    current_path <- reticulate::py_to_r(sys$path)
    if (!(pythonpath %in% current_path)) {
      sys$path$insert(0L, pythonpath)
    }
    .bayesgm_state$models_module <- NULL
    .bayesgm_state$datasets_module <- NULL
  }

  invisible(TRUE)
}

bayesgm_available <- function(configure = TRUE) {
  if (configure) {
    configure_bayesgm()
  }

  tryCatch(
    reticulate::py_module_available("bayesgm"),
    error = function(e) FALSE
  )
}

install_bayesgm_python <- function(envname = NULL,
                                   method = c("auto", "virtualenv", "conda"),
                                   local_path = NULL,
                                   python = NULL) {
  method <- match.arg(method)
  python <- .nzchar_or_null(python)

  if (!is.null(python)) {
    reticulate::use_python(python, required = FALSE)
  }

  package_spec <- if (is.null(local_path)) {
    "bayesgm"
  } else {
    normalizePath(local_path, winslash = "/", mustWork = TRUE)
  }

  pip_install <- method != "conda"
  reticulate::py_install(packages = package_spec, envname = envname, pip = pip_install)
  invisible(TRUE)
}

.import_bayesgm_models <- function() {
  configure_bayesgm()

  if (!bayesgm_available(configure = FALSE)) {
    stop(
      paste(
        "The Python package 'bayesgm' is not available to reticulate.",
        "Run install_bayesgm_python() or set BAYESGM_PYTHONPATH / RETICULATE_PYTHON before calling CausalBGM() or BGM()."
      ),
      call. = FALSE
    )
  }

  if (is.null(.bayesgm_state$models_module)) {
    .bayesgm_state$models_module <- reticulate::import("bayesgm.models", delay_load = FALSE)
  }

  .bayesgm_state$models_module
}

.import_bayesgm_datasets <- function() {
  configure_bayesgm()

  if (!bayesgm_available(configure = FALSE)) {
    stop(
      paste(
        "The Python package 'bayesgm' is not available to reticulate.",
        "Run install_bayesgm_python() or set BAYESGM_PYTHONPATH / RETICULATE_PYTHON before calling dataset loaders."
      ),
      call. = FALSE
    )
  }

  if (is.null(.bayesgm_state$datasets_module)) {
    .bayesgm_state$datasets_module <- reticulate::import("bayesgm.datasets", delay_load = FALSE)
  }

  .bayesgm_state$datasets_module
}

.to_numpy_matrix <- function(x, name, ncol_expected = NULL) {
  if (is.data.frame(x)) {
    x <- as.matrix(x)
  }

  if (is.null(dim(x))) {
    x <- matrix(x, ncol = 1L)
  } else {
    x <- as.matrix(x)
  }

  if (!is.numeric(x)) {
    stop(sprintf("`%s` must be numeric.", name), call. = FALSE)
  }

  if (!is.null(ncol_expected) && ncol(x) != ncol_expected) {
    stop(
      sprintf("`%s` must have %d column(s); got %d.", name, ncol_expected, ncol(x)),
      call. = FALSE
    )
  }

  reticulate::np_array(x, dtype = "float32")
}

.prepare_triplet <- function(x, y, v) {
  x_np <- .to_numpy_matrix(x, "x", ncol_expected = 1L)
  y_np <- .to_numpy_matrix(y, "y", ncol_expected = 1L)
  v_np <- .to_numpy_matrix(v, "v")

  n_rows <- c(dim(x_np)[1], dim(y_np)[1], dim(v_np)[1])
  if (length(unique(n_rows)) != 1L) {
    stop("`x`, `y`, and `v` must have the same number of rows.", call. = FALSE)
  }

  list(x = x_np, y = y_np, v = v_np)
}

.drop_nulls <- function(x) {
  x[!vapply(x, is.null, logical(1))]
}
