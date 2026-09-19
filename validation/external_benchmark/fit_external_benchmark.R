#!/usr/bin/env Rscript

# Isolated Sugawara--Katahira factual-task implementation. This file does not
# import or alter the project's cognitive-model registry.

args <- commandArgs(trailingOnly = TRUE)
options(digits = 17)
arg_value <- function(name, default = NULL) {
  position <- match(name, args)
  if (is.na(position)) return(default)
  if (position == length(args)) stop(paste("missing value for", name))
  args[[position + 1]]
}

root <- normalizePath(arg_value("--root", "."), mustWork = TRUE)
output <- arg_value(
  "--output",
  file.path(root, "validation", "external_benchmark", "fit_work")
)
workers <- as.integer(arg_value("--workers", "1"))
participant_argument <- arg_value("--participants", "all")
local_library <- file.path(
  root, "validation", "external_benchmark", "raw", "r_library"
)
.libPaths(c(local_library, .libPaths()))

required <- c("Rsolnp", "numDeriv", "yaml")
missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) stop(paste("missing R packages:", paste(missing, collapse = ", ")))

config <- yaml::read_yaml(
  file.path(root, "configs", "validation", "external_benchmark_v1.yaml")
)
if (!identical(config$status, "owner_frozen")) stop("benchmark config is not frozen")
if (!identical(config$comparison$metric, "participant_laplace_log_marginal_likelihood")) {
  stop("benchmark comparison metric changed")
}
implementation <- config$implementation
if (!identical(implementation$optimizer, "Rsolnp::solnp")) {
  stop("benchmark optimizer changed")
}

data_path <- file.path(
  root, "validation", "external_benchmark", "raw",
  "figshare_10042319_v3", "factual.csv"
)
data <- read.csv(data_path, stringsAsFactors = FALSE, check.names = FALSE)
expected_columns <- c(
  "subjectid", "type", "reward", "choice", "false_start", "rt", "key"
)
if (!identical(names(data), expected_columns)) stop("factual data columns changed")
if (nrow(data) != 27456L || length(unique(data$subjectid)) != 143L) {
  stop("factual data identity changed")
}
participant_counts <- table(data$subjectid)
if (any(participant_counts != 192L)) stop("participant trial counts changed")

models <- list(
  standard_rl = list(parameters = c("alpha", "beta"), asymmetric = FALSE, perseverance = FALSE, gradual = FALSE),
  asymmetry = list(parameters = c("alpha_pos", "alpha_neg", "beta"), asymmetric = TRUE, perseverance = FALSE, gradual = FALSE),
  perseverance_impulsive = list(parameters = c("alpha", "beta", "phi"), asymmetric = FALSE, perseverance = TRUE, gradual = FALSE),
  perseverance_gradual = list(parameters = c("alpha", "beta", "tau", "phi"), asymmetric = FALSE, perseverance = TRUE, gradual = TRUE),
  hybrid_impulsive = list(parameters = c("alpha_pos", "alpha_neg", "beta", "phi"), asymmetric = TRUE, perseverance = TRUE, gradual = FALSE),
  hybrid_gradual = list(parameters = c("alpha_pos", "alpha_neg", "beta", "tau", "phi"), asymmetric = TRUE, perseverance = TRUE, gradual = TRUE)
)
if (!identical(names(models), unlist(config$models))) stop("six-model bank changed")

epsilon <- 1e-6
bounds <- function(specification) {
  lower <- upper <- setNames(numeric(length(specification$parameters)), specification$parameters)
  for (name in specification$parameters) {
    if (startsWith(name, "alpha") || name == "tau") {
      lower[[name]] <- epsilon
      upper[[name]] <- 1 - epsilon
    } else if (name == "beta") {
      lower[[name]] <- epsilon
      upper[[name]] <- Inf
    } else if (name == "phi") {
      lower[[name]] <- -10
      upper[[name]] <- 10
    }
  }
  list(lower = lower, upper = upper)
}

log_prior <- function(parameters) {
  value <- 0
  for (name in names(parameters)) {
    x <- parameters[[name]]
    if (startsWith(name, "alpha")) {
      value <- value + dbeta(x, 1.1, 1.1, log = TRUE)
    } else if (name == "beta") {
      value <- value + dgamma(x, shape = 1.2, scale = 5.0, log = TRUE)
    } else if (name == "tau") {
      value <- value + dbeta(x, 1.0, 1.0, log = TRUE)
    } else if (name == "phi") {
      value <- value + dnorm(x, mean = 0, sd = sqrt(5), log = TRUE)
    }
  }
  value
}

participant_log_likelihood <- function(parameters, trials, specification) {
  log_likelihood <- 0
  q_values <- numeric(8)
  choice_traces <- numeric(8)
  for (index in seq_len(nrow(trials))) {
    if (index == 97L) {
      q_values[] <- 0
      choice_traces[] <- 0
    }
    chosen <- as.integer(trials$choice[[index]])
    if (chosen == 0L) next
    if (chosen < 1L || chosen > 8L) return(-Inf)
    unchosen <- if (chosen %% 2L == 1L) chosen + 1L else chosen - 1L
    linear_predictor <- parameters[["beta"]] * (
      q_values[[chosen]] - q_values[[unchosen]]
    )
    if (specification$perseverance) {
      linear_predictor <- linear_predictor + parameters[["phi"]] * (
        choice_traces[[chosen]] - choice_traces[[unchosen]]
      )
    }
    log_likelihood <- log_likelihood + plogis(
      linear_predictor, log.p = TRUE
    )
    prediction_error <- as.numeric(trials$reward[[index]]) - q_values[[chosen]]
    alpha <- if (specification$asymmetric) {
      if (prediction_error >= 0) parameters[["alpha_pos"]] else parameters[["alpha_neg"]]
    } else {
      parameters[["alpha"]]
    }
    q_values[[chosen]] <- q_values[[chosen]] + alpha * prediction_error
    if (specification$perseverance) {
      tau <- if (specification$gradual) parameters[["tau"]] else 1.0
      choice_traces[[chosen]] <- choice_traces[[chosen]] + tau * (
        1 - choice_traces[[chosen]]
      )
      choice_traces[[unchosen]] <- choice_traces[[unchosen]] + tau * (
        0 - choice_traces[[unchosen]]
      )
    }
  }
  log_likelihood
}

negative_log_posterior <- function(parameters, trials, specification) {
  names(parameters) <- specification$parameters
  prior <- log_prior(parameters)
  if (!is.finite(prior)) return(Inf)
  likelihood <- participant_log_likelihood(parameters, trials, specification)
  if (!is.finite(likelihood)) return(Inf)
  -(likelihood + prior)
}

central_start <- function(specification) {
  values <- c(
    alpha = 0.5, alpha_pos = 0.5, alpha_neg = 0.5,
    beta = 1.0, tau = 0.5, phi = 0.0
  )
  values[specification$parameters]
}

random_start <- function(specification) {
  result <- setNames(numeric(length(specification$parameters)), specification$parameters)
  for (name in specification$parameters) {
    if (startsWith(name, "alpha")) {
      result[[name]] <- min(max(rbeta(1, 1.1, 1.1), epsilon), 1 - epsilon)
    } else if (name == "beta") {
      result[[name]] <- max(rgamma(1, shape = 1.2, scale = 5.0), epsilon)
    } else if (name == "tau") {
      result[[name]] <- min(max(runif(1), epsilon), 1 - epsilon)
    } else if (name == "phi") {
      result[[name]] <- min(max(rnorm(1, 0, sqrt(5)), -10), 10)
    }
  }
  result
}

fit_model <- function(subject_id, trials, model_name, specification) {
  model_index <- match(model_name, names(models))
  set.seed(as.integer(implementation$start_seed) + 100L * subject_id + model_index)
  starts <- list(central_start(specification))
  for (start_index in 2:as.integer(implementation$deterministic_multistarts)) {
    starts[[start_index]] <- random_start(specification)
  }
  limits <- bounds(specification)
  candidates <- vector("list", length(starts))
  for (start_index in seq_along(starts)) {
    fit <- tryCatch(
      suppressWarnings(
        Rsolnp::solnp(
          pars = starts[[start_index]],
          fun = negative_log_posterior,
          trials = trials,
          specification = specification,
          LB = limits$lower,
          UB = limits$upper,
          control = list(
            trace = 0,
            tol = as.numeric(implementation$convergence$tolerance),
            delta = as.numeric(implementation$convergence$delta),
            outer.iter = as.integer(implementation$convergence$outer_iterations),
            inner.iter = as.integer(implementation$convergence$inner_iterations)
          )
        )
      ),
      error = function(error) NULL
    )
    if (!is.null(fit) && is.finite(tail(fit$values, 1))) {
      candidates[[start_index]] <- list(
        fit = fit,
        objective = tail(fit$values, 1),
        start_index = start_index
      )
    }
  }
  candidates <- Filter(Negate(is.null), candidates)
  converged <- Filter(function(candidate) candidate$fit$convergence == 0, candidates)
  if (!length(converged)) {
    return(data.frame(
      subject_id = subject_id, model = model_name, status = "optimizer_failed",
      convergence = NA_integer_, start_index = NA_integer_, n_parameters = length(specification$parameters),
      log_likelihood = NA_real_, log_prior = NA_real_, log_posterior = NA_real_,
      laplace_log_marginal_likelihood = NA_real_, hessian_min_eigenvalue = NA_real_,
      hessian_error = NA_character_,
      parameters_json = NA_character_, stringsAsFactors = FALSE
    ))
  }
  objectives <- vapply(converged, function(candidate) candidate$objective, numeric(1))
  best <- converged[[which.min(objectives)]]
  parameters <- best$fit$pars
  names(parameters) <- specification$parameters
  hessian_error <- NA_character_
  hessian <- tryCatch(
    numDeriv::hessian(
      func = negative_log_posterior,
      x = parameters,
      method = "Richardson",
      trials = trials,
      specification = specification
    ),
    error = function(error) {
      hessian_error <<- conditionMessage(error)
      NULL
    }
  )
  if (is.null(hessian) || any(!is.finite(hessian))) {
    hessian_eigenvalues <- NA_real_
    status <- "hessian_failed"
    lml <- NA_real_
  } else {
    hessian <- (hessian + t(hessian)) / 2
    hessian_eigenvalues <- eigen(hessian, symmetric = TRUE, only.values = TRUE)$values
    if (any(!is.finite(hessian_eigenvalues)) || min(hessian_eigenvalues) <= 0) {
      status <- "hessian_not_positive_definite"
      lml <- NA_real_
    } else {
      status <- "ok"
      dimension <- length(parameters)
      log_posterior <- -negative_log_posterior(parameters, trials, specification)
      lml <- log_posterior + dimension / 2 * log(2 * pi) -
        0.5 * sum(log(hessian_eigenvalues))
    }
  }
  likelihood <- participant_log_likelihood(parameters, trials, specification)
  prior <- log_prior(parameters)
  data.frame(
    subject_id = subject_id,
    model = model_name,
    status = status,
    convergence = as.integer(best$fit$convergence),
    start_index = as.integer(best$start_index),
    n_parameters = length(parameters),
    log_likelihood = likelihood,
    log_prior = prior,
    log_posterior = likelihood + prior,
    laplace_log_marginal_likelihood = lml,
    hessian_min_eigenvalue = if (all(is.na(hessian_eigenvalues))) NA_real_ else min(hessian_eigenvalues),
    hessian_error = hessian_error,
    parameters_json = jsonlite::toJSON(as.list(parameters), auto_unbox = TRUE, digits = 17),
    stringsAsFactors = FALSE
  )
}

fit_participant <- function(subject_id) {
  jobs_root <- file.path(output, "jobs")
  dir.create(jobs_root, recursive = TRUE, showWarnings = FALSE)
  destination <- file.path(jobs_root, sprintf("participant_%03d.csv", subject_id))
  if (file.exists(destination)) {
    existing <- tryCatch(read.csv(destination, stringsAsFactors = FALSE), error = function(error) NULL)
    if (!is.null(existing) && nrow(existing) == length(models) && all(existing$status == "ok")) {
      return(destination)
    }
  }
  trials <- data[data$subjectid == subject_id, , drop = FALSE]
  rows <- lapply(
    names(models),
    function(model_name) fit_model(subject_id, trials, model_name, models[[model_name]])
  )
  result <- do.call(rbind, rows)
  temporary <- paste0(destination, ".tmp")
  write.csv(result, temporary, row.names = FALSE)
  if (!file.rename(temporary, destination)) stop("could not atomically write participant result")
  message(sprintf(
    "participant=%03d ok=%d/%d",
    subject_id, sum(result$status == "ok"), nrow(result)
  ))
  destination
}

if (Sys.getenv("GATE_B_SOURCE_ONLY", unset = "0") != "1") {
  participant_ids <- sort(unique(as.integer(data$subjectid)))
  if (participant_argument != "all") {
    participant_ids <- as.integer(strsplit(participant_argument, ",", fixed = TRUE)[[1]])
    if (any(!participant_ids %in% unique(data$subjectid))) stop("unknown participant requested")
  }
  dir.create(output, recursive = TRUE, showWarnings = FALSE)
  if (.Platform$OS.type == "unix" && workers > 1L) {
    invisible(parallel::mclapply(participant_ids, fit_participant, mc.cores = workers))
  } else {
    invisible(lapply(participant_ids, fit_participant))
  }
}
