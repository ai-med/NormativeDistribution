#!/usr/bin/env Rscript
# Fit and apply skew-t (ST1) GAMLSS models for regional lung PCA scores.
#
# Each retained PCA coordinate has a conditional ST1 distribution. Covariates
# model its location (mu) and scale (sigma); skewness (nu) and tail parameter
# (tau) are intercept-only. Prediction returns distribution parameters, PIT
# probabilities, standard-normal residuals, and conditional component medians.
# Medians are returned in the original PCA units after reversing score scaling.
#
# Modes invoked by run.py:
#   fit    - fit full-data marginals and generate out-of-fold PIT residuals;
#   score  - apply compact saved models to supplied scores and covariates.
# The export mode converts full fitted models to parameter-only RDS lists.
# These lists retain coefficients and encoding metadata, with no training rows,
# residual arrays, fitted values, calls, or attached training environments.
#
# Inference requires gamlss.dist; fitting additionally requires gamlss.
# See README.md for input formats and examples using the Python entry point.

suppressPackageStartupMessages(library(gamlss.dist))


# Retain only the parameters and metadata required for prediction.
compact_model <- function(fit, region, component, scaling = 1) {
  if (!isTRUE(fit$converged))
    stop('Cannot export a nonconverged fit')

  links <- vapply(
    c('mu', 'sigma', 'nu', 'tau'),
    function(p) fit[[paste0(p, '.link')]],
    ''
  )

  if (!identical(unname(links), c('identity', 'log', 'identity', 'log')))
    stop('Unsupported ST1 links')

  coefficients <- lapply(
    c('mu', 'sigma', 'nu', 'tau'),
    function(p) fit[[paste0(p, '.coefficients')]]
  )

  if (any(!is.finite(unlist(coefficients))) ||
      length(coefficients[[3]]) != 1L || length(coefficients[[4]]) != 1L)
    stop('Unsupported or nonfinite coefficients')

  if (!is.finite(scaling) || scaling <= 0)
    stop('Invalid score scaling')

  list(
    region = as.character(region),
    component = as.integer(component),
    scaling = as.numeric(scaling),
    mu_formula = paste(deparse(fit$mu.formula), collapse = ' '),
    sigma_formula = paste(deparse(fit$sigma.formula), collapse = ' '),
    levels = fit$mu.xlevels,
    contrasts = attr(fit$mu.x, 'contrasts'),
    coefficients = coefficients,
    links = unname(links)
  )
}


# Recreate design matrices from the saved factor levels and formula strings.
predict_one <- function(model, dat, include_median = TRUE) {
  d <- dat

  for (v in names(model$levels)) {
    if (anyNA(d[[v]]) || any(!d[[v]] %in% model$levels[[v]]))
      stop('Missing or unseen factor level: ', v)

    d[[v]] <- factor(d[[v]], levels = model$levels[[v]])
  }

  mu_terms <- delete.response(
    terms(as.formula(model$mu_formula, env = baseenv()))
  )
  mm <- model.matrix(mu_terms, data = d, contrasts.arg = model$contrasts)

  sigma_terms <- terms(as.formula(model$sigma_formula, env = baseenv()))
  sigma_factors <- intersect(names(model$contrasts), all.vars(sigma_terms))
  smm <- model.matrix(
    sigma_terms,
    data = d,
    contrasts.arg = if (length(sigma_factors))
      model$contrasts[sigma_factors] else NULL
  )

  coef <- model$coefficients

  if (!all(names(coef[[1]]) %in% colnames(mm)) ||
      !all(names(coef[[2]]) %in% colnames(smm)))
    stop('Incompatible design matrix')

  mu <- as.numeric(mm[, names(coef[[1]]), drop = FALSE] %*% coef[[1]])
  sigma <- exp(as.numeric(smm[, names(coef[[2]]), drop = FALSE] %*% coef[[2]]))
  nu <- rep(unname(coef[[3]][[1]]), nrow(d))
  tau <- rep(exp(unname(coef[[4]][[1]])), nrow(d))

  # Convert observed scores to PIT probabilities and standard-normal residuals.
  raw_p <- vapply(
    seq_len(nrow(d)),
    function(i) pST1(
      d$observed_score[i] / model$scaling,
      mu[i], sigma[i], nu[i], tau[i]
    ),
    numeric(1)
  )

  if (any(!is.finite(raw_p)))
    stop('Nonfinite ST1 CDF')

  pit <- pmin(pmax(raw_p, 1e-12), 1 - 1e-12)

  # Cross-fitting needs only PIT residuals; avoid unused inverse-CDF solves.
  median <- if (include_median) qST1(
    rep(.5, nrow(d)), mu, sigma, nu, tau
  ) * model$scaling else rep(NA_real_, nrow(d))

  data.frame(
    mu = mu, sigma = sigma, nu = nu, tau = tau,
    pit = pit, z = qnorm(pit), median = median
  )
}


# Apply each marginal model to its corresponding lobe and PCA component.
predict_models <- function(models, data) {
  parts <- lapply(models, function(m) {
    d <- data[
      data$region == m$region & data$component == m$component,
      , drop = FALSE
    ]

    if (!nrow(d))
      stop('Missing model coordinate in scoring input')

    cbind(
      d[c('id', 'region', 'component', 'observed_score')],
      predict_one(m, d)
    )
  })

  do.call(rbind, parts)
}


# Retry unconverged fits with a larger iteration budget, then fail explicitly.
fit_one <- function(data, mu_formula, sigma_formula) {
  family <- ST1()
  family$rqres <- expression(rep(NA_real_, length(y)))

  attempt <- function(cycles) gamlss::gamlss(
    as.formula(mu_formula),
    sigma.formula = as.formula(sigma_formula),
    nu.formula = ~1,
    tau.formula = ~1,
    family = family,
    data = data,
    control = gamlss::gamlss.control(n.cyc = cycles, trace = FALSE)
  )

  fit <- tryCatch(attempt(500), error = function(e) NULL)

  if (is.null(fit) || !isTRUE(fit$converged))
    fit <- attempt(1000)

  if (!isTRUE(fit$converged))
    stop('ST1 fit did not converge after retry')

  fit
}


# Fit each marginal on all training rows and on the supplied training folds.
fit_models <- function(data, output) {
  suppressPackageStartupMessages(library(gamlss))

  factors <- intersect(c('sex', 'manufacturer', 'site'), names(data))
  for (v in factors) data[[v]] <- factor(data[[v]])

  varying <- factors[vapply(data[factors], nlevels, integer(1)) > 1L]
  mu_formula <- paste('score ~', paste(c('age', 'KVP', varying), collapse = ' + '))
  sigma_formula <- if ('site' %in% varying) '~site' else '~1'

  keys <- unique(data[c('region', 'component')])
  models <- list()
  residuals <- list()

  for (i in seq_len(nrow(keys))) {
    d <- data[
      data$region == keys$region[i] & data$component == keys$component[i],
      , drop = FALSE
    ]

    scaling <- sd(d$observed_score)
    if (!is.finite(scaling) || scaling <= 0)
      stop('Degenerate PCA score')

    d$score <- d$observed_score / scaling
    fit <- fit_one(d, mu_formula, sigma_formula)
    models[[i]] <- compact_model(fit, keys$region[i], keys$component[i], scaling)

    for (fold in sort(unique(d$fold))) {
      tr <- d[d$fold != fold, , drop = FALSE]
      te <- d[d$fold == fold, , drop = FALSE]

      for (v in varying)
        if (length(unique(tr[[v]])) != nlevels(d[[v]]))
          stop('Training fold lacks factor levels: ', v)

      fit <- fit_one(tr, mu_formula, sigma_formula)
      m <- compact_model(fit, keys$region[i], keys$component[i], scaling)
      residuals[[length(residuals) + 1L]] <- cbind(
        te[c('id', 'region', 'component')],
        predict_one(m, te, include_median = FALSE)
      )
    }

    message('Fitted ', keys$region[i], ' component ', keys$component[i])
  }

  saveRDS(models, file.path(output, 'gamlss.rds'))
  write.csv(
    do.call(rbind, residuals),
    file.path(output, 'oof_predictions.csv'),
    row.names = FALSE
  )
}


# Dispatch the requested mode and keep command-line parsing in one place.
main <- function(args) {
  mode <- args[[1]]

  arg <- function(k) {
    i <- match(k, args)
    if (is.na(i) || i == length(args)) stop('Missing ', k)
    args[[i + 1L]]
  }

  if (mode == 'export') {
    source <- arg('--source')
    output <- arg('--output')
    manifest <- read.csv(file.path(source, 'model_manifest.csv'), check.names = FALSE)

    models <- lapply(seq_len(nrow(manifest)), function(i) {
      row <- manifest[i, ]
      scaling <- if ('score_scaling_constant' %in% names(row))
        row$score_scaling_constant else 1

      fit <- readRDS(file.path(source, 'models', basename(row$model_file)))
      compact_model(fit, row$region, as.integer(sub('score_', '', row$score)), scaling)
    })

    saveRDS(models, output)
  } else if (mode == 'score') {
    data <- read.csv(
      arg('--input'), colClasses = c(id = 'character'), check.names = FALSE
    )
    predictions <- predict_models(readRDS(arg('--models')), data)
    write.csv(predictions, arg('--output'), row.names = FALSE)
  } else if (mode == 'fit') {
    data <- read.csv(
      arg('--input'), colClasses = c(id = 'character'), check.names = FALSE
    )
    fit_models(data, arg('--output'))
  } else stop('Expected export, score, or fit')
}


if (sys.nframe() == 0L) main(commandArgs(TRUE))
