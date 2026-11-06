# Conditional association models for papers and country-field outcomes.
run_associations <- function(project_dir = getwd(), config_path = NULL) {
  needed <- c('arrow', 'data.table', 'fixest', 'jsonlite', 'tidyselect')
  absent <- needed[!vapply(needed, requireNamespace, logical(1), quietly = TRUE)]
  if (length(absent)) stop('Run setup_R.R first. Missing packages: ', paste(absent, collapse = ', '))
  if (is.null(config_path)) {
    config_path <- Sys.getenv('GKA_CONFIG', unset = file.path(project_dir, 'config.json'))
  }
  config_path <- normalizePath(path.expand(config_path), winslash = '/', mustWork = TRUE)
  cfg <- jsonlite::fromJSON(config_path)
  out <- path.expand(cfg$output_dir)
  if (!grepl('^(/|[A-Za-z]:[/\\\\]|\\\\\\\\)', out)) out <- file.path(dirname(config_path), out)
  path <- file.path(out, 'data', 'r_papers.parquet')
  if (!file.exists(path)) stop('Run Python notebooks through 05 first: ', path)
  dir.create(file.path(out, 'r'), recursive = TRUE, showWarnings = FALSE)
  Sys.setenv(OPENBLAS_NUM_THREADS = '1', MKL_NUM_THREADS = '1', OMP_NUM_THREADS = '8')
  fixest::setFixest_nthreads(min(8L, as.integer(cfg$threads)))
  cols <- c('year', 'primary_field', 'source_code', 'team_size', 'known_countries',
            'known_fields', 'joint_refs', 'reference_coverage', 'joint_cross',
            'geo_cross', 'field_cross', 'mean_ref_age_joint', 'AIProb', 'sample_weight')
  dat <- data.table::as.data.table(arrow::read_parquet(path, col_select = tidyselect::all_of(cols)))
  n_export <- nrow(dat)
  dat <- dat[is.finite(joint_cross) & is.finite(team_size) & team_size > 0 &
               primary_field >= 0 & source_code >= 0 & joint_refs >= 2 &
               is.finite(mean_ref_age_joint) & is.finite(sample_weight) & sample_weight > 0]
  dat[, `:=`(international = as.integer(known_countries >= 2), log_team = log(team_size),
             log_refs = log(joint_refs), field_year = interaction(primary_field, year, drop = TRUE))]
  if (nrow(dat) < 100 || data.table::uniqueN(dat$source_code) < 10) {
    stop('Insufficient complete cases or source clusters for the specified paper models.')
  }
  rhs <- 'international + log_team + log_refs + mean_ref_age_joint + known_fields'
  models <- list()
  scope <- list()
  for (outcome in c('joint_cross', 'geo_cross', 'field_cross')) {
    formula <- as.formula(paste(outcome, '~', rhs, '| field_year + source_code'))
    fit <- fixest::feols(formula, data = dat, weights = ~sample_weight,
                         vcov = ~source_code, lean = TRUE, mem.clean = TRUE)
    models[[paste0('paper_', outcome)]] <- fit
    scope[[length(scope) + 1L]] <- data.frame(model = paste0('paper_', outcome),
       unit = 'one citing paper', n_export = n_export, n_complete = nrow(dat),
       n_model = stats::nobs(fit), weights = 'uniform paper sampling weight',
       fixed_effects = 'primary Fields x year + publication source',
       covariance = 'publication-source clustered',
       interpretation = 'conditional association; primary-Fields FE is a sensitivity specification')
  }
  # Optional AI profile uses only valid scores and never defines the main population.
  ai <- dat[is.finite(AIProb) & AIProb >= 0 & AIProb <= 1]
  if (nrow(ai) >= 100 && data.table::uniqueN(ai$AIProb > .5) == 2L &&
      data.table::uniqueN(ai$source_code) >= 10) {
    ai[, ai_high := as.integer(AIProb > .5)]
    fit <- fixest::feols(joint_cross ~ ai_high + international + log_team + log_refs +
                  mean_ref_age_joint + known_fields | field_year + source_code,
                  data = ai, weights = ~sample_weight, vcov = ~source_code,
                  lean = TRUE, mem.clean = TRUE)
    models[['paper_AI_profile']] <- fit
    scope[[length(scope) + 1L]] <- data.frame(model = 'paper_AI_profile', unit = 'one classified citing paper',
       n_export = n_export, n_complete = nrow(ai), n_model = stats::nobs(fit),
       weights = 'uniform paper sampling weight', fixed_effects = 'primary Fields x year + source',
       covariance = 'publication-source clustered', interpretation = 'optional AI-classified subset')
  }
  rm(ai, dat); gc()
  roles <- data.table::fread(file.path(out, 'tables', 'role_country_field_year.csv'), encoding = 'UTF-8')
  roles <- roles[is.finite(R_asm) & is.finite(RCA) & RCA >= 0 & E_asm >= cfg$min_cell_mass]
  if (nrow(roles) >= 100 && data.table::uniqueN(roles$country) >= 10 && data.table::uniqueN(roles$year) >= 10) {
    roles[, `:=`(cf = interaction(country, field, drop = TRUE), fy = interaction(field, year, drop = TRUE))]
    fit <- fixest::feols(R_asm ~ log1p(RCA) + log1p(E_asm) | cf + fy,
       data = roles, weights = ~E_asm, vcov = ~country + year, lean = TRUE, mem.clean = TRUE)
    models[['country_field_assembly']] <- fit
    scope[[length(scope) + 1L]] <- data.frame(model = 'country_field_assembly',
       unit = 'country x Fields x year cell', n_export = nrow(roles), n_complete = nrow(roles),
       n_model = stats::nobs(fit), weights = 'observed assembly exposure E_asm',
       fixed_effects = 'country x Fields + Fields x year', covariance = 'country and year two-way clustered',
       interpretation = 'aggregate conditional association with fractional country-field cells')
  }
  coef <- data.table::rbindlist(lapply(names(models), function(label) {
    fit <- models[[label]]; ct <- fixest::coeftable(fit); ci <- stats::confint(fit)
    data.table::data.table(model = label, term = rownames(ct), estimate = ct[, 1],
       std_error = ct[, 2], statistic = ct[, 3], p_value = ct[, 4],
       conf_low = ci[, 1], conf_high = ci[, 2], n = stats::nobs(fit))
  }), fill = TRUE)
  data.table::fwrite(coef, file.path(out, 'r', 'coefficients.csv'))
  data.table::fwrite(data.table::rbindlist(scope), file.path(out, 'r', 'model_scope.csv'))
  draw_coefficients <- function() {
    groups <- unique(coef$model); nr <- ceiling(length(groups) / 2)
    old <- par(mfrow = c(nr, 2), mar = c(3.2, 8.2, 2.7, 1.0), mgp = c(2.0, 0.6, 0),
               cex.axis = 0.63, cex.lab = 0.72, cex.main = 0.76)
    on.exit(par(old), add = TRUE)
    labels <- c(international = '2+ known countries', log_team = 'Log authors',
                log_refs = 'Log joint references', mean_ref_age_joint = 'Reference age',
                known_fields = 'Focal Fields count', ai_high = 'AIProb > 0.5',
                'log1p(RCA)' = 'Log(1 + RCA)', 'log1p(E_asm)' = 'Log(1 + exposure)')
    for (g in groups) {
      t <- coef[model == g]; yy <- seq_len(nrow(t)); xx <- range(c(t$conf_low, t$conf_high), finite = TRUE)
      span <- max(diff(xx), 1e-5); xx <- xx + c(-1, 1) * span * 0.08
      plot(t$estimate, yy, xlim = xx, ylim = c(0.5, nrow(t) + 0.5), yaxt = 'n',
           xlab = 'Conditional association (95% CI)', ylab = '', pch = 19, col = '#008B83',
           main = g, cex = 0.65, bty = 'l')
      segments(t$conf_low, yy, t$conf_high, yy, col = '#008B83', lwd = 1.0)
      abline(v = 0, col = '#9AA5AE', lty = 3)
      text <- labels[t$term]; text[is.na(text)] <- t$term[is.na(text)]
      axis(2, at = yy, labels = text, las = 1, tick = FALSE)
    }
  }
  height <- max(4.5, 2.5 * ceiling(length(models) / 2))
  grDevices::pdf(file.path(out, 'r', 'R_associations.pdf'), width = 180 / 25.4, height = height)
  tryCatch(draw_coefficients(), finally = grDevices::dev.off())
  grDevices::png(file.path(out, 'r', 'R_associations.png'), width = 180 / 25.4,
                height = height, units = 'in', res = 450)
  tryCatch(draw_coefficients(), finally = grDevices::dev.off())
  capture.output(fixest::etable(models), file = file.path(out, 'r', 'associations.txt'))
  latex <- fixest::etable(models, tex = TRUE)
  writeLines(as.character(latex), file.path(out, 'r', 'associations.tex'))
  saveRDS(models, file.path(out, 'r', 'associations.rds'), compress = TRUE)
  capture.output(sessionInfo(), file = file.path(out, 'r', 'R_session.txt'))
  writeLines(c('All coefficients are descriptive conditional associations.',
     'Reference lists encode observed co-use, not psychological integration or scientific quality.',
     'Primary-Fields FE and exposure-weighted aggregate models target different descriptive estimands.',
     'Clustered standard errors do not correct metadata coverage, classifier error or all network dependence.',
     'No causal, treatment, event-study or experimental interpretation is warranted.'),
     file.path(out, 'r', 'interpretation.txt'))
  print(coef)
  invisible(models)
}
