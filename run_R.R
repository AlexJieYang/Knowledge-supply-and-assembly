args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 0L && (length(args) != 2L || args[1] != '--config')) {
  stop('Usage: Rscript run_R.R [--config config.local.json]')
}
file_arg <- grep('^--file=', commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(file_arg) != 1L) stop('Start this entry point with Rscript.')
project <- dirname(normalizePath(sub('^--file=', '', file_arg), winslash = '/', mustWork = TRUE))
source(file.path(project, 'R', 'analysis.R'), encoding = 'UTF-8')
run_associations(project, config_path = if (length(args)) args[2] else NULL)
