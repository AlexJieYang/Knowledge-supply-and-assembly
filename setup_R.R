# Run in a fresh R >=4.3 session, with the project as working directory.
if (getRversion() < '4.3.0') stop('Use R >=4.3.')
pkgs <- c('arrow', 'data.table', 'fixest', 'jsonlite', 'tidyselect', 'IRkernel')
missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing)) {
  args <- list(pkgs = missing, repos = 'https://cloud.r-project.org')
  if (.Platform$OS.type == 'windows') args$type <- 'binary'
  do.call(install.packages, args)
}
IRkernel::installspec(name = 'global-knowledge-r', displayname = 'R - Global Knowledge Assembly', user = TRUE)
info <- system2('jupyter', c('kernelspec', 'list', '--json'), stdout = TRUE)
info <- jsonlite::fromJSON(paste(info, collapse = '\n'), simplifyVector = FALSE)
kernel_path <- file.path(info$kernelspecs[['global-knowledge-r']]$resource_dir, 'kernel.json')
spec <- jsonlite::fromJSON(kernel_path, simplifyVector = FALSE)
if (is.null(spec$env)) spec$env <- list()
for (name in c('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')) spec$env[[name]] <- '1'
spec$env[['OMP_NUM_THREADS']] <- '8'
jsonlite::write_json(spec, kernel_path, auto_unbox = TRUE, pretty = TRUE)
cat('R kernel registered. Select R - Global Knowledge Assembly for 06_R_Conditional_Associations.ipynb.\n')
