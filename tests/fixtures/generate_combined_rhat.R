# Run from the repository root with R 4.5.1, posterior 1.7.0, jsonlite 2.0.0.
# Rscript tests/fixtures/generate_combined_rhat.R
# Prints independent reference values; does not overwrite retained evidence.
stopifnot(getRversion() == "4.5.1")
stopifnot(packageVersion("posterior") == "1.7.0")
stopifnot(packageVersion("jsonlite") == "2.0.0")
stopifnot(packageVersion("matrixStats") == "1.5.0")
fixture <- jsonlite::fromJSON("tests/fixtures/combined_rhat.json", simplifyVector = FALSE)
for (case in fixture$cases) {
  x <- do.call(cbind, lapply(case$chains, unlist))
  location <- posterior:::.rhat(posterior:::z_scale(posterior:::.split_chains(x)))
  folded <- posterior:::.rhat(posterior:::z_scale(
    posterior:::.split_chains(posterior:::fold_draws(x))))
  combined <- posterior::rhat(x)
  cat(case$name, sprintf("location=%.17g folded=%.17g combined=%.17g\n",
                        location, folded, combined))
}
print(posterior:::.split_chains)
sessionInfo()
