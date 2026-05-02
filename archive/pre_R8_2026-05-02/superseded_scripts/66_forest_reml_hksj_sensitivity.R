#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(metafor)
  library(tidyr)
  library(purrr)
})

input_path <- "/umed-projekty/rtpipeline/manuscript/analysis/tables/forest_cohort_heterogeneity.csv"
out_summary <- "/umed-projekty/rtpipeline/manuscript/analysis/tables/forest_heterogeneity_reml_hksj_2026-04-23.csv"
out_thorax <- "/umed-projekty/rtpipeline/manuscript/analysis/tables/forest_heterogeneity_reml_hksj_thorax_leave1out_2026-04-23.csv"

eps <- 1e-6

dat <- read_csv(input_path, show_col_types = FALSE) %>%
  mutate(
    icc_clip = pmin(pmax(median_icc, eps), 1 - eps),
    yi = qlogis(icc_clip),
    se_logit = pmax(bootstrap_se, eps) / (icc_clip * (1 - icc_clip)),
    vi = se_logit^2
  )

fit_one <- function(df) {
  res <- rma(yi = yi, vi = vi, data = df, method = "REML", test = "knha")
  pred <- predict(res)
  tibble(
    body_region = df$body_region[[1]],
    feature_family = df$feature_family[[1]],
    k_cohorts = nrow(df),
    pooled_icc_reml = plogis(res$b[[1]]),
    ci_lower_reml = plogis(res$ci.lb),
    ci_upper_reml = plogis(res$ci.ub),
    prediction_lower_reml = plogis(pred$pi.lb),
    prediction_upper_reml = plogis(pred$pi.ub),
    tau2_reml_logit = res$tau2,
    I2_reml_pct = res$I2,
    Q_reml = res$QE,
    Q_p_reml = res$QEp,
    hk_se_logit = res$se[[1]]
  )
}

leave1out_one <- function(df) {
  res <- rma(yi = yi, vi = vi, data = df, method = "REML", test = "knha")
  loo <- leave1out(res)
  tibble(
    body_region = df$body_region[[1]],
    feature_family = df$feature_family[[1]],
    omitted_cohort = df$cohort,
    pooled_icc_reml_full = plogis(res$b[[1]]),
    pooled_icc_reml_leave1out = plogis(loo$estimate),
    ci_lower_leave1out = plogis(loo$ci.lb),
    ci_upper_leave1out = plogis(loo$ci.ub),
    abs_shift = abs(plogis(loo$estimate) - plogis(res$b[[1]]))
  )
}

summary_tbl <- dat %>%
  group_by(body_region, feature_family) %>%
  group_split() %>%
  map_dfr(fit_one) %>%
  arrange(factor(body_region, levels = c("Brain", "Pelvis", "Thorax")), feature_family)

thorax_loo <- dat %>%
  filter(body_region == "Thorax") %>%
  group_by(body_region, feature_family) %>%
  group_split() %>%
  map_dfr(leave1out_one) %>%
  arrange(feature_family, desc(abs_shift))

write_csv(summary_tbl, out_summary)
write_csv(thorax_loo, out_thorax)

message("Wrote: ", out_summary)
message("Wrote: ", out_thorax)
