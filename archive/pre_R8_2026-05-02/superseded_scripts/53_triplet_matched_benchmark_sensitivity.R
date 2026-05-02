#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(arrow)
  library(dplyr)
  library(tidyr)
})

project_root <- "/umed-projekty/rtpipeline/manuscript"
analysis_dir <- file.path(project_root, "analysis")
data_dir <- file.path(analysis_dir, "data")
table_dir <- file.path(analysis_dir, "tables")

raw_candidates <- c(
  file.path(data_dir, "nsclc_io_raw_features_per_patient.parquet"),
  "/home/kgs24/rtpipeline_manuscript/analysis/data/nsclc_io_raw_features_per_patient.parquet"
)

raw_path <- raw_candidates[file.exists(raw_candidates)][1]
if (is.na(raw_path)) {
  stop(
    "Could not resolve nsclc_io_raw_features_per_patient.parquet from:\n",
    paste(raw_candidates, collapse = "\n")
  )
}

dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(table_dir, recursive = TRUE, showWarnings = FALSE)

get_family <- function(feature_name) {
  feature_name <- tolower(feature_name)
  case_when(
    grepl("_shape_", feature_name) ~ "shape",
    grepl("_firstorder_", feature_name) ~ "firstorder",
    grepl("_glcm_", feature_name) ~ "glcm",
    grepl("_glrlm_", feature_name) ~ "glrlm",
    grepl("_glszm_", feature_name) ~ "glszm",
    grepl("_gldm_", feature_name) ~ "gldm",
    grepl("_ngtdm_", feature_name) ~ "ngtdm",
    TRUE ~ "unknown"
  )
}

cv_pct <- function(x) {
  mean_x <- mean(x, na.rm = TRUE)
  sd_x <- stats::sd(x, na.rm = TRUE)
  if (is.na(mean_x) || is.na(sd_x)) {
    return(NA_real_)
  }
  if (abs(mean_x) < 1e-12) {
    return(ifelse(abs(sd_x) < 1e-12, 0, NA_real_))
  }
  100 * sd_x / abs(mean_x)
}

raw_df <- read_parquet(raw_path) %>%
  mutate(family = get_family(feature_name)) %>%
  filter(family != "unknown")

required_cols <- c("patient_id", "arm", "rater", "feature_name", "value", "family")
missing_cols <- setdiff(required_cols, names(raw_df))
if (length(missing_cols) > 0) {
  stop("Missing required columns: ", paste(missing_cols, collapse = ", "))
}

shared_patients <- raw_df %>%
  count(patient_id, arm) %>%
  pivot_wider(names_from = arm, values_from = n, values_fill = 0) %>%
  filter(ai > 0, human > 0) %>%
  pull(patient_id) %>%
  sort()

if (length(shared_patients) == 0) {
  stop("No shared AI/human patients found in raw benchmark parquet.")
}

missing_human_only <- raw_df %>%
  distinct(arm, patient_id) %>%
  tidyr::pivot_wider(names_from = arm, values_from = patient_id, values_fn = list) %>%
  summarise(
    human_patients = list(sort(unique(unlist(human)))),
    ai_patients = list(sort(unique(unlist(ai))))
  ) %>%
  mutate(human_only = list(setdiff(human_patients[[1]], ai_patients[[1]]))) %>%
  pull(human_only) %>%
  .[[1]]

raw_df <- raw_df %>%
  filter(patient_id %in% shared_patients)

ai_cov <- raw_df %>%
  filter(arm == "ai") %>%
  group_by(patient_id, feature_name, family) %>%
  summarise(
    ai_cov_pct = cv_pct(value),
    .groups = "drop"
  )

human_raters <- raw_df %>%
  filter(arm == "human") %>%
  distinct(rater) %>%
  arrange(rater) %>%
  pull(rater)

triplets <- combn(human_raters, 3, simplify = FALSE)
triplet_ids <- vapply(triplets, function(x) paste(x, collapse = "-"), character(1))

human_triplet_cov <- bind_rows(
  lapply(seq_along(triplets), function(i) {
    raters <- triplets[[i]]
    raw_df %>%
      filter(arm == "human", rater %in% raters) %>%
      group_by(patient_id, feature_name, family) %>%
      summarise(
        human_cov_pct = cv_pct(value),
        .groups = "drop"
      ) %>%
      mutate(
        human_triplet = triplet_ids[[i]],
        human_triplet_raters = paste(raters, collapse = ",")
      )
  })
)

paired_df <- human_triplet_cov %>%
  inner_join(ai_cov, by = c("patient_id", "feature_name", "family")) %>%
  mutate(
    delta_pct = human_cov_pct - ai_cov_pct,
    ai_lower = delta_pct > 0
  )

overall_by_triplet <- paired_df %>%
  group_by(human_triplet, human_triplet_raters) %>%
  summarise(
    scope = "overall",
    shared_patients = n_distinct(patient_id),
    patient_feature_pairs = n(),
    median_human_cov_pct = median(human_cov_pct, na.rm = TRUE),
    median_ai_cov_pct = median(ai_cov_pct, na.rm = TRUE),
    median_delta_pct = median(delta_pct, na.rm = TRUE),
    ai_lower_fraction_pct = 100 * mean(ai_lower, na.rm = TRUE),
    .groups = "drop"
  )

family_by_triplet <- paired_df %>%
  group_by(family, human_triplet, human_triplet_raters) %>%
  summarise(
    scope = family[1],
    shared_patients = n_distinct(patient_id),
    patient_feature_pairs = n(),
    median_human_cov_pct = median(human_cov_pct, na.rm = TRUE),
    median_ai_cov_pct = median(ai_cov_pct, na.rm = TRUE),
    median_delta_pct = median(delta_pct, na.rm = TRUE),
    ai_lower_fraction_pct = 100 * mean(ai_lower, na.rm = TRUE),
    .groups = "drop"
  )

triplet_level <- bind_rows(overall_by_triplet, family_by_triplet) %>%
  arrange(scope, human_triplet)

summary_table <- triplet_level %>%
  group_by(scope) %>%
  summarise(
    shared_patients = max(shared_patients),
    human_triplets_total = n(),
    triplets_with_positive_delta = sum(median_delta_pct > 0, na.rm = TRUE),
    median_human_cov_pct = median(median_human_cov_pct, na.rm = TRUE),
    median_ai_cov_pct = median(median_ai_cov_pct, na.rm = TRUE),
    median_triplet_delta_pct = median(median_delta_pct, na.rm = TRUE),
    min_triplet_delta_pct = min(median_delta_pct, na.rm = TRUE),
    max_triplet_delta_pct = max(median_delta_pct, na.rm = TRUE),
    median_ai_lower_fraction_pct = median(ai_lower_fraction_pct, na.rm = TRUE),
    min_ai_lower_fraction_pct = min(ai_lower_fraction_pct, na.rm = TRUE),
    max_ai_lower_fraction_pct = max(ai_lower_fraction_pct, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    missing_human_only_patients = if (length(missing_human_only) == 0) {
      ""
    } else {
      paste(missing_human_only, collapse = ",")
    },
    raw_source_path = raw_path
  ) %>%
  arrange(factor(scope, levels = c("overall", "shape", "firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm")))

write_parquet(
  paired_df,
  file.path(data_dir, "nsclc_io_triplet_matched_sensitivity.parquet")
)

write.csv(
  overall_by_triplet,
  file.path(table_dir, "nsclc_io_triplet_matched_overall.csv"),
  row.names = FALSE
)

write.csv(
  family_by_triplet,
  file.path(table_dir, "nsclc_io_triplet_matched_family.csv"),
  row.names = FALSE
)

write.csv(
  summary_table,
  file.path(table_dir, "nsclc_io_triplet_matched_summary.csv"),
  row.names = FALSE
)

message("wrote ", file.path(data_dir, "nsclc_io_triplet_matched_sensitivity.parquet"))
message("wrote ", file.path(table_dir, "nsclc_io_triplet_matched_overall.csv"))
message("wrote ", file.path(table_dir, "nsclc_io_triplet_matched_family.csv"))
message("wrote ", file.path(table_dir, "nsclc_io_triplet_matched_summary.csv"))
