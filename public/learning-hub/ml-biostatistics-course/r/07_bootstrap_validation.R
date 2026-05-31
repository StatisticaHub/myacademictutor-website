# Machine Learning in Biostatistics
# Script 07: Bootstrap Validation
#
# This script estimates optimism in AUC using a simple bootstrap approach.
# It is written for teaching, so the steps are explicit.

library(tidyverse)
library(mlbench)
library(pROC)

set.seed(123)

data("PimaIndiansDiabetes")

diabetes_data <- PimaIndiansDiabetes %>%
  mutate(
    diabetes = factor(diabetes, levels = c("neg", "pos"))
  )

fit_model <- function(data) {
  glm(
    diabetes ~ pregnant + glucose + pressure + triceps +
      insulin + mass + pedigree + age,
    data = data,
    family = binomial
  )
}

calculate_auc <- function(model, data) {
  predicted_risk <- predict(model, newdata = data, type = "response")

  roc_object <- roc(
    response = data$diabetes,
    predictor = predicted_risk,
    levels = c("neg", "pos"),
    quiet = TRUE
  )

  as.numeric(auc(roc_object))
}

apparent_model <- fit_model(diabetes_data)
apparent_auc <- calculate_auc(apparent_model, diabetes_data)

cat("\nApparent AUC:\n")
print(apparent_auc)

n_boot <- 100
n <- nrow(diabetes_data)

bootstrap_results <- map_dfr(1:n_boot, function(b) {
  bootstrap_index <- sample(seq_len(n), size = n, replace = TRUE)

  bootstrap_data <- diabetes_data[bootstrap_index, ]

  bootstrap_model <- fit_model(bootstrap_data)

  auc_bootstrap <- calculate_auc(bootstrap_model, bootstrap_data)
  auc_original <- calculate_auc(bootstrap_model, diabetes_data)

  tibble(
    bootstrap_sample = b,
    auc_bootstrap = auc_bootstrap,
    auc_original = auc_original,
    optimism = auc_bootstrap - auc_original
  )
})

cat("\nFirst few bootstrap results:\n")
print(head(bootstrap_results))

optimism_estimate <- mean(bootstrap_results$optimism)
optimism_corrected_auc <- apparent_auc - optimism_estimate

summary_table <- tibble(
  apparent_auc = apparent_auc,
  mean_optimism = optimism_estimate,
  optimism_corrected_auc = optimism_corrected_auc
)

cat("\nBootstrap validation summary:\n")
print(summary_table)

plot(
  bootstrap_results$bootstrap_sample,
  bootstrap_results$optimism,
  type = "h",
  xlab = "Bootstrap sample",
  ylab = "Optimism",
  main = "Estimated Optimism Across Bootstrap Samples"
)

abline(h = optimism_estimate, lty = 2)

cat("\nLearning points:\n")
cat("1. Apparent performance is usually optimistic.\n")
cat("2. Bootstrap validation estimates optimism.\n")
cat("3. Optimism-corrected performance is more realistic.\n")
cat("4. Bootstrap validation is internal validation.\n")
cat("5. External validation is still needed before clinical use.\n")