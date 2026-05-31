# Machine Learning in Biostatistics
# Script 06: Cross-Validation
#
# This script performs simple 5-fold cross-validation for a diabetes
# prediction model using logistic regression.

library(tidyverse)
library(mlbench)
library(pROC)

set.seed(123)

data("PimaIndiansDiabetes")

diabetes_data <- PimaIndiansDiabetes %>%
  mutate(
    diabetes = factor(diabetes, levels = c("neg", "pos"))
  )

# Create stratified folds manually.
# This keeps the positive and negative outcome groups reasonably balanced.

k <- 5

diabetes_data <- diabetes_data %>%
  group_by(diabetes) %>%
  mutate(fold = sample(rep(1:k, length.out = n()))) %>%
  ungroup()

cv_results <- map_dfr(1:k, function(current_fold) {
  train_data <- diabetes_data %>%
    filter(fold != current_fold) %>%
    select(-fold)

  validation_data <- diabetes_data %>%
    filter(fold == current_fold) %>%
    select(-fold)

  model <- glm(
    diabetes ~ pregnant + glucose + pressure + triceps +
      insulin + mass + pedigree + age,
    data = train_data,
    family = binomial
  )

  validation_data <- validation_data %>%
    mutate(
      predicted_risk = predict(model, newdata = validation_data, type = "response")
    )

  roc_object <- roc(
    response = validation_data$diabetes,
    predictor = validation_data$predicted_risk,
    levels = c("neg", "pos"),
    quiet = TRUE
  )

  tibble(
    fold = current_fold,
    auc = as.numeric(auc(roc_object)),
    n_validation = nrow(validation_data),
    events_validation = sum(validation_data$diabetes == "pos")
  )
})

cat("\nCross-validation results:\n")
print(cv_results)

cv_summary <- cv_results %>%
  summarise(
    mean_auc = mean(auc),
    sd_auc = sd(auc),
    min_auc = min(auc),
    max_auc = max(auc)
  )

cat("\nCross-validation summary:\n")
print(cv_summary)

plot(
  cv_results$fold,
  cv_results$auc,
  type = "b",
  ylim = c(0.5, 1),
  xlab = "Fold",
  ylab = "AUC",
  main = "AUC Across 5 Cross-Validation Folds"
)

abline(h = mean(cv_results$auc), lty = 2)

cat("\nLearning points:\n")
cat("1. Cross-validation evaluates the model across multiple splits.\n")
cat("2. It gives a more stable internal estimate than one split.\n")
cat("3. It is still internal validation, not external validation.\n")
cat("4. Preprocessing and feature selection should happen inside each fold.\n")