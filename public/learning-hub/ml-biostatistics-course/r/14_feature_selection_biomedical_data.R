# Machine Learning in Biostatistics
# Script 14: Feature Selection in Biomedical Data
#
# This script demonstrates why feature selection should be done
# inside the validation process, not before validation.

library(tidyverse)
library(mlbench)
library(pROC)

set.seed(123)

data("PimaIndiansDiabetes")

diabetes_data <- PimaIndiansDiabetes %>%
  mutate(
    diabetes = factor(diabetes, levels = c("neg", "pos"))
  )

candidate_predictors <- c(
  "pregnant",
  "glucose",
  "pressure",
  "triceps",
  "insulin",
  "mass",
  "pedigree",
  "age"
)

rank_predictors_by_auc <- function(data, predictors) {
  map_dfr(predictors, function(current_predictor) {
    roc_object <- roc(
      response = data$diabetes,
      predictor = data[[current_predictor]],
      levels = c("neg", "pos"),
      quiet = TRUE
    )

    auc_value <- as.numeric(auc(roc_object))

    tibble(
      predictor = current_predictor,
      univariable_auc = max(auc_value, 1 - auc_value)
    )
  }) %>%
    arrange(desc(univariable_auc))
}

evaluate_model <- function(train_data, test_data, selected_predictors) {
  model_formula <- as.formula(
    paste("diabetes ~", paste(selected_predictors, collapse = " + "))
  )

  model <- glm(
    model_formula,
    data = train_data,
    family = binomial
  )

  risk <- predict(
    model,
    newdata = test_data,
    type = "response"
  )

  roc_object <- roc(
    response = test_data$diabetes,
    predictor = risk,
    levels = c("neg", "pos"),
    quiet = TRUE
  )

  as.numeric(auc(roc_object))
}

train_index <- sample(seq_len(nrow(diabetes_data)), size = round(0.80 * nrow(diabetes_data)))

train_data <- diabetes_data[train_index, ]
test_data <- diabetes_data[-train_index, ]

full_data_ranking <- rank_predictors_by_auc(
  data = diabetes_data,
  predictors = candidate_predictors
)

leakage_selected_predictors <- full_data_ranking %>%
  slice_head(n = 4) %>%
  pull(predictor)

train_data_ranking <- rank_predictors_by_auc(
  data = train_data,
  predictors = candidate_predictors
)

honest_selected_predictors <- train_data_ranking %>%
  slice_head(n = 4) %>%
  pull(predictor)

leakage_auc <- evaluate_model(
  train_data = train_data,
  test_data = test_data,
  selected_predictors = leakage_selected_predictors
)

honest_auc <- evaluate_model(
  train_data = train_data,
  test_data = test_data,
  selected_predictors = honest_selected_predictors
)

feature_selection_comparison <- tibble(
  workflow = c(
    "Leakage-prone: features selected using full data",
    "Honest: features selected using training data only"
  ),
  selected_predictors = c(
    paste(leakage_selected_predictors, collapse = ", "),
    paste(honest_selected_predictors, collapse = ", ")
  ),
  test_auc = c(
    leakage_auc,
    honest_auc
  )
)

cat("\nFeature ranking using the full dataset:\n")
print(full_data_ranking)

cat("\nFeature ranking using training data only:\n")
print(train_data_ranking)

cat("\nComparison of feature selection workflows:\n")
print(feature_selection_comparison)

cat("\nLearning points:\n")
cat("1. Feature selection is part of model development.\n")
cat("2. Feature selection should use training data only.\n")
cat("3. Selecting features before validation can cause leakage.\n")
cat("4. Leakage can make model performance look better than it really is.\n")
cat("5. Selected features are prediction features, not automatically causal factors.\n")