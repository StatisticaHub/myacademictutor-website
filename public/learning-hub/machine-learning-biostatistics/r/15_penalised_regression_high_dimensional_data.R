# Machine Learning in Biostatistics
# Script 15: Penalised Regression for High-Dimensional Data
#
# This script simulates a high-dimensional biomedical prediction problem
# and compares ridge, lasso, and elastic net.

library(tidyverse)
library(pROC)
library(glmnet)

set.seed(123)

n_patients <- 250
n_features <- 600
n_signal_features <- 10

x <- matrix(
  rnorm(n_patients * n_features),
  nrow = n_patients,
  ncol = n_features
)

colnames(x) <- paste0("feature_", seq_len(n_features))

true_coefficients <- c(
  rep(0.8, n_signal_features),
  rep(0, n_features - n_signal_features)
)

linear_predictor <- x %*% true_coefficients

probability <- 1 / (1 + exp(-linear_predictor))

outcome <- rbinom(
  n = n_patients,
  size = 1,
  prob = probability
)

train_index <- sample(seq_len(n_patients), size = round(0.80 * n_patients))

x_train <- x[train_index, ]
x_test <- x[-train_index, ]

y_train <- outcome[train_index]
y_test <- outcome[-train_index]

ridge_model <- cv.glmnet(
  x = x_train,
  y = y_train,
  family = "binomial",
  alpha = 0
)

lasso_model <- cv.glmnet(
  x = x_train,
  y = y_train,
  family = "binomial",
  alpha = 1
)

elastic_net_model <- cv.glmnet(
  x = x_train,
  y = y_train,
  family = "binomial",
  alpha = 0.5
)

ridge_risk <- as.numeric(
  predict(
    ridge_model,
    newx = x_test,
    s = "lambda.min",
    type = "response"
  )
)

lasso_risk <- as.numeric(
  predict(
    lasso_model,
    newx = x_test,
    s = "lambda.min",
    type = "response"
  )
)

elastic_net_risk <- as.numeric(
  predict(
    elastic_net_model,
    newx = x_test,
    s = "lambda.min",
    type = "response"
  )
)

calculate_auc <- function(actual, risk) {
  roc_object <- roc(
    response = actual,
    predictor = risk,
    quiet = TRUE
  )

  as.numeric(auc(roc_object))
}

count_selected_features <- function(model) {
  coefficients <- coef(model, s = "lambda.min")

  coefficient_table <- tibble(
    term = rownames(coefficients),
    coefficient = as.numeric(coefficients)
  )

  coefficient_table %>%
    filter(term != "(Intercept)", coefficient != 0) %>%
    nrow()
}

model_comparison <- tibble(
  model = c(
    "Ridge regression",
    "Lasso regression",
    "Elastic net"
  ),
  alpha = c(0, 1, 0.5),
  lambda_min = c(
    ridge_model$lambda.min,
    lasso_model$lambda.min,
    elastic_net_model$lambda.min
  ),
  test_auc = c(
    calculate_auc(y_test, ridge_risk),
    calculate_auc(y_test, lasso_risk),
    calculate_auc(y_test, elastic_net_risk)
  ),
  selected_features = c(
    count_selected_features(ridge_model),
    count_selected_features(lasso_model),
    count_selected_features(elastic_net_model)
  )
)

cat("\nHigh-dimensional dataset summary:\n")
cat("Number of patients:", n_patients, "\n")
cat("Number of predictors:", n_features, "\n")
cat("Number of true signal predictors:", n_signal_features, "\n")
cat("Outcome prevalence:", round(mean(outcome), 3), "\n")

cat("\nModel comparison:\n")
print(model_comparison)

lasso_coefficients <- coef(lasso_model, s = "lambda.min")

lasso_selected <- tibble(
  term = rownames(lasso_coefficients),
  coefficient = as.numeric(lasso_coefficients)
) %>%
  filter(term != "(Intercept)", coefficient != 0)

cat("\nFeatures selected by lasso:\n")
print(lasso_selected)

elastic_net_coefficients <- coef(elastic_net_model, s = "lambda.min")

elastic_net_selected <- tibble(
  term = rownames(elastic_net_coefficients),
  coefficient = as.numeric(elastic_net_coefficients)
) %>%
  filter(term != "(Intercept)", coefficient != 0)

cat("\nFeatures selected by elastic net:\n")
print(elastic_net_selected)

plot(lasso_model, main = "High-Dimensional Lasso Cross-Validation")
plot(elastic_net_model, main = "High-Dimensional Elastic Net Cross-Validation")

cat("\nLearning points:\n")
cat("1. High-dimensional data can have more predictors than patients.\n")
cat("2. Ordinary regression is often unstable in high-dimensional settings.\n")
cat("3. Ridge shrinks many coefficients but usually keeps predictors.\n")
cat("4. Lasso can select a smaller set of features.\n")
cat("5. Elastic net combines ridge and lasso behaviour.\n")
cat("6. High-dimensional models need careful validation and leakage prevention.\n")
cat("7. Selected features are not automatically causal or biologically confirmed.\n")