# Machine Learning in Biostatistics
# Script 10: Regularisation and Feature Selection
#
# This script compares ordinary logistic regression, ridge regression,
# lasso regression, and elastic net for diabetes prediction.

library(tidyverse)
library(mlbench)
library(pROC)
library(glmnet)

set.seed(123)

data("PimaIndiansDiabetes")

diabetes_data <- PimaIndiansDiabetes %>%
  mutate(
    diabetes = factor(diabetes, levels = c("neg", "pos"))
  )

train_index <- sample(seq_len(nrow(diabetes_data)), size = round(0.80 * nrow(diabetes_data)))

train_data <- diabetes_data[train_index, ]
test_data <- diabetes_data[-train_index, ]

x_train <- model.matrix(
  diabetes ~ pregnant + glucose + pressure + triceps +
    insulin + mass + pedigree + age,
  data = train_data
)[, -1]

x_test <- model.matrix(
  diabetes ~ pregnant + glucose + pressure + triceps +
    insulin + mass + pedigree + age,
  data = test_data
)[, -1]

y_train <- ifelse(train_data$diabetes == "pos", 1, 0)
y_test_factor <- test_data$diabetes

ordinary_model <- glm(
  diabetes ~ pregnant + glucose + pressure + triceps +
    insulin + mass + pedigree + age,
  data = train_data,
  family = binomial
)

ordinary_risk <- predict(
  ordinary_model,
  newdata = test_data,
  type = "response"
)

ridge_model <- cv.glmnet(
  x = x_train,
  y = y_train,
  family = "binomial",
  alpha = 0
)

ridge_risk <- as.numeric(
  predict(
    ridge_model,
    newx = x_test,
    s = "lambda.min",
    type = "response"
  )
)

lasso_model <- cv.glmnet(
  x = x_train,
  y = y_train,
  family = "binomial",
  alpha = 1
)

lasso_risk <- as.numeric(
  predict(
    lasso_model,
    newx = x_test,
    s = "lambda.min",
    type = "response"
  )
)

elastic_net_model <- cv.glmnet(
  x = x_train,
  y = y_train,
  family = "binomial",
  alpha = 0.5
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
    levels = c("neg", "pos"),
    quiet = TRUE
  )

  as.numeric(auc(roc_object))
}

model_comparison <- tibble(
  model = c(
    "Ordinary logistic regression",
    "Ridge regression",
    "Lasso regression",
    "Elastic net"
  ),
  auc = c(
    calculate_auc(y_test_factor, ordinary_risk),
    calculate_auc(y_test_factor, ridge_risk),
    calculate_auc(y_test_factor, lasso_risk),
    calculate_auc(y_test_factor, elastic_net_risk)
  )
)

cat("\nModel comparison using test AUC:\n")
print(model_comparison)

lasso_coefficients <- coef(lasso_model, s = "lambda.min")

lasso_selected <- tibble(
  term = rownames(lasso_coefficients),
  coefficient = as.numeric(lasso_coefficients)
) %>%
  filter(coefficient != 0)

cat("\nFeatures selected by lasso:\n")
print(lasso_selected)

elastic_net_coefficients <- coef(elastic_net_model, s = "lambda.min")

elastic_net_selected <- tibble(
  term = rownames(elastic_net_coefficients),
  coefficient = as.numeric(elastic_net_coefficients)
) %>%
  filter(coefficient != 0)

cat("\nFeatures selected by elastic net:\n")
print(elastic_net_selected)

plot(ridge_model, main = "Ridge Cross-Validation")
plot(lasso_model, main = "Lasso Cross-Validation")
plot(elastic_net_model, main = "Elastic Net Cross-Validation")

cat("\nLearning points:\n")
cat("1. Ridge shrinks coefficients but usually keeps predictors.\n")
cat("2. Lasso can shrink some coefficients to zero.\n")
cat("3. Elastic net combines ridge and lasso behaviour.\n")
cat("4. Selected features are prediction features, not automatically causal factors.\n")
cat("5. Feature selection must be included inside validation to avoid leakage.\n")