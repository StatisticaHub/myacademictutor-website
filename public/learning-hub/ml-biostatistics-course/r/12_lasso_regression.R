# Machine Learning in Biostatistics
# Script 12: Lasso Regression
#
# This script fits lasso regression for diabetes prediction
# and shows which predictors are selected.

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

calculate_auc <- function(actual, risk) {
  roc_object <- roc(
    response = actual,
    predictor = risk,
    levels = c("neg", "pos"),
    quiet = TRUE
  )

  as.numeric(auc(roc_object))
}

lasso_auc <- calculate_auc(y_test_factor, lasso_risk)

cat("\nLasso model test AUC:\n")
print(lasso_auc)

lasso_coefficients <- coef(lasso_model, s = "lambda.min")

lasso_coefficients_table <- tibble(
  term = rownames(lasso_coefficients),
  coefficient = as.numeric(lasso_coefficients)
)

cat("\nAll lasso coefficients:\n")
print(lasso_coefficients_table)

lasso_selected <- lasso_coefficients_table %>%
  filter(coefficient != 0)

cat("\nFeatures selected by lasso:\n")
print(lasso_selected)

cat("\nNumber of selected terms including intercept:\n")
print(nrow(lasso_selected))

plot(lasso_model, main = "Lasso Cross-Validation")

cat("\nLearning points:\n")
cat("1. Lasso regression shrinks coefficients towards zero.\n")
cat("2. Lasso can shrink some coefficients exactly to zero.\n")
cat("3. Predictors with zero coefficients are removed from the model.\n")
cat("4. Lasso-selected features are prediction features, not automatically causal factors.\n")
cat("5. Feature selection must be included inside validation to avoid leakage.\n")