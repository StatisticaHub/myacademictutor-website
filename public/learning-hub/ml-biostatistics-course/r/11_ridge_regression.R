# Machine Learning in Biostatistics
# Script 11: Ridge Regression
#
# This script fits ordinary logistic regression and ridge regression
# for diabetes prediction and compares their test AUC.

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
    "Ridge regression"
  ),
  auc = c(
    calculate_auc(y_test_factor, ordinary_risk),
    calculate_auc(y_test_factor, ridge_risk)
  )
)

cat("\nModel comparison using test AUC:\n")
print(model_comparison)

ordinary_coefficients <- tibble(
  term = names(coef(ordinary_model)),
  ordinary_coefficient = as.numeric(coef(ordinary_model))
)

ridge_coefficients <- coef(ridge_model, s = "lambda.min")

ridge_coefficients_table <- tibble(
  term = rownames(ridge_coefficients),
  ridge_coefficient = as.numeric(ridge_coefficients)
)

coefficient_comparison <- ordinary_coefficients %>%
  left_join(ridge_coefficients_table, by = "term")

cat("\nOrdinary logistic regression coefficients versus ridge coefficients:\n")
print(coefficient_comparison)

plot(ridge_model, main = "Ridge Cross-Validation")

cat("\nLearning points:\n")
cat("1. Ridge regression shrinks coefficients towards zero.\n")
cat("2. Ridge usually keeps predictors in the model.\n")
cat("3. Ridge can be useful when predictors are correlated.\n")
cat("4. Ridge may reduce overfitting compared with ordinary logistic regression.\n")
cat("5. Ridge models still need validation and calibration assessment.\n")