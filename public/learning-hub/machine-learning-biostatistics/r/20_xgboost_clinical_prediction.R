# Machine Learning in Biostatistics
# Script 20: XGBoost for Clinical Prediction
#
# This script fits an XGBoost model for diabetes prediction.
# It uses a training/test split, early stopping, AUC, Brier score,
# feature importance and a grouped calibration summary.
#
# Main ideas:
# 1. XGBoost is an efficient implementation of gradient boosted trees.
# 2. It can model non-linear relationships and interactions.
# 3. Regularisation and subsampling help control overfitting.
# 4. Early stopping helps choose the number of boosting rounds.
# 5. XGBoost still needs calibration and clinical validation.

library(tidyverse)
library(mlbench)
library(pROC)
library(xgboost)

set.seed(123)

data("PimaIndiansDiabetes")

diabetes_data <- PimaIndiansDiabetes %>%
  mutate(
    diabetes = factor(diabetes, levels = c("neg", "pos"))
  )

cat("\nDataset summary:\n")
cat("Number of patients:", nrow(diabetes_data), "\n")
cat("Outcome distribution:\n")
print(table(diabetes_data$diabetes))

train_index <- sample(
  seq_len(nrow(diabetes_data)),
  size = round(0.80 * nrow(diabetes_data))
)

train_data <- diabetes_data[train_index, ]
test_data <- diabetes_data[-train_index, ]

cat("\nTraining patients:", nrow(train_data), "\n")
cat("Test patients:", nrow(test_data), "\n")

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
y_test <- ifelse(test_data$diabetes == "pos", 1, 0)

dtrain <- xgb.DMatrix(
  data = x_train,
  label = y_train
)

dtest <- xgb.DMatrix(
  data = x_test,
  label = y_test
)

watchlist <- list(
  train = dtrain,
  test = dtest
)

xgb_model <- xgb.train(
  data = dtrain,
  objective = "binary:logistic",
  eval_metric = "auc",
  nrounds = 500,
  max_depth = 3,
  eta = 0.03,
  subsample = 0.80,
  colsample_bytree = 0.80,
  min_child_weight = 5,
  lambda = 1,
  alpha = 0,
  watchlist = watchlist,
  early_stopping_rounds = 25,
  verbose = 0
)

cat("\nBest iteration selected by early stopping:\n")
print(xgb_model$best_iteration)

cat("\nBest score reported by XGBoost:\n")
print(xgb_model$best_score)

xgb_risk <- predict(
  xgb_model,
  newdata = dtest
)

xgb_roc <- roc(
  response = test_data$diabetes,
  predictor = xgb_risk,
  levels = c("neg", "pos"),
  quiet = TRUE
)

xgb_auc <- as.numeric(auc(xgb_roc))
xgb_brier <- mean((xgb_risk - y_test)^2)

cat("\nXGBoost test AUC:\n")
print(xgb_auc)

cat("\nXGBoost test Brier score:\n")
print(xgb_brier)

plot(
  xgb_roc,
  main = "XGBoost ROC Curve",
  legacy.axes = TRUE
)

importance_table <- xgb.importance(
  feature_names = colnames(x_train),
  model = xgb_model
)

cat("\nXGBoost feature importance:\n")
print(importance_table)

xgb.plot.importance(
  importance_matrix = importance_table,
  main = "XGBoost Feature Importance"
)

xgb_predictions <- test_data %>%
  mutate(
    predicted_risk = xgb_risk,
    observed_numeric = y_test
  )

calibration_table <- xgb_predictions %>%
  mutate(risk_group = ntile(predicted_risk, 5)) %>%
  group_by(risk_group) %>%
  summarise(
    mean_predicted_risk = mean(predicted_risk),
    observed_risk = mean(observed_numeric),
    n = n(),
    .groups = "drop"
  )

cat("\nGrouped calibration summary for XGBoost:\n")
print(calibration_table)

plot(
  calibration_table$mean_predicted_risk,
  calibration_table$observed_risk,
  xlim = c(0, 1),
  ylim = c(0, 1),
  xlab = "Mean predicted risk",
  ylab = "Observed risk",
  main = "Grouped Calibration: XGBoost"
)

abline(0, 1, lty = 2)

logistic_model <- glm(
  diabetes ~ pregnant + glucose + pressure + triceps +
    insulin + mass + pedigree + age,
  data = train_data,
  family = binomial
)

logistic_risk <- predict(
  logistic_model,
  newdata = test_data,
  type = "response"
)

logistic_roc <- roc(
  response = test_data$diabetes,
  predictor = logistic_risk,
  levels = c("neg", "pos"),
  quiet = TRUE
)

model_comparison <- tibble(
  model = c("XGBoost", "Logistic regression"),
  test_auc = c(
    xgb_auc,
    as.numeric(auc(logistic_roc))
  ),
  test_brier = c(
    xgb_brier,
    mean((logistic_risk - y_test)^2)
  )
)

cat("\nModel comparison:\n")
print(model_comparison)

cat("\nLearning points:\n")
cat("1. XGBoost is a fast and regularised implementation of gradient boosted trees.\n")
cat("2. It uses tuning parameters such as learning rate, max depth, subsampling and regularisation.\n")
cat("3. Early stopping helps reduce overfitting by stopping when validation performance stops improving.\n")
cat("4. Feature importance describes predictive use, not causality.\n")
cat("5. AUC should be complemented by calibration, Brier score and clinical usefulness assessment.\n")
cat("6. External validation is needed before strong clinical claims are made.\n")
