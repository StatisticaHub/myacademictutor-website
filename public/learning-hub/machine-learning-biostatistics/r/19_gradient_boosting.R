# Machine Learning in Biostatistics
# Script 19: Gradient Boosting
#
# This script fits a gradient boosting model for diabetes prediction.
# It compares gradient boosting with logistic regression and shows
# AUC, Brier score, relative influence and grouped calibration.
#
# Main ideas:
# 1. Gradient boosting builds trees sequentially.
# 2. Each new tree tries to improve the current model.
# 3. Learning rate controls how strongly each new tree contributes.
# 4. Tree depth controls how complex each tree can be.
# 5. Boosting can overfit if tuning is careless.

library(tidyverse)
library(mlbench)
library(pROC)
library(gbm)

set.seed(123)

data("PimaIndiansDiabetes")

diabetes_data <- PimaIndiansDiabetes %>%
  mutate(
    diabetes = factor(diabetes, levels = c("neg", "pos")),
    diabetes_numeric = ifelse(diabetes == "pos", 1, 0)
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

calculate_auc <- function(actual, risk) {
  roc_object <- roc(
    response = actual,
    predictor = risk,
    levels = c("neg", "pos"),
    quiet = TRUE
  )

  as.numeric(auc(roc_object))
}

calculate_brier <- function(actual, risk) {
  observed <- as.numeric(actual == "pos")
  mean((risk - observed)^2)
}

boost_model <- gbm(
  diabetes_numeric ~ pregnant + glucose + pressure + triceps +
    insulin + mass + pedigree + age,
  data = train_data,
  distribution = "bernoulli",
  n.trees = 1500,
  interaction.depth = 2,
  shrinkage = 0.01,
  n.minobsinnode = 10,
  cv.folds = 5,
  verbose = FALSE
)

best_trees <- gbm.perf(
  boost_model,
  method = "cv",
  plot.it = TRUE
)

cat("\nBest number of trees selected using internal cross-validation:\n")
print(best_trees)

boost_risk <- predict(
  boost_model,
  newdata = test_data,
  n.trees = best_trees,
  type = "response"
)

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

model_comparison <- tibble(
  model = c("Gradient boosting", "Logistic regression"),
  test_auc = c(
    calculate_auc(test_data$diabetes, boost_risk),
    calculate_auc(test_data$diabetes, logistic_risk)
  ),
  test_brier = c(
    calculate_brier(test_data$diabetes, boost_risk),
    calculate_brier(test_data$diabetes, logistic_risk)
  )
)

cat("\nModel comparison:\n")
print(model_comparison)

relative_influence <- summary(
  boost_model,
  n.trees = best_trees,
  plotit = FALSE
)

cat("\nRelative influence table:\n")
print(relative_influence)

boost_predictions <- test_data %>%
  mutate(
    predicted_risk = boost_risk,
    observed_numeric = as.numeric(diabetes == "pos")
  )

calibration_table <- boost_predictions %>%
  mutate(risk_group = ntile(predicted_risk, 5)) %>%
  group_by(risk_group) %>%
  summarise(
    mean_predicted_risk = mean(predicted_risk),
    observed_risk = mean(observed_numeric),
    n = n(),
    .groups = "drop"
  )

cat("\nGrouped calibration summary for gradient boosting:\n")
print(calibration_table)

plot(
  calibration_table$mean_predicted_risk,
  calibration_table$observed_risk,
  xlim = c(0, 1),
  ylim = c(0, 1),
  xlab = "Mean predicted risk",
  ylab = "Observed risk",
  main = "Grouped Calibration: Gradient Boosting"
)

abline(0, 1, lty = 2)

boost_roc <- roc(
  response = test_data$diabetes,
  predictor = boost_risk,
  levels = c("neg", "pos"),
  quiet = TRUE
)

plot(
  boost_roc,
  main = "Gradient Boosting ROC Curve",
  legacy.axes = TRUE
)

cat("\nLearning points:\n")
cat("1. Gradient boosting builds trees sequentially rather than independently.\n")
cat("2. Each new tree improves the current model by focusing on remaining errors.\n")
cat("3. Important tuning choices include number of trees, tree depth and learning rate.\n")
cat("4. Internal cross-validation can help choose the number of trees.\n")
cat("5. Boosting can overfit if too many trees or too much complexity are used.\n")
cat("6. A boosted model still needs calibration and clinical usefulness assessment.\n")
