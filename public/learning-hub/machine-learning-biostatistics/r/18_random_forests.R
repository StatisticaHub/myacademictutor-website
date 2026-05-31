# Machine Learning in Biostatistics
# Script 18: Random Forests
#
# This script fits a random forest for diabetes prediction.
# It compares a single decision tree with a random forest and
# demonstrates AUC, Brier score, variable importance and calibration.
#
# Main ideas:
# 1. A random forest averages many decision trees.
# 2. Bootstrap samples are used to build different trees.
# 3. Random predictor selection makes trees less similar.
# 4. Averaging many trees can reduce instability.
# 5. Variable importance is not causal importance.

library(tidyverse)
library(mlbench)
library(pROC)
library(rpart)
library(rpart.plot)
library(randomForest)

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

single_tree <- rpart(
  diabetes ~ pregnant + glucose + pressure + triceps +
    insulin + mass + pedigree + age,
  data = train_data,
  method = "class",
  control = rpart.control(
    cp = 0.01,
    maxdepth = 4,
    minsplit = 20
  )
)

single_tree_risk <- predict(
  single_tree,
  newdata = test_data,
  type = "prob"
)[, "pos"]

random_forest_model <- randomForest(
  diabetes ~ pregnant + glucose + pressure + triceps +
    insulin + mass + pedigree + age,
  data = train_data,
  ntree = 500,
  mtry = 3,
  importance = TRUE
)

forest_risk <- predict(
  random_forest_model,
  newdata = test_data,
  type = "prob"
)[, "pos"]

model_comparison <- tibble(
  model = c("Single decision tree", "Random forest"),
  test_auc = c(
    calculate_auc(test_data$diabetes, single_tree_risk),
    calculate_auc(test_data$diabetes, forest_risk)
  ),
  test_brier = c(
    calculate_brier(test_data$diabetes, single_tree_risk),
    calculate_brier(test_data$diabetes, forest_risk)
  )
)

cat("\nModel comparison:\n")
print(model_comparison)

cat("\nRandom forest model summary:\n")
print(random_forest_model)

cat("\nOut-of-bag error by number of trees is stored in the model object.\n")
cat("Final out-of-bag error estimate:\n")
print(random_forest_model$err.rate[nrow(random_forest_model$err.rate), ])

importance_table <- importance(random_forest_model) %>%
  as.data.frame() %>%
  rownames_to_column("predictor") %>%
  arrange(desc(MeanDecreaseGini))

cat("\nVariable importance table:\n")
print(importance_table)

varImpPlot(
  random_forest_model,
  main = "Random Forest Variable Importance"
)

forest_predictions <- test_data %>%
  mutate(
    predicted_risk = forest_risk,
    observed_numeric = as.numeric(diabetes == "pos")
  )

calibration_table <- forest_predictions %>%
  mutate(risk_group = ntile(predicted_risk, 5)) %>%
  group_by(risk_group) %>%
  summarise(
    mean_predicted_risk = mean(predicted_risk),
    observed_risk = mean(observed_numeric),
    n = n(),
    .groups = "drop"
  )

cat("\nGrouped calibration summary for random forest:\n")
print(calibration_table)

plot(
  calibration_table$mean_predicted_risk,
  calibration_table$observed_risk,
  xlim = c(0, 1),
  ylim = c(0, 1),
  xlab = "Mean predicted risk",
  ylab = "Observed risk",
  main = "Grouped Calibration: Random Forest"
)

abline(0, 1, lty = 2)

forest_roc <- roc(
  response = test_data$diabetes,
  predictor = forest_risk,
  levels = c("neg", "pos"),
  quiet = TRUE
)

plot(
  forest_roc,
  main = "Random Forest ROC Curve",
  legacy.axes = TRUE
)

rpart.plot(
  single_tree,
  type = 2,
  extra = 104,
  fallen.leaves = TRUE,
  main = "Single Decision Tree"
)

cat("\nLearning points:\n")
cat("1. A random forest averages predictions from many decision trees.\n")
cat("2. Bootstrap sampling makes each tree use a slightly different dataset.\n")
cat("3. Random predictor selection makes the trees less similar to each other.\n")
cat("4. Averaging can reduce instability compared with a single tree.\n")
cat("5. Variable importance should be interpreted as predictive importance, not causal importance.\n")
cat("6. Random forests still need calibration, clinical usefulness assessment and external validation.\n")
