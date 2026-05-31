# Machine Learning in Biostatistics
# Script 16: Decision Trees for Clinical Prediction
#
# This script fits a decision tree for diabetes prediction.
# It also compares the tree with logistic regression and checks
# discrimination, calibration and simple threshold-based classification.

library(tidyverse)
library(mlbench)
library(pROC)
library(rpart)
library(rpart.plot)

set.seed(123)

# 1. Load and prepare data

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

cat("\nDataset summary:\n")
cat("Number of patients:", nrow(diabetes_data), "\n")
cat("Outcome distribution:\n")
print(table(diabetes_data$diabetes))

# 2. Training/test split

train_index <- sample(
  seq_len(nrow(diabetes_data)),
  size = round(0.80 * nrow(diabetes_data))
)

train_data <- diabetes_data[train_index, ]
test_data <- diabetes_data[-train_index, ]

cat("\nTraining patients:", nrow(train_data), "\n")
cat("Test patients:", nrow(test_data), "\n")

# 3. Fit a decision tree

tree_model <- rpart(
  diabetes ~ pregnant + glucose + pressure + triceps +
    insulin + mass + pedigree + age,
  data = train_data,
  method = "class",
  control = rpart.control(
    cp = 0.01,
    minsplit = 20,
    minbucket = 7,
    maxdepth = 4
  )
)

cat("\nDecision tree complexity table:\n")
printcp(tree_model)

cat("\nDecision tree variable importance:\n")
print(tree_model$variable.importance)

# 4. Plot the tree

rpart.plot(
  tree_model,
  type = 2,
  extra = 104,
  fallen.leaves = TRUE,
  main = "Decision Tree for Diabetes Prediction"
)

# 5. Predict risk on test data

tree_risk <- predict(
  tree_model,
  newdata = test_data,
  type = "prob"
)[, "pos"]

test_predictions <- test_data %>%
  mutate(
    predicted_risk_tree = tree_risk,
    predicted_class_tree = factor(
      ifelse(predicted_risk_tree >= 0.50, "pos", "neg"),
      levels = c("neg", "pos")
    )
  )

cat("\nFirst few tree predictions:\n")
print(
  test_predictions %>%
    select(diabetes, predicted_risk_tree, predicted_class_tree) %>%
    head()
)

# 6. Confusion matrix at threshold 0.50

confusion_matrix <- table(
  observed = test_predictions$diabetes,
  predicted = test_predictions$predicted_class_tree
)

cat("\nConfusion matrix at threshold 0.50:\n")
print(confusion_matrix)

sensitivity <- confusion_matrix["pos", "pos"] /
  sum(confusion_matrix["pos", ])

specificity <- confusion_matrix["neg", "neg"] /
  sum(confusion_matrix["neg", ])

cat("\nSensitivity at threshold 0.50:", round(sensitivity, 3), "\n")
cat("Specificity at threshold 0.50:", round(specificity, 3), "\n")

# 7. ROC curve and AUC

tree_roc <- roc(
  response = test_predictions$diabetes,
  predictor = test_predictions$predicted_risk_tree,
  levels = c("neg", "pos"),
  quiet = TRUE
)

tree_auc <- as.numeric(auc(tree_roc))

cat("\nDecision tree test AUC:\n")
print(tree_auc)

plot(
  tree_roc,
  main = "Decision Tree ROC Curve",
  legacy.axes = TRUE
)

# 8. Calibration-style grouped risk summary

calibration_table <- test_predictions %>%
  mutate(risk_group = ntile(predicted_risk_tree, 5)) %>%
  group_by(risk_group) %>%
  summarise(
    mean_predicted_risk = mean(predicted_risk_tree),
    observed_risk = mean(diabetes == "pos"),
    n = n(),
    .groups = "drop"
  )

cat("\nGrouped calibration summary:\n")
print(calibration_table)

plot(
  calibration_table$mean_predicted_risk,
  calibration_table$observed_risk,
  xlim = c(0, 1),
  ylim = c(0, 1),
  xlab = "Mean predicted risk",
  ylab = "Observed risk",
  main = "Grouped Calibration: Decision Tree"
)

abline(0, 1, lty = 2)

# 9. Compare with logistic regression

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
  model = c("Decision tree", "Logistic regression"),
  test_auc = c(
    as.numeric(auc(tree_roc)),
    as.numeric(auc(logistic_roc))
  ),
  brier_score = c(
    mean((tree_risk - as.numeric(test_data$diabetes == "pos"))^2),
    mean((logistic_risk - as.numeric(test_data$diabetes == "pos"))^2)
  )
)

cat("\nModel comparison:\n")
print(model_comparison)

# 10. Learning points

cat("\nLearning points:\n")
cat("1. A decision tree splits patients into risk groups using predictor values.\n")
cat("2. Tree splits are easy to visualise and can resemble clinical decision rules.\n")
cat("3. A single tree can be unstable and may overfit if it grows too deep.\n")
cat("4. Test AUC assesses discrimination, but calibration should also be checked.\n")
cat("5. A clinically interpretable model is not automatically a clinically valid model.\n")
