# Machine Learning in Biostatistics
# Script 17: Tree Overfitting, Tree Depth and Pruning
#
# This script shows why decision trees can overfit.
# It compares a shallow tree, a deep tree, and a pruned tree
# for diabetes prediction.
#
# Main ideas:
# 1. A shallow tree may be too simple.
# 2. A deep tree may fit training data too closely.
# 3. A pruned tree removes unnecessary branches.
# 4. Training performance and test performance should be compared.
# 5. Calibration should still be checked after choosing a tree.

library(tidyverse)
library(mlbench)
library(pROC)
library(rpart)
library(rpart.plot)

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

fit_tree <- function(cp, maxdepth, minsplit, minbucket) {
  rpart(
    diabetes ~ pregnant + glucose + pressure + triceps +
      insulin + mass + pedigree + age,
    data = train_data,
    method = "class",
    control = rpart.control(
      cp = cp,
      maxdepth = maxdepth,
      minsplit = minsplit,
      minbucket = minbucket
    )
  )
}

evaluate_tree <- function(model, model_name) {
  train_risk <- predict(
    model,
    newdata = train_data,
    type = "prob"
  )[, "pos"]

  test_risk <- predict(
    model,
    newdata = test_data,
    type = "prob"
  )[, "pos"]

  tibble(
    model = model_name,
    number_of_splits = sum(model$frame$var != "<leaf>"),
    terminal_nodes = sum(model$frame$var == "<leaf>"),
    train_auc = calculate_auc(train_data$diabetes, train_risk),
    test_auc = calculate_auc(test_data$diabetes, test_risk),
    train_brier = calculate_brier(train_data$diabetes, train_risk),
    test_brier = calculate_brier(test_data$diabetes, test_risk)
  )
}

shallow_tree <- fit_tree(
  cp = 0.02,
  maxdepth = 2,
  minsplit = 40,
  minbucket = 20
)

deep_tree <- fit_tree(
  cp = 0.0001,
  maxdepth = 20,
  minsplit = 2,
  minbucket = 1
)

cat("\nShallow tree complexity table:\n")
printcp(shallow_tree)

cat("\nDeep tree complexity table:\n")
printcp(deep_tree)

best_cp <- deep_tree$cptable[
  which.min(deep_tree$cptable[, "xerror"]),
  "CP"
]

pruned_tree <- prune(
  deep_tree,
  cp = best_cp
)

cat("\nBest complexity parameter selected from cross-validation error:\n")
print(best_cp)

cat("\nPruned tree complexity table:\n")
printcp(pruned_tree)

performance_comparison <- bind_rows(
  evaluate_tree(shallow_tree, "Shallow tree"),
  evaluate_tree(deep_tree, "Deep tree"),
  evaluate_tree(pruned_tree, "Pruned tree")
)

cat("\nTree performance comparison:\n")
print(performance_comparison)

performance_long <- performance_comparison %>%
  select(model, train_auc, test_auc) %>%
  pivot_longer(
    cols = c(train_auc, test_auc),
    names_to = "dataset",
    values_to = "auc"
  )

ggplot(performance_long, aes(x = model, y = auc, fill = dataset)) +
  geom_col(position = "dodge") +
  ylim(0, 1) +
  labs(
    title = "Training AUC versus Test AUC",
    x = "Tree model",
    y = "AUC"
  )

plotcp(
  deep_tree,
  main = "Complexity Parameter Plot for Deep Tree"
)

rpart.plot(
  shallow_tree,
  type = 2,
  extra = 104,
  fallen.leaves = TRUE,
  main = "Shallow Decision Tree"
)

rpart.plot(
  pruned_tree,
  type = 2,
  extra = 104,
  fallen.leaves = TRUE,
  main = "Pruned Decision Tree"
)

pruned_test_risk <- predict(
  pruned_tree,
  newdata = test_data,
  type = "prob"
)[, "pos"]

calibration_table <- test_data %>%
  mutate(predicted_risk = pruned_test_risk) %>%
  mutate(risk_group = ntile(predicted_risk, 5)) %>%
  group_by(risk_group) %>%
  summarise(
    mean_predicted_risk = mean(predicted_risk),
    observed_risk = mean(diabetes == "pos"),
    n = n(),
    .groups = "drop"
  )

cat("\nGrouped calibration summary for pruned tree:\n")
print(calibration_table)

plot(
  calibration_table$mean_predicted_risk,
  calibration_table$observed_risk,
  xlim = c(0, 1),
  ylim = c(0, 1),
  xlab = "Mean predicted risk",
  ylab = "Observed risk",
  main = "Grouped Calibration: Pruned Tree"
)

abline(0, 1, lty = 2)

cat("\nLearning points:\n")
cat("1. A shallow tree may underfit if it is too simple.\n")
cat("2. A deep tree may overfit if it follows noise in the training data.\n")
cat("3. Overfitting can appear as high training AUC but lower test AUC.\n")
cat("4. Pruning removes unnecessary branches and can improve generalisation.\n")
cat("5. A pruned tree still needs discrimination, calibration, and external validation assessment.\n")
