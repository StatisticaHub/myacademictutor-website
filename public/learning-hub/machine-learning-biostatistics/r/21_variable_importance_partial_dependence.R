# Machine Learning in Biostatistics
# Script 21: Variable Importance and Partial Dependence
#
# This script fits a random forest and uses variable importance
# and partial dependence to interpret the model.
#
# Main ideas:
# 1. Variable importance describes how strongly a model uses predictors.
# 2. Importance is predictive importance, not causal importance.
# 3. Partial dependence shows average predicted risk across predictor values.
# 4. Partial dependence can be misleading when predictors are correlated.
# 5. Interpretation should be combined with clinical knowledge and validation.

library(tidyverse)
library(mlbench)
library(pROC)
library(randomForest)
library(pdp)

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

forest_model <- randomForest(
  diabetes ~ pregnant + glucose + pressure + triceps +
    insulin + mass + pedigree + age,
  data = train_data,
  ntree = 500,
  mtry = 3,
  importance = TRUE
)

forest_risk <- predict(
  forest_model,
  newdata = test_data,
  type = "prob"
)[, "pos"]

forest_roc <- roc(
  response = test_data$diabetes,
  predictor = forest_risk,
  levels = c("neg", "pos"),
  quiet = TRUE
)

forest_auc <- as.numeric(auc(forest_roc))
forest_brier <- mean((forest_risk - as.numeric(test_data$diabetes == "pos"))^2)

cat("\nRandom forest test AUC:\n")
print(forest_auc)

cat("\nRandom forest Brier score:\n")
print(forest_brier)

importance_table <- importance(forest_model) %>%
  as.data.frame() %>%
  rownames_to_column("predictor") %>%
  arrange(desc(MeanDecreaseGini))

cat("\nVariable importance table:\n")
print(importance_table)

varImpPlot(
  forest_model,
  main = "Random Forest Variable Importance"
)

glucose_partial <- partial(
  object = forest_model,
  pred.var = "glucose",
  train = train_data,
  prob = TRUE,
  which.class = "pos"
)

mass_partial <- partial(
  object = forest_model,
  pred.var = "mass",
  train = train_data,
  prob = TRUE,
  which.class = "pos"
)

age_partial <- partial(
  object = forest_model,
  pred.var = "age",
  train = train_data,
  prob = TRUE,
  which.class = "pos"
)

plotPartial(
  glucose_partial,
  main = "Partial Dependence of Diabetes Risk on Glucose",
  xlab = "Glucose",
  ylab = "Average predicted probability"
)

plotPartial(
  mass_partial,
  main = "Partial Dependence of Diabetes Risk on BMI / Mass",
  xlab = "BMI / Mass",
  ylab = "Average predicted probability"
)

plotPartial(
  age_partial,
  main = "Partial Dependence of Diabetes Risk on Age",
  xlab = "Age",
  ylab = "Average predicted probability"
)

two_way_partial <- partial(
  object = forest_model,
  pred.var = c("glucose", "mass"),
  train = train_data,
  prob = TRUE,
  which.class = "pos"
)

plotPartial(
  two_way_partial,
  levelplot = TRUE,
  main = "Two-Way Partial Dependence: Glucose and BMI"
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

cat("\nGrouped calibration summary:\n")
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

cat("\nInterpretation prompts:\n")
cat("1. Which predictors appear important for prediction?\n")
cat("2. Are important predictors clinically plausible?\n")
cat("3. Could correlated predictors distort importance rankings?\n")
cat("4. Do partial dependence plots show clinically plausible risk patterns?\n")
cat("5. Are the displayed predictor values realistic combinations of patients?\n")
cat("6. Is the model calibrated well enough for risk-based decisions?\n")

cat("\nLearning points:\n")
cat("1. Variable importance describes model use of predictors, not causality.\n")
cat("2. Partial dependence describes average model behaviour, not individual patient biology.\n")
cat("3. Correlated predictors can make importance and partial dependence misleading.\n")
cat("4. Interpretation should be checked against clinical knowledge and validation data.\n")
cat("5. Complex model interpretation should not replace calibration and clinical usefulness assessment.\n")
