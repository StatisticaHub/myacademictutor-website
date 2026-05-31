# Machine Learning in Biostatistics
# Script 02: Validation, Overfitting and Data Leakage
#
# This script introduces the idea that training performance can look better
# than test performance.
#
# We use the Pima Indians Diabetes dataset because it is simple, familiar,
# and useful for teaching binary clinical prediction.
#
# Main ideas:
# - split data into training and test sets
# - fit a logistic regression model
# - compare training and test performance
# - demonstrate why test performance matters
# - show how leakage can make performance look artificially strong

library(tidyverse)
library(mlbench)
library(pROC)

set.seed(123)

data("PimaIndiansDiabetes")

diabetes_data <- PimaIndiansDiabetes %>%
  mutate(
    diabetes = factor(diabetes, levels = c("neg", "pos"))
  )

# Quick look at the data

glimpse(diabetes_data)

diabetes_data %>%
  count(diabetes)

# The outcome is diabetes status:
# neg = diabetes negative
# pos = diabetes positive
#
# The positive class is "pos".

# Train/test split

n <- nrow(diabetes_data)

train_index <- sample(
  x = seq_len(n),
  size = round(0.80 * n)
)

train_data <- diabetes_data[train_index, ]
test_data <- diabetes_data[-train_index, ]

train_data %>% count(diabetes)
test_data %>% count(diabetes)

# Fit a simple logistic regression model

model_basic <- glm(
  diabetes ~ pregnant + glucose + pressure + triceps +
    insulin + mass + pedigree + age,
  data = train_data,
  family = binomial
)

summary(model_basic)

# Predicted probabilities in training data

train_data <- train_data %>%
  mutate(
    predicted_risk = predict(
      model_basic,
      newdata = train_data,
      type = "response"
    )
  )

# Predicted probabilities in test data

test_data <- test_data %>%
  mutate(
    predicted_risk = predict(
      model_basic,
      newdata = test_data,
      type = "response"
    )
  )

# AUC in training data

train_roc <- roc(
  response = train_data$diabetes,
  predictor = train_data$predicted_risk,
  levels = c("neg", "pos")
)

train_auc <- auc(train_roc)

# AUC in test data

test_roc <- roc(
  response = test_data$diabetes,
  predictor = test_data$predicted_risk,
  levels = c("neg", "pos")
)

test_auc <- auc(test_roc)

performance_comparison <- tibble(
  dataset = c("Training data", "Test data"),
  auc = c(as.numeric(train_auc), as.numeric(test_auc))
)

print(performance_comparison)

# Interpretation:
#
# Training performance is usually optimistic because the model was fitted
# using the training data.
#
# Test performance is more useful because the test data were not used to
# fit the model.
#
# In medical prediction, we care about performance on future patients,
# not only on the patients used for model development.

# Convert probabilities to predicted classes using a 0.50 threshold

threshold <- 0.50

train_data <- train_data %>%
  mutate(
    predicted_class = ifelse(predicted_risk >= threshold, "pos", "neg"),
    predicted_class = factor(predicted_class, levels = c("neg", "pos"))
  )

test_data <- test_data %>%
  mutate(
    predicted_class = ifelse(predicted_risk >= threshold, "pos", "neg"),
    predicted_class = factor(predicted_class, levels = c("neg", "pos"))
  )

# Simple function to calculate classification metrics

calculate_metrics <- function(actual, predicted) {
  tab <- table(predicted, actual)

  tn <- tab["neg", "neg"]
  fp <- tab["pos", "neg"]
  fn <- tab["neg", "pos"]
  tp <- tab["pos", "pos"]

  accuracy <- (tp + tn) / sum(tab)
  sensitivity <- tp / (tp + fn)
  specificity <- tn / (tn + fp)

  tibble(
    accuracy = accuracy,
    sensitivity = sensitivity,
    specificity = specificity
  )
}

train_metrics <- calculate_metrics(
  actual = train_data$diabetes,
  predicted = train_data$predicted_class
)

test_metrics <- calculate_metrics(
  actual = test_data$diabetes,
  predicted = test_data$predicted_class
)

metric_comparison <- bind_rows(
  train_metrics %>% mutate(dataset = "Training data"),
  test_metrics %>% mutate(dataset = "Test data")
) %>%
  select(dataset, accuracy, sensitivity, specificity)

print(metric_comparison)

# Demonstrating a leakage problem

# Now we create an artificial leakage variable.
# This variable is almost the outcome itself, but with slight noise.
#
# In real medical data, leakage may be less obvious.
# For example, a post-diagnosis variable might accidentally be used
# to predict diagnosis at baseline.

diabetes_data_leakage <- diabetes_data %>%
  mutate(
    leakage_variable = ifelse(diabetes == "pos", 1, 0),
    leakage_variable = leakage_variable + rnorm(n(), mean = 0, sd = 0.05)
  )

train_data_leakage <- diabetes_data_leakage[train_index, ]
test_data_leakage <- diabetes_data_leakage[-train_index, ]

model_with_leakage <- glm(
  diabetes ~ pregnant + glucose + pressure + triceps +
    insulin + mass + pedigree + age + leakage_variable,
  data = train_data_leakage,
  family = binomial
)

test_data_leakage <- test_data_leakage %>%
  mutate(
    predicted_risk = predict(
      model_with_leakage,
      newdata = test_data_leakage,
      type = "response"
    )
  )

leakage_roc <- roc(
  response = test_data_leakage$diabetes,
  predictor = test_data_leakage$predicted_risk,
  levels = c("neg", "pos")
)

leakage_auc <- auc(leakage_roc)

leakage_comparison <- tibble(
  model = c("Correct model", "Model with leakage"),
  test_auc = c(as.numeric(test_auc), as.numeric(leakage_auc))
)

print(leakage_comparison)

# Interpretation:
#
# The leakage model will look extremely strong because it contains a variable
# that directly encodes the outcome.
#
# This is not a useful model.
#
# In real clinical prediction, leakage can occur when we use information
# that would not be available at the prediction time.
#
# Examples:
# - using discharge information to predict admission risk
# - using follow-up test results to predict baseline diagnosis
# - selecting features before validation
# - preprocessing the full dataset before train/test split

# Plot ROC curves for correct model and leakage model

plot(
  test_roc,
  col = "black",
  lwd = 2,
  main = "ROC Curve: Correct Model vs Leakage Model"
)

plot(
  leakage_roc,
  col = "gray40",
  lwd = 2,
  add = TRUE
)

abline(a = 0, b = 1, lty = 2)

legend(
  "bottomright",
  legend = c(
    paste0("Correct model, AUC = ", round(test_auc, 3)),
    paste0("Leakage model, AUC = ", round(leakage_auc, 3))
  ),
  lwd = 2,
  col = c("black", "gray40"),
  bty = "n"
)

# Final learning points

cat("\nLearning points:\n")
cat("1. Training performance is usually optimistic.\n")
cat("2. Test performance is more useful for estimating future performance.\n")
cat("3. Data leakage can make a model look unrealistically strong.\n")
cat("4. Predictors must be available at the intended prediction time.\n")
cat("5. Validation is central to medical machine learning.\n")