# Machine Learning in Biostatistics
# Script 03: Supervised Learning Basics
#
# This script introduces a basic supervised learning workflow:
# define the outcome, define predictors, split the data, fit a model,
# and evaluate predictions.

library(tidyverse)
library(mlbench)
library(pROC)

set.seed(123)

data("PimaIndiansDiabetes")

diabetes_data <- PimaIndiansDiabetes %>%
  mutate(
    diabetes = factor(diabetes, levels = c("neg", "pos"))
  )

cat("\nDataset overview:\n")
print(glimpse(diabetes_data))

cat("\nOutcome distribution:\n")
print(diabetes_data %>% count(diabetes))

predictor_names <- c(
  "pregnant",
  "glucose",
  "pressure",
  "triceps",
  "insulin",
  "mass",
  "pedigree",
  "age"
)

outcome_name <- "diabetes"

cat("\nOutcome:\n")
print(outcome_name)

cat("\nPredictors:\n")
print(predictor_names)

n <- nrow(diabetes_data)

train_index <- sample(seq_len(n), size = round(0.80 * n))

train_data <- diabetes_data[train_index, ]
test_data <- diabetes_data[-train_index, ]

cat("\nTraining outcome distribution:\n")
print(train_data %>% count(diabetes))

cat("\nTest outcome distribution:\n")
print(test_data %>% count(diabetes))

model_basic <- glm(
  diabetes ~ pregnant + glucose + pressure + triceps +
    insulin + mass + pedigree + age,
  data = train_data,
  family = binomial
)

cat("\nModel summary:\n")
print(summary(model_basic))

test_data <- test_data %>%
  mutate(
    predicted_risk = predict(model_basic, newdata = test_data, type = "response")
  )

cat("\nFirst few predicted risks:\n")
print(
  test_data %>%
    select(diabetes, predicted_risk) %>%
    head(10)
)

threshold <- 0.50

test_data <- test_data %>%
  mutate(
    predicted_class = factor(
      ifelse(predicted_risk >= threshold, "pos", "neg"),
      levels = c("neg", "pos")
    )
  )

cat("\nConfusion matrix at threshold 0.50:\n")
print(table(Predicted = test_data$predicted_class, Actual = test_data$diabetes))

roc_object <- roc(
  response = test_data$diabetes,
  predictor = test_data$predicted_risk,
  levels = c("neg", "pos"),
  quiet = TRUE
)

cat("\nTest AUC:\n")
print(auc(roc_object))

plot(
  roc_object,
  lwd = 2,
  main = paste0("ROC Curve: Supervised Learning Example, AUC = ", round(auc(roc_object), 3))
)

abline(a = 0, b = 1, lty = 2)

cat("\nLearning points:\n")
cat("1. Supervised learning requires predictors and a known outcome.\n")
cat("2. The model learns from training data.\n")
cat("3. Test data are used to evaluate performance on unseen patients.\n")
cat("4. Predicted probabilities can be converted into classes using a threshold.\n")
cat("5. AUC summarises discrimination across thresholds.\n")