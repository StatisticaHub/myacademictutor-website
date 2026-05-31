# Machine Learning in Biostatistics
# Script 04: Logistic Regression for Clinical Prediction
#
# This script uses logistic regression as a clinical prediction model.
# The focus is on predicted probabilities, odds ratios, and test-set performance.

library(tidyverse)
library(mlbench)
library(pROC)
library(broom)

set.seed(123)

data("PimaIndiansDiabetes")

diabetes_data <- PimaIndiansDiabetes %>%
  mutate(
    diabetes = factor(diabetes, levels = c("neg", "pos"))
  )

train_index <- sample(seq_len(nrow(diabetes_data)), size = round(0.80 * nrow(diabetes_data)))

train_data <- diabetes_data[train_index, ]
test_data <- diabetes_data[-train_index, ]

logistic_model <- glm(
  diabetes ~ pregnant + glucose + pressure + triceps +
    insulin + mass + pedigree + age,
  data = train_data,
  family = binomial
)

cat("\nLogistic regression coefficients:\n")
print(tidy(logistic_model))

odds_ratio_table <- tidy(logistic_model) %>%
  mutate(
    odds_ratio = exp(estimate),
    conf_low = exp(estimate - 1.96 * std.error),
    conf_high = exp(estimate + 1.96 * std.error)
  ) %>%
  select(term, estimate, odds_ratio, conf_low, conf_high, p.value)

cat("\nOdds ratio table:\n")
print(odds_ratio_table)

test_data <- test_data %>%
  mutate(
    predicted_risk = predict(logistic_model, newdata = test_data, type = "response")
  )

cat("\nPredicted risks for first 10 test patients:\n")
print(
  test_data %>%
    select(diabetes, glucose, mass, age, predicted_risk) %>%
    head(10)
)

risk_summary <- test_data %>%
  summarise(
    min_risk = min(predicted_risk),
    median_risk = median(predicted_risk),
    mean_risk = mean(predicted_risk),
    max_risk = max(predicted_risk)
  )

cat("\nPredicted risk summary:\n")
print(risk_summary)

threshold <- 0.50

test_data <- test_data %>%
  mutate(
    predicted_class = factor(
      ifelse(predicted_risk >= threshold, "pos", "neg"),
      levels = c("neg", "pos")
    )
  )

cat("\nConfusion matrix:\n")
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
  main = paste0("ROC Curve: Logistic Regression, AUC = ", round(auc(roc_object), 3))
)

abline(a = 0, b = 1, lty = 2)

cat("\nClinical interpretation prompts:\n")
cat("1. Which predictors have clinically sensible directions?\n")
cat("2. Does a predictor being statistically significant prove good prediction? No.\n")
cat("3. Are predicted probabilities useful for risk stratification?\n")
cat("4. Should threshold 0.50 always be used? No.\n")
cat("5. Would this model need external validation? Yes.\n")