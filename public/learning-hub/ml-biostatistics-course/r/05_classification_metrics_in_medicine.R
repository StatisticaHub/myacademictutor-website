# Machine Learning in Biostatistics
# Script 05: Classification Metrics in Medicine
#
# This script shows why accuracy is not enough.
# We calculate sensitivity, specificity, PPV, NPV, accuracy, and AUC.

library(tidyverse)
library(mlbench)
library(pROC)

set.seed(123)

data("PimaIndiansDiabetes")

diabetes_data <- PimaIndiansDiabetes %>%
  mutate(
    diabetes = factor(diabetes, levels = c("neg", "pos"))
  )

train_index <- sample(seq_len(nrow(diabetes_data)), size = round(0.80 * nrow(diabetes_data)))

train_data <- diabetes_data[train_index, ]
test_data <- diabetes_data[-train_index, ]

model <- glm(
  diabetes ~ pregnant + glucose + pressure + triceps +
    insulin + mass + pedigree + age,
  data = train_data,
  family = binomial
)

test_data <- test_data %>%
  mutate(
    predicted_risk = predict(model, newdata = test_data, type = "response")
  )

calculate_classification_metrics <- function(actual, risk, threshold) {
  predicted <- factor(
    ifelse(risk >= threshold, "pos", "neg"),
    levels = c("neg", "pos")
  )

  tab <- table(Predicted = predicted, Actual = actual)

  tn <- tab["neg", "neg"]
  fp <- tab["pos", "neg"]
  fn <- tab["neg", "pos"]
  tp <- tab["pos", "pos"]

  accuracy <- (tp + tn) / sum(tab)
  sensitivity <- tp / (tp + fn)
  specificity <- tn / (tn + fp)
  ppv <- tp / (tp + fp)
  npv <- tn / (tn + fn)

  tibble(
    threshold = threshold,
    accuracy = accuracy,
    sensitivity = sensitivity,
    specificity = specificity,
    ppv = ppv,
    npv = npv,
    true_positive = tp,
    false_positive = fp,
    false_negative = fn,
    true_negative = tn
  )
}

metrics_050 <- calculate_classification_metrics(
  actual = test_data$diabetes,
  risk = test_data$predicted_risk,
  threshold = 0.50
)

cat("\nClassification metrics at threshold 0.50:\n")
print(metrics_050)

threshold_grid <- seq(0.20, 0.80, by = 0.10)

threshold_results <- map_dfr(
  threshold_grid,
  ~ calculate_classification_metrics(
    actual = test_data$diabetes,
    risk = test_data$predicted_risk,
    threshold = .x
  )
)

cat("\nMetrics across different thresholds:\n")
print(threshold_results)

roc_object <- roc(
  response = test_data$diabetes,
  predictor = test_data$predicted_risk,
  levels = c("neg", "pos"),
  quiet = TRUE
)

cat("\nAUC:\n")
print(auc(roc_object))

plot(
  roc_object,
  lwd = 2,
  main = paste0("ROC Curve: Classification Metrics, AUC = ", round(auc(roc_object), 3))
)

abline(a = 0, b = 1, lty = 2)

plot(
  threshold_results$threshold,
  threshold_results$sensitivity,
  type = "b",
  ylim = c(0, 1),
  xlab = "Threshold",
  ylab = "Metric value",
  main = "Sensitivity and Specificity Across Thresholds"
)

lines(
  threshold_results$threshold,
  threshold_results$specificity,
  type = "b",
  lty = 2
)

legend(
  "right",
  legend = c("Sensitivity", "Specificity"),
  lty = c(1, 2),
  pch = 1,
  bty = "n"
)

cat("\nClinical interpretation:\n")
cat("1. Lower thresholds usually increase sensitivity.\n")
cat("2. Higher thresholds usually increase specificity.\n")
cat("3. Screening often prioritises sensitivity.\n")
cat("4. Confirmatory decisions may require higher specificity.\n")
cat("5. Accuracy alone can hide clinically important errors.\n")