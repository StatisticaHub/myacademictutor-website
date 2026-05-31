# Machine Learning in Biostatistics
# Case Study 1: Diabetes Risk Prediction
#
# This script builds a full teaching case study using logistic regression.
# It generates figures for the case study page and prints interpretation-friendly outputs.

library(tidyverse)
library(mlbench)
library(pROC)
library(broom)

set.seed(123)

dir.create("figures", showWarnings = FALSE)

data("PimaIndiansDiabetes")

diabetes_data <- PimaIndiansDiabetes %>%
  mutate(
    diabetes = factor(diabetes, levels = c("neg", "pos"))
  )

outcome_plot <- diabetes_data %>%
  count(diabetes) %>%
  ggplot(aes(x = diabetes, y = n)) +
  geom_col() +
  labs(
    title = "Outcome Distribution",
    subtitle = "Diabetes status in the Pima Indians Diabetes dataset",
    x = "Diabetes status",
    y = "Number of patients"
  ) +
  theme_minimal(base_size = 13)

ggsave(
  filename = "figures/diabetes_outcome_distribution.png",
  plot = outcome_plot,
  width = 7,
  height = 5,
  dpi = 300
)

train_index <- sample(
  seq_len(nrow(diabetes_data)),
  size = round(0.80 * nrow(diabetes_data))
)

train_data <- diabetes_data[train_index, ]
test_data <- diabetes_data[-train_index, ]

logistic_model <- glm(
  diabetes ~ pregnant + glucose + pressure + triceps +
    insulin + mass + pedigree + age,
  data = train_data,
  family = binomial
)

model_summary <- tidy(logistic_model) %>%
  mutate(
    odds_ratio = exp(estimate),
    conf_low = exp(estimate - 1.96 * std.error),
    conf_high = exp(estimate + 1.96 * std.error)
  )

cat("\nModel summary with odds ratios:\n")
print(model_summary)

test_data <- test_data %>%
  mutate(
    predicted_risk = predict(logistic_model, newdata = test_data, type = "response"),
    observed = ifelse(diabetes == "pos", 1, 0)
  )

threshold <- 0.50

test_data <- test_data %>%
  mutate(
    predicted_class = factor(
      ifelse(predicted_risk >= threshold, "pos", "neg"),
      levels = c("neg", "pos")
    )
  )

tab <- table(
  Predicted = factor(test_data$predicted_class, levels = c("neg", "pos")),
  Actual = factor(test_data$diabetes, levels = c("neg", "pos"))
)

tn <- tab["neg", "neg"]
fp <- tab["pos", "neg"]
fn <- tab["neg", "pos"]
tp <- tab["pos", "pos"]

performance_table <- tibble(
  metric = c(
    "Accuracy",
    "Sensitivity",
    "Specificity",
    "Positive predictive value",
    "Negative predictive value"
  ),
  value = c(
    (tp + tn) / sum(tab),
    tp / (tp + fn),
    tn / (tn + fp),
    tp / (tp + fp),
    tn / (tn + fn)
  )
)

cat("\nConfusion matrix:\n")
print(tab)

cat("\nPerformance table:\n")
print(performance_table)

roc_object <- roc(
  response = test_data$diabetes,
  predictor = test_data$predicted_risk,
  levels = c("neg", "pos"),
  quiet = TRUE
)

auc_value <- auc(roc_object)

png(
  filename = "figures/diabetes_roc_curve.png",
  width = 1800,
  height = 1400,
  res = 250
)

plot(
  roc_object,
  main = paste0("ROC Curve: Diabetes Prediction, AUC = ", round(auc_value, 3)),
  lwd = 3
)

abline(a = 0, b = 1, lty = 2)

dev.off()

brier_score <- mean((test_data$predicted_risk - test_data$observed)^2)

cat("\nBrier score:\n")
print(brier_score)

calibration_data <- test_data %>%
  mutate(risk_group = ntile(predicted_risk, 10)) %>%
  group_by(risk_group) %>%
  summarise(
    mean_predicted_risk = mean(predicted_risk),
    observed_risk = mean(observed),
    n = n(),
    .groups = "drop"
  )

cat("\nCalibration data:\n")
print(calibration_data)

calibration_plot <- ggplot(
  calibration_data,
  aes(x = mean_predicted_risk, y = observed_risk)
) +
  geom_point(size = 3) +
  geom_line() +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  labs(
    title = "Calibration Plot",
    subtitle = "Mean predicted risk compared with observed diabetes risk",
    x = "Mean predicted risk",
    y = "Observed risk"
  ) +
  theme_minimal(base_size = 13)

ggsave(
  filename = "figures/diabetes_calibration_plot.png",
  plot = calibration_plot,
  width = 7,
  height = 5,
  dpi = 300
)

calculate_metrics <- function(actual, risk, threshold) {
  predicted <- factor(
    ifelse(risk >= threshold, "pos", "neg"),
    levels = c("neg", "pos")
  )

  tab <- table(
  Predicted = factor(predicted, levels = c("neg", "pos")),
  Actual = factor(actual, levels = c("neg", "pos"))
  )
  
  tn <- tab["neg", "neg"]
  fp <- tab["pos", "neg"]
  fn <- tab["neg", "pos"]
  tp <- tab["pos", "pos"]

  tibble(
    threshold = threshold,
    accuracy = (tp + tn) / sum(tab),
    sensitivity = tp / (tp + fn),
    specificity = tn / (tn + fp)
  )
}

threshold_results <- map_dfr(
  c(0.20, 0.30, 0.40, 0.50, 0.60, 0.70),
  ~ calculate_metrics(
    actual = test_data$diabetes,
    risk = test_data$predicted_risk,
    threshold = .x
  )
)

cat("\nThreshold results:\n")
print(threshold_results)

cat("\nExpected output files:\n")
cat("figures/diabetes_outcome_distribution.png\n")
cat("figures/diabetes_roc_curve.png\n")
cat("figures/diabetes_calibration_plot.png\n")

cat("\nClinical interpretation prompts:\n")
cat("1. Is sensitivity high enough for screening?\n")
cat("2. Is specificity high enough to avoid excessive false positives?\n")
cat("3. Does the ROC AUC suggest useful discrimination?\n")
cat("4. Does the calibration plot suggest reliable risk estimates?\n")
cat("5. Would this model need external validation before clinical use?\n")