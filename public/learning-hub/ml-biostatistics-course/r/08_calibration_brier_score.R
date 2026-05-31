# Machine Learning in Biostatistics
# Script 08: Calibration and Brier Score
#
# This script assesses calibration and calculates the Brier score
# for a diabetes prediction model.

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
    predicted_risk = predict(model, newdata = test_data, type = "response"),
    observed = ifelse(diabetes == "pos", 1, 0)
  )

brier_score <- mean((test_data$predicted_risk - test_data$observed)^2)

cat("\nBrier score:\n")
print(brier_score)

roc_object <- roc(
  response = test_data$diabetes,
  predictor = test_data$predicted_risk,
  levels = c("neg", "pos"),
  quiet = TRUE
)

cat("\nAUC:\n")
print(auc(roc_object))

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

plot(
  calibration_data$mean_predicted_risk,
  calibration_data$observed_risk,
  xlim = c(0, 1),
  ylim = c(0, 1),
  pch = 19,
  xlab = "Mean predicted risk",
  ylab = "Observed risk",
  main = "Calibration Plot"
)

lines(
  calibration_data$mean_predicted_risk,
  calibration_data$observed_risk
)

abline(a = 0, b = 1, lty = 2)

# Calibration model
#
# A simple calibration model regresses observed outcome on the logit
# of predicted risk. This gives a rough calibration intercept and slope.

eps <- 1e-6

test_data <- test_data %>%
  mutate(
    predicted_risk_safe = pmin(pmax(predicted_risk, eps), 1 - eps),
    logit_predicted_risk = log(predicted_risk_safe / (1 - predicted_risk_safe))
  )

calibration_model <- glm(
  observed ~ logit_predicted_risk,
  data = test_data,
  family = binomial
)

calibration_intercept <- coef(calibration_model)[1]
calibration_slope <- coef(calibration_model)[2]

calibration_summary <- tibble(
  calibration_intercept = calibration_intercept,
  calibration_slope = calibration_slope,
  brier_score = brier_score,
  auc = as.numeric(auc(roc_object))
)

cat("\nCalibration summary:\n")
print(calibration_summary)

cat("\nLearning points:\n")
cat("1. AUC assesses discrimination, not calibration.\n")
cat("2. Calibration checks whether predicted risks match observed risks.\n")
cat("3. Points above the diagonal suggest underestimation of risk.\n")
cat("4. Points below the diagonal suggest overestimation of risk.\n")
cat("5. The Brier score measures probability prediction error.\n")