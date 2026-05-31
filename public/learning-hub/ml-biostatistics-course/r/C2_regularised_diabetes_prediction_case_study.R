# Machine Learning in Biostatistics
# Case Study 2: Regularised Diabetes Risk Prediction
#
# This script extends Case Study 1 by comparing ordinary logistic regression,
# ridge regression, lasso regression, and elastic net.
#
# It generates figures for the case study page and prints interpretation-friendly outputs.

library(tidyverse)
library(mlbench)
library(pROC)
library(glmnet)
library(broom)

set.seed(123)

dir.create("figures", showWarnings = FALSE)

data("PimaIndiansDiabetes")

diabetes_data <- PimaIndiansDiabetes %>%
  mutate(
    diabetes = factor(diabetes, levels = c("neg", "pos"))
  )

train_index <- sample(
  seq_len(nrow(diabetes_data)),
  size = round(0.80 * nrow(diabetes_data))
)

train_data <- diabetes_data[train_index, ]
test_data <- diabetes_data[-train_index, ]

x_train <- model.matrix(
  diabetes ~ pregnant + glucose + pressure + triceps +
    insulin + mass + pedigree + age,
  data = train_data
)[, -1]

x_test <- model.matrix(
  diabetes ~ pregnant + glucose + pressure + triceps +
    insulin + mass + pedigree + age,
  data = test_data
)[, -1]

y_train <- ifelse(train_data$diabetes == "pos", 1, 0)
y_test <- ifelse(test_data$diabetes == "pos", 1, 0)
y_test_factor <- test_data$diabetes

calculate_auc <- function(actual, risk) {
  roc_object <- roc(
    response = actual,
    predictor = risk,
    levels = c("neg", "pos"),
    quiet = TRUE
  )

  as.numeric(auc(roc_object))
}

calculate_brier_score <- function(actual_binary, risk) {
  mean((actual_binary - risk)^2)
}

create_calibration_data <- function(actual_binary, risk, model_name) {
  tibble(
    observed = actual_binary,
    predicted_risk = risk
  ) %>%
    mutate(
      risk_group = ntile(predicted_risk, 10)
    ) %>%
    group_by(risk_group) %>%
    summarise(
      mean_predicted_risk = mean(predicted_risk),
      observed_risk = mean(observed),
      n = n(),
      .groups = "drop"
    ) %>%
    mutate(
      model = model_name
    )
}

ordinary_model <- glm(
  diabetes ~ pregnant + glucose + pressure + triceps +
    insulin + mass + pedigree + age,
  data = train_data,
  family = binomial
)

ordinary_risk <- predict(
  ordinary_model,
  newdata = test_data,
  type = "response"
)

ridge_model <- cv.glmnet(
  x = x_train,
  y = y_train,
  family = "binomial",
  alpha = 0
)

ridge_risk <- as.numeric(
  predict(
    ridge_model,
    newx = x_test,
    s = "lambda.min",
    type = "response"
  )
)

lasso_model <- cv.glmnet(
  x = x_train,
  y = y_train,
  family = "binomial",
  alpha = 1
)

lasso_risk <- as.numeric(
  predict(
    lasso_model,
    newx = x_test,
    s = "lambda.min",
    type = "response"
  )
)

elastic_net_model <- cv.glmnet(
  x = x_train,
  y = y_train,
  family = "binomial",
  alpha = 0.5
)

elastic_net_risk <- as.numeric(
  predict(
    elastic_net_model,
    newx = x_test,
    s = "lambda.min",
    type = "response"
  )
)

model_comparison <- tibble(
  model = c(
    "Ordinary logistic regression",
    "Ridge regression",
    "Lasso regression",
    "Elastic net"
  ),
  auc = c(
    calculate_auc(y_test_factor, ordinary_risk),
    calculate_auc(y_test_factor, ridge_risk),
    calculate_auc(y_test_factor, lasso_risk),
    calculate_auc(y_test_factor, elastic_net_risk)
  ),
  brier_score = c(
    calculate_brier_score(y_test, ordinary_risk),
    calculate_brier_score(y_test, ridge_risk),
    calculate_brier_score(y_test, lasso_risk),
    calculate_brier_score(y_test, elastic_net_risk)
  )
)

cat("\nCase Study 2: Regularised Diabetes Risk Prediction\n")

cat("\nModel comparison using test AUC and Brier score:\n")
print(model_comparison)

model_comparison_long <- model_comparison %>%
  mutate(
    model = factor(
      model,
      levels = c(
        "Ordinary logistic regression",
        "Ridge regression",
        "Lasso regression",
        "Elastic net"
      )
    )
  ) %>%
  pivot_longer(
    cols = c(auc, brier_score),
    names_to = "metric",
    values_to = "value"
  ) %>%
  mutate(
    metric = recode(
      metric,
      auc = "AUC: higher is better",
      brier_score = "Brier score: lower is better"
    )
  )

performance_plot <- ggplot(
  model_comparison_long,
  aes(x = model, y = value)
) +
  geom_col() +
  coord_flip() +
  facet_wrap(~ metric, scales = "free_x") +
  labs(
    title = "Regularised Diabetes Prediction: Model Performance",
    subtitle = "AUC measures discrimination; Brier score measures probability prediction error",
    x = "Model",
    y = "Metric value"
  ) +
  theme_minimal(base_size = 13)

ggsave(
  filename = "figures/regularised_diabetes_model_performance.png",
  plot = performance_plot,
  width = 9,
  height = 5,
  dpi = 300
)

ordinary_coefficients <- tidy(ordinary_model) %>%
  select(term, ordinary_coefficient = estimate)

ridge_coefficients <- coef(ridge_model, s = "lambda.min")

ridge_coefficients_table <- tibble(
  term = rownames(ridge_coefficients),
  ridge_coefficient = as.numeric(ridge_coefficients)
)

coefficient_comparison <- ordinary_coefficients %>%
  left_join(ridge_coefficients_table, by = "term")

cat("\nOrdinary logistic regression coefficients versus ridge coefficients:\n")
print(coefficient_comparison)

coefficient_plot_data <- coefficient_comparison %>%
  filter(term != "(Intercept)") %>%
  pivot_longer(
    cols = c(ordinary_coefficient, ridge_coefficient),
    names_to = "model",
    values_to = "coefficient"
  ) %>%
  mutate(
    model = recode(
      model,
      ordinary_coefficient = "Ordinary logistic regression",
      ridge_coefficient = "Ridge regression"
    )
  )

coefficient_plot <- ggplot(
  coefficient_plot_data,
  aes(x = reorder(term, coefficient), y = coefficient, fill = model)
) +
  geom_col(position = "dodge") +
  coord_flip() +
  labs(
    title = "Ordinary Logistic Regression vs Ridge Coefficients",
    subtitle = "Ridge shrinks coefficients towards zero",
    x = "Predictor",
    y = "Coefficient",
    fill = "Model"
  ) +
  theme_minimal(base_size = 13)

ggsave(
  filename = "figures/regularised_diabetes_ridge_shrinkage.png",
  plot = coefficient_plot,
  width = 8,
  height = 5,
  dpi = 300
)

lasso_coefficients <- coef(lasso_model, s = "lambda.min")

lasso_selected <- tibble(
  term = rownames(lasso_coefficients),
  coefficient = as.numeric(lasso_coefficients)
) %>%
  filter(coefficient != 0)

cat("\nFeatures selected by lasso:\n")
print(lasso_selected)

elastic_net_coefficients <- coef(elastic_net_model, s = "lambda.min")

elastic_net_selected <- tibble(
  term = rownames(elastic_net_coefficients),
  coefficient = as.numeric(elastic_net_coefficients)
) %>%
  filter(coefficient != 0)

cat("\nFeatures selected by elastic net:\n")
print(elastic_net_selected)

selected_features <- bind_rows(
  lasso_selected %>%
    mutate(model = "Lasso regression"),
  elastic_net_selected %>%
    mutate(model = "Elastic net")
) %>%
  filter(term != "(Intercept)")

selected_features_plot <- ggplot(
  selected_features,
  aes(x = reorder(term, coefficient), y = coefficient)
) +
  geom_col() +
  coord_flip() +
  facet_wrap(~ model, scales = "free_y") +
  labs(
    title = "Selected Features from Lasso and Elastic Net",
    subtitle = "Non-zero coefficients are retained by the model",
    x = "Predictor",
    y = "Coefficient"
  ) +
  theme_minimal(base_size = 13)

ggsave(
  filename = "figures/regularised_diabetes_selected_features.png",
  plot = selected_features_plot,
  width = 9,
  height = 5,
  dpi = 300
)

ordinary_calibration <- create_calibration_data(
  actual_binary = y_test,
  risk = ordinary_risk,
  model_name = "Ordinary logistic regression"
)

ridge_calibration <- create_calibration_data(
  actual_binary = y_test,
  risk = ridge_risk,
  model_name = "Ridge regression"
)

lasso_calibration <- create_calibration_data(
  actual_binary = y_test,
  risk = lasso_risk,
  model_name = "Lasso regression"
)

elastic_net_calibration <- create_calibration_data(
  actual_binary = y_test,
  risk = elastic_net_risk,
  model_name = "Elastic net"
)

calibration_data <- bind_rows(
  ordinary_calibration,
  ridge_calibration,
  lasso_calibration,
  elastic_net_calibration
)

cat("\nCalibration data:\n")
print(calibration_data)

calibration_plot <- ggplot(
  calibration_data,
  aes(x = mean_predicted_risk, y = observed_risk)
) +
  geom_point(size = 2.5) +
  geom_line() +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  facet_wrap(~ model) +
  labs(
    title = "Calibration Comparison",
    subtitle = "Mean predicted risk compared with observed diabetes risk",
    x = "Mean predicted risk",
    y = "Observed risk"
  ) +
  theme_minimal(base_size = 13)

ggsave(
  filename = "figures/regularised_diabetes_calibration_comparison.png",
  plot = calibration_plot,
  width = 9,
  height = 7,
  dpi = 300
)

roc_ordinary <- roc(
  response = y_test_factor,
  predictor = ordinary_risk,
  levels = c("neg", "pos"),
  quiet = TRUE
)

roc_ridge <- roc(
  response = y_test_factor,
  predictor = ridge_risk,
  levels = c("neg", "pos"),
  quiet = TRUE
)

roc_lasso <- roc(
  response = y_test_factor,
  predictor = lasso_risk,
  levels = c("neg", "pos"),
  quiet = TRUE
)

roc_elastic_net <- roc(
  response = y_test_factor,
  predictor = elastic_net_risk,
  levels = c("neg", "pos"),
  quiet = TRUE
)

png(
  filename = "figures/regularised_diabetes_roc_comparison.png",
  width = 1800,
  height = 1400,
  res = 250
)

plot(
  roc_ordinary,
  main = "ROC Curves: Regularised Diabetes Prediction",
  lwd = 3
)

lines(roc_ridge, lwd = 3, lty = 2)
lines(roc_lasso, lwd = 3, lty = 3)
lines(roc_elastic_net, lwd = 3, lty = 4)

abline(a = 0, b = 1, lty = 2)

legend(
  "bottomright",
  legend = c(
    paste0("Ordinary logistic, AUC = ", round(model_comparison$auc[1], 3)),
    paste0("Ridge, AUC = ", round(model_comparison$auc[2], 3)),
    paste0("Lasso, AUC = ", round(model_comparison$auc[3], 3)),
    paste0("Elastic net, AUC = ", round(model_comparison$auc[4], 3))
  ),
  lwd = 3,
  lty = c(1, 2, 3, 4),
  bty = "n"
)

dev.off()

png(
  filename = "figures/regularised_diabetes_ridge_cv.png",
  width = 1800,
  height = 1400,
  res = 250
)

plot(ridge_model, main = "Ridge Cross-Validation")

dev.off()

png(
  filename = "figures/regularised_diabetes_lasso_cv.png",
  width = 1800,
  height = 1400,
  res = 250
)

plot(lasso_model, main = "Lasso Cross-Validation")

dev.off()

png(
  filename = "figures/regularised_diabetes_elastic_net_cv.png",
  width = 1800,
  height = 1400,
  res = 250
)

plot(elastic_net_model, main = "Elastic Net Cross-Validation")

dev.off()

cat("\nRegularisation tuning values:\n")
cat("Ridge lambda.min:", ridge_model$lambda.min, "\n")
cat("Lasso lambda.min:", lasso_model$lambda.min, "\n")
cat("Elastic net lambda.min:", elastic_net_model$lambda.min, "\n")

cat("\nExpected output files:\n")
cat("figures/regularised_diabetes_model_performance.png\n")
cat("figures/regularised_diabetes_ridge_shrinkage.png\n")
cat("figures/regularised_diabetes_selected_features.png\n")
cat("figures/regularised_diabetes_calibration_comparison.png\n")
cat("figures/regularised_diabetes_roc_comparison.png\n")
cat("figures/regularised_diabetes_ridge_cv.png\n")
cat("figures/regularised_diabetes_lasso_cv.png\n")
cat("figures/regularised_diabetes_elastic_net_cv.png\n")

cat("\nClinical interpretation prompts:\n")
cat("1. Which model has the highest test AUC?\n")
cat("2. Which model has the lowest Brier score?\n")
cat("3. Does ridge shrink coefficients compared with ordinary logistic regression?\n")
cat("4. Which features are selected by lasso?\n")
cat("5. Which features are selected by elastic net?\n")
cat("6. Are selected features automatically causal? No.\n")
cat("7. Do the calibration plots suggest reliable predicted probabilities?\n")
cat("8. Would these models need external validation before clinical use? Yes.\n")