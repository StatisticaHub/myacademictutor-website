# Machine Learning in Biostatistics
# Script 13: Elastic Net
#
# This script fits elastic net models with different alpha values
# and compares their performance for diabetes prediction.

library(tidyverse)
library(mlbench)
library(pROC)
library(glmnet)

set.seed(123)

data("PimaIndiansDiabetes")

diabetes_data <- PimaIndiansDiabetes %>%
  mutate(
    diabetes = factor(diabetes, levels = c("neg", "pos"))
  )

train_index <- sample(seq_len(nrow(diabetes_data)), size = round(0.80 * nrow(diabetes_data)))

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

alpha_values <- c(0, 0.25, 0.50, 0.75, 1)

elastic_net_results <- map_dfr(alpha_values, function(current_alpha) {
  elastic_net_model <- cv.glmnet(
    x = x_train,
    y = y_train,
    family = "binomial",
    alpha = current_alpha
  )

  elastic_net_risk <- as.numeric(
    predict(
      elastic_net_model,
      newx = x_test,
      s = "lambda.min",
      type = "response"
    )
  )

  elastic_net_auc <- calculate_auc(y_test_factor, elastic_net_risk)

  tibble(
    alpha = current_alpha,
    lambda_min = elastic_net_model$lambda.min,
    test_auc = elastic_net_auc,
    model_object = list(elastic_net_model)
  )
})

cat("\nElastic net comparison across alpha values:\n")
print(
  elastic_net_results %>%
    select(alpha, lambda_min, test_auc)
)

best_elastic_net <- elastic_net_results %>%
  arrange(desc(test_auc)) %>%
  slice(1)

best_alpha <- best_elastic_net$alpha
best_model <- best_elastic_net$model_object[[1]]

best_coefficients <- coef(best_model, s = "lambda.min")

best_selected <- tibble(
  term = rownames(best_coefficients),
  coefficient = as.numeric(best_coefficients)
) %>%
  filter(coefficient != 0)

cat("\nBest alpha based on test AUC:\n")
print(best_alpha)

cat("\nFeatures selected by the best elastic net model:\n")
print(best_selected)

plot(
  elastic_net_results$alpha,
  elastic_net_results$test_auc,
  type = "b",
  xlab = "Alpha",
  ylab = "Test AUC",
  main = "Elastic Net: Alpha versus Test AUC"
)

cat("\nLearning points:\n")
cat("1. Elastic net combines ridge and lasso behaviour.\n")
cat("2. Alpha controls the balance between ridge and lasso.\n")
cat("3. Alpha = 0 gives ridge regression.\n")
cat("4. Alpha = 1 gives lasso regression.\n")
cat("5. Elastic net can be useful when predictors are correlated.\n")