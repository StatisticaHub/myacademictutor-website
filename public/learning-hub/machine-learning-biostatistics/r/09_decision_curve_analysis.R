# Machine Learning in Biostatistics
# Script 09: Decision Curve Analysis
#
# This script implements a simple decision curve analysis manually.
# It compares a model-based strategy with treat-all and treat-none strategies.

library(tidyverse)
library(mlbench)

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

calculate_net_benefit <- function(observed, predicted_risk, threshold) {
  predicted_positive <- ifelse(predicted_risk >= threshold, 1, 0)

  tp <- sum(predicted_positive == 1 & observed == 1)
  fp <- sum(predicted_positive == 1 & observed == 0)

  n <- length(observed)

  net_benefit <- (tp / n) - (fp / n) * (threshold / (1 - threshold))

  net_benefit
}

calculate_treat_all_net_benefit <- function(observed, threshold) {
  prevalence <- mean(observed == 1)

  prevalence - (1 - prevalence) * (threshold / (1 - threshold))
}

thresholds <- seq(0.05, 0.80, by = 0.01)

decision_curve <- tibble(
  threshold = thresholds
) %>%
  mutate(
    model = map_dbl(
      threshold,
      ~ calculate_net_benefit(
        observed = test_data$observed,
        predicted_risk = test_data$predicted_risk,
        threshold = .x
      )
    ),
    treat_all = map_dbl(
      threshold,
      ~ calculate_treat_all_net_benefit(
        observed = test_data$observed,
        threshold = .x
      )
    ),
    treat_none = 0
  )

cat("\nFirst few decision curve results:\n")
print(head(decision_curve))

plot(
  decision_curve$threshold,
  decision_curve$model,
  type = "l",
  lwd = 2,
  ylim = range(decision_curve$model, decision_curve$treat_all, decision_curve$treat_none),
  xlab = "Threshold probability",
  ylab = "Net benefit",
  main = "Decision Curve Analysis"
)

lines(decision_curve$threshold, decision_curve$treat_all, lty = 2, lwd = 2)
lines(decision_curve$threshold, decision_curve$treat_none, lty = 3, lwd = 2)

legend(
  "topright",
  legend = c("Model", "Treat all", "Treat none"),
  lty = c(1, 2, 3),
  lwd = 2,
  bty = "n"
)

relevant_thresholds <- decision_curve %>%
  filter(threshold >= 0.10, threshold <= 0.40) %>%
  summarise(
    mean_model_net_benefit = mean(model),
    mean_treat_all_net_benefit = mean(treat_all),
    mean_treat_none_net_benefit = mean(treat_none)
  )

cat("\nAverage net benefit between thresholds 0.10 and 0.40:\n")
print(relevant_thresholds)

cat("\nLearning points:\n")
cat("1. Decision curve analysis evaluates clinical usefulness.\n")
cat("2. Threshold probability reflects when action is taken.\n")
cat("3. A model is useful if it improves net benefit over simple alternatives.\n")
cat("4. Good AUC does not automatically mean clinical usefulness.\n")
cat("5. Decision curves must be interpreted using clinically realistic thresholds.\n")