# Machine Learning in Biostatistics
# R setup file

packages <- c(
  "tidyverse",
  "tidymodels",
  "survival",
  "survminer",
  "glmnet",
  "ranger",
  "xgboost",
  "pROC",
  "yardstick",
  "vip"
)

installed <- rownames(installed.packages())
missing <- setdiff(packages, installed)

if (length(missing) > 0) {
  install.packages(missing)
}

library(tidyverse)
library(tidymodels)
library(survival)

sessionInfo()