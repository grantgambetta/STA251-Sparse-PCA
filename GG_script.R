library(elasticnet)
library(sparsepca)
library(RCurl)
library(dplyr)
library(ggplot2)

winequality <- read.csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep = ";")
head(winequality)

