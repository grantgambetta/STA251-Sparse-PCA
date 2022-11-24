library(elasticnet)
library(sparsepca)
library(RCurl)
library(dplyr)
library(ggplot2)
library(stargazer)
options(scipen = 999)

winequality <- read.csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep = ";")
head(winequality)

X <- winequality %>% select(-quality)
#X_scaled <- X %>% mutate_all(~(scale(.) %>% as.vector))

############################## PCA ##############################
pca <- prcomp(X, center = TRUE, scale. = TRUE)
summary(pca)
print(pca$rotation[, 1:5]) # loadings matrix

explained_var <- pca$sdev^2 / sum(pca$sdev^2)
explained_var_df <- data.frame(PC = paste0("PC", 1:11),
                               explained_variance = explained_var)
explained_var_df$PC <- factor(explained_var_df$PC, levels = explained_var_df$PC)
explained_var_df$cumulative_exp_var <- cumsum(explained_var)

ggplot(data = explained_var_df, aes(x = PC, y = explained_var, group = 1)) +
  geom_point(size = 3) +
  geom_line() +
  xlab('Principal Component') + 
  ylab('Explained Variance') + 
  ggtitle('Scree Plot - PCA on Wine Data')

ggplot(data = explained_var_df, aes(x = as.factor(PC), y = cumulative_exp_var, group = 1)) +
  geom_point(size = 3) +
  geom_line() +
  xlab('Principal Component') + 
  ylab('Cumulative Explained Variance') + 
  ggtitle('Scree Plot - PCA on Wine Data')

############################## SPARSE PCA USING ELASTICNET PACKAGE ##############################
X_scaled <- X %>% mutate_all(~(scale(.) %>% as.vector))

K = 6
sparse_pca <- elasticnet::spca(X_scaled, K = K, lambda = 1e-6, para = c(0.1, 0.2, 0.3, 0.3, 0.3, 0.5), type = 'predictor')
print(sparse_pca$loadings)
print(sparse_pca$pev)

explained_df_spca <- data.frame(PC = paste0("PC", seq(1, K, by=1)),
                               explained_variance = sparse_pca$pev)
explained_df_spca$PC <- factor(explained_df_spca$PC, levels = explained_df_spca$PC)
explained_df_spca$cumulative_exp_var <- cumsum(sparse_pca$pev)

ggplot(data = explained_df_spca, aes(x = PC, y = explained_variance, group = 1)) +
  geom_point(size = 3) +
  geom_line() +
  xlab('Principal Component') + 
  ylab('Explained Variance') + 
  ggtitle('Scree Plot - SPCA on Wine Data')

ggplot(data = explained_df_spca, aes(x = as.factor(PC), y = cumulative_exp_var, group = 1)) +
  geom_point(size = 3) +
  geom_line() +
  xlab('Principal Component') + 
  ylab('Cumulative Explained Variance') + 
  ggtitle('Scree Plot - SPCA on Wine Data')

############################## SPARSE PCA USING SPARSEPCA PACKAGE ##############################
k = 6
sparse_pca2 <- sparsepca::spca(X, k = k, alpha = 0.001, beta = 1e-4, center = TRUE, scale = TRUE)
summary(sparse_pca2)
print(sparse_pca2$loadings)

alphas <- seq(0.001, 0.05, by = 0.001)
var_result <- c()
for (alpha in alphas) {
  res <- sparsepca::spca(X, k = 6, alpha = alpha, beta = 1e-4, center = TRUE, scale = TRUE)
  var <- summary(res)[4, 6]
  var_result <- append(var_result, var)
}

result <- as.data.frame(cbind(alphas, var_result))
print(result)
