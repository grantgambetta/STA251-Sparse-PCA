library(elasticnet)
library(sparsepca)
library(RCurl)
library(dplyr)
library(ggplot2)
library(readxl)
library(knitr)
library(stargazer)
library(gridExtra)
options(scipen = 999)

# https://archive.ics.uci.edu/ml/datasets/wine+quality
winequality <- read.csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep = ";")
head(winequality)

X <- winequality %>% select(-quality)
print(dim(X))
print(colnames(X))
print(summary(X))
stargazer(X)

#######################################  WINE DATASET ####################################### 
############################## PCA ##############################
pca <- prcomp(X, center = TRUE, scale. = TRUE)
summary(pca)
stargazer(pca$rotation[, 1:6])
print(pca$rotation[, 1:6]) # loadings matrix

explained_var <- pca$sdev^2 / sum(pca$sdev^2)
explained_var_df <- data.frame(PC = paste0("PC", 1:11),
                               explained_variance = explained_var)
explained_var_df$PC <- factor(explained_var_df$PC, levels = explained_var_df$PC)
explained_var_df$cumulative_exp_var <- cumsum(explained_var)

ggplot(data = explained_var_df, aes(x = PC, y = explained_var, group = 1)) +
  geom_point(size = 3) +
  geom_line() +
  xlab('Principal Component') + 
  ylab('% of Explained Variance') + 
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

ggplot(data = explained_df_spca, aes(x = PC, y = cumulative_exp_var, group = 1)) +
  geom_point(size = 3) +
  geom_line() +
  xlab('Principal Component') + 
  ylab('Cumulative Explained Variance') + 
  ggtitle('Scree Plot - SPCA on Wine Data')

############################## SPARSE PCA USING SPARSEPCA PACKAGE ##############################
k = 6
sparse_pca2 <- sparsepca::spca(X, k = k, alpha = 0.001, beta = 1e-4, center = TRUE, scale = TRUE)
summary(sparse_pca2)
sparse_loadings <- sparse_pca2$loadings
rownames(sparse_loadings) <- colnames(X)
colnames(sparse_loadings) <- c('PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6')
print(sparse_loadings)
stargazer(sparse_loadings)

alphas <- seq(0.001, 0.05, by = 0.001)
var_result <- c()
for (alpha in alphas) {
  res <- sparsepca::spca(X, k = 6, alpha = alpha, beta = 1e-4, center = TRUE, scale = TRUE)
  var <- summary(res)[4, 6]
  var_result <- append(var_result, var)
}

result_spca_wine <- as.data.frame(cbind(alphas, var_result))
colnames(result_spca_wine) <- c('alpha', 'cev')
print(result_spca_wine)

ggplot(data = result_spca_wine, aes(x = alphas, y = cev)) +
  geom_point(size = 1.3) +
  geom_line() +
  xlab('Alpha') + 
  ylab('Cumulative Explained Variance') + 
  ggtitle('Cumulative Explained Variance vs Alpha - SPCA on the Wine Data')

k = 6
sparse_pca2_best <- sparsepca::spca(X, k = k, alpha = 0.004, beta = 1e-4, center = TRUE, scale = TRUE)
summary(sparse_pca2_best)

sparse_loadings_best <- sparse_pca2_best$loadings
rownames(sparse_loadings_best) <- colnames(X)
colnames(sparse_loadings_best) <- c('PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6')
print(sparse_loadings_best)
stargazer(sparse_loadings_best)

#######################################  LSVT VOICE DATASET ####################################### 

# https://archive.ics.uci.edu/ml/datasets/LSVT+Voice+Rehabilitation

voice_data <- readxl::read_excel('LSVT_voice_rehabilitation.xlsx')
print(head(voice_data))
print(dim(voice_data))

pca <- prcomp(voice_data, center = TRUE, scale. = TRUE)
summary(pca)

n_components <- 15
explained_var <- pca$sdev^2 / sum(pca$sdev^2)
explained_var_df <- data.frame(PC = paste0("PC", 1:n_components),
                               explained_variance = explained_var[1:n_components])
explained_var_df$PC <- factor(explained_var_df$PC, levels = explained_var_df$PC)
explained_var_df$cumulative_exp_var <- cumsum(explained_var[1:n_components])

ggplot(data = explained_var_df, aes(x = seq(1, n_components, by=1), y = cumulative_exp_var, group = 1)) +
  geom_line() +
  xlab('Principal Components') + 
  ylab('Cumulative Explained Variance') + 
  ggtitle('Scree Plot - PCA on LSVT Voice Data')

alphas <- c(0.00005, 0.0001, 0.0002, 0.0003, 0.0005, 0.0008, 0.002, 0.004, 0.006, 0.008, 0.01)
var_result <- c()
p_sparse_df <- data.frame(matrix(ncol = 15, nrow = 0))
colnames(p_sparse_df) <- c('PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 
                           'PC8', 'PC9', 'PC10', 'PC11', 'PC12', 'PC13', 'PC14', 'PC15')
for (alpha in alphas) {
  res <- sparsepca::spca(voice_data, k = 15, alpha = alpha, beta = 1e-4, center = TRUE, scale = TRUE)
  
  loadings <- as.data.frame(res$loadings)
  percent_sparse <- as.numeric(colSums(loadings == 0)/nrow(loadings) * 100)
  p_sparse_df <- structure(rbind(p_sparse_df, percent_sparse), .Names = names(p_sparse_df))
  
  var <- summary(res)[4, 15]
  var_result <- append(var_result, var)
}

result_df <- as.data.frame(cbind(alphas, var_result, p_sparse_df))
colnames(result_df)[1:2] <- c('alpha', 'cev')
result_df$avg_sparsity <- apply(result_df[, 3:length(colnames(result_df))], 1, mean)
print(result_df)

p1 <- ggplot(data = result_df, aes(x = alphas, y = var_result)) +
  geom_line() +
  geom_point() +
  geom_text(
    label=paste('a=', result_df$alpha), 
    nudge_x = 0.0007, nudge_y = 0.002, 
    check_overlap = T
  ) +
  theme(axis.text = element_text(size = 13), 
        axis.title = element_text(size = 15),
        plot.title = element_text(size = 17)) +
  xlab('Alpha') + 
  scale_x_continuous(breaks=seq(min(result_df$alpha), max(result_df$alpha), 0.003)) +
  ylab('Cumulative Explained Variance') + 
  ggtitle('Cumulative Explained Variance vs Alpha - SPCA on LSVT Data with 15 PCs')

p2 <- ggplot(data = result_df, aes(x = alphas, y = avg_sparsity)) +
  geom_line() +
  geom_point() +
  geom_text(
    label=paste('a=', result_df$alpha), 
    nudge_x = 0.0006, nudge_y = -1, 
    check_overlap = T
  ) +
  theme(axis.text = element_text(size = 13), 
        axis.title = element_text(size = 15),
        plot.title = element_text(size = 17)) +
  xlab('Alpha') + 
  scale_x_continuous(breaks=seq(min(result_df$alpha), max(result_df$alpha), 0.003)) +
  ylab('Sparsity %') + 
  ggtitle('Sparsity % vs Alpha - SPCA on LSVT Data with 15 PCs')

grid.arrange(p1, p2, nrow=2)

# best sparse pca after choosing alpha
best_spca <- sparsepca::spca(voice_data, k = 15, alpha = 0.0002, beta = 1e-4, center = TRUE, scale = TRUE)
summary(best_spca)
loadings <- as.data.frame(best_spca$loadings)
print(loadings)

