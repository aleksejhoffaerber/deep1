# Overfitting Prevention

library(RCurl)
library(jsonlite)
library(caret)
library(e1071)
library(dplyr)
library(readr)
library(skimr)

library(statmod)
library(MASS)

library(nnet)         # only one hidden layer
library(neuralnet)    # only one hidden layer, advanced training possibs
library(RSNNS)        # variety of NN models

library(deepnet)      # DBN and RBM capabilities 
# library(darch)        # same, but pure R code (slow)
library(h2o)          # java-based, fast
library(parallel)
library(doSNOW)

library(glmnet)

# ----- IMPORTANT LESSONS 
# 1) different regularizations can be used in different layers e.g. to include more parameters



set.seed(123)

X <- mvrnorm(n = 200, mu = c(0, 0, 0, 0, 0), # multivariate normal dist // mu = means, sigma = covariance
             Sigma = matrix(c(
               1, .9999, .99, .99, .10,
               .9999, 1, .99, .99, .10,
               .99, .99, 1, .99, .10,
               .99, .99, .99, 1, .10,
               .10, .10, .10, .10, 1
             ), ncol = 5))

Y <- rnorm(200, 3 + X %*% matrix(c(1, 1, 1, 1, 0)), .5)

m.ols <- lm(Y[1:100] ~ X[1:100])
m.lasso.cv <- cv.glmnet(X[1:100,], Y[1:100], alpha = 1) # alpha determines the L1 penalty

plot(m.lasso.cv)

# 2) L2 = weight decay, penalty is applied at every update (so a multiplicative penalty)

m.ridge.cv <- cv.glmnet(X[1:100,], Y[1:100], alpha = 0)
plot(m.ridge.cv)

# ----- MODELLING PART

digits.train <- read.csv("data/train.csv") %>% 
  mutate(label = as.factor(label))

i <- 1:5000
digits.x <- digits.train[i, -1]
digits.y <- digits.train[i, 1]

c1 <- h2o.init(max_mem_size = "3G", nthreads = 2)

cl <- makeCluster(4)
clusterEvalQ(cl, {
  source("packages.R")
})

registerDoSNOW(cl)

set.seed(1234)

digits.decay.m1 <- lapply(c(100, 150), function(its) {
  caret::train(digits.x, digits.y,
        method = "nnet",
        tuneGrid = expand.grid(
          .size = c(10),
          .decay = c(0, .1)),
        trControl = trainControl(method = "cv", number = 5, repeats = 1),
        MaxNWts = 10000,
        maxit = its)
})


