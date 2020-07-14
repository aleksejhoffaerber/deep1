library(RCurl)
library(jsonlite)
library(caret)
library(e1071)
library(dplyr)
library(readr)
library(skimr)

library(statmod)
library(MASS)

# install.packages(c("nnet","neuralnet", "RSNNS", "h2o", "deepnet", "darch"))
library(nnet)         # only one hidden layer
library(neuralnet)    # only one hidden layer, advanced training possibs
library(RSNNS)        # variety of NN models

library(deepnet)      # DBN and RBM capabilities 
library(darch)        # same, but pure R code (slow)
library(h2o)          # java-based, fast

# ------ INITIALIZING H2O

c1 <- h2o.init(max_mem_size = "3G", nthreads = 2)
# fail because Java version too recent (works with 13 though)

# ------ FIRST MODEL TRAINING
tr <- read_csv("data/train.csv")
dim(tr)
colnames(tr)
# skim(tr)              # extremely sparse

tr$label <- as.factor(tr$label)
i <- 1:5000

dig.tr <- tr[i, -1]   # w/o label as predictor
dig.ts <- as.vector(t(tr[i, 1])) # needs to be a sole vector

barplot(table(dig.ts)) # table, because factor variable, must be in numeric, summarized format
                       # equal distribution, no need to alter the modelling approach

set.seed(1234)

dig.m1 <- caret::train(x = dig.tr, y = dig.ts, # train from the caret package, functions as a wrapper
                       method = "nnet",
                       tuneGrid = expand.grid( # grid for hyperparameter tuning
                         .size = c(5),  # 5 hidden neurons
                         .decay = 0.1), # learning rate
                       trControl = trainControl(method = "none"),
                       MaxNWts = 10000, # max weights
                       maxit = 100) # max iterations

# takeaway: read documentation, mb methods expect something very special that
# is not written in the warning (factor/numeric value may be an indicator for vectors)

dig.pd <- predict(dig.m1)

barplot(table(dig.pd))
# low accuracy because distribution is too differing
