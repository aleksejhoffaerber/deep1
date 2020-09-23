
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

# ------ FIRST MODEL TRAINING
# Case: We receive a df that says why label/number (Y variable) is predicted based on whether there are pixels in one of the 784 px areas
tr <- read_csv("data/train.csv")
dim(tr)
colnames(tr)
# skim(tr)              # extremely sparse

tr$label <- as.factor(tr$label)
i <- 1:5000

dig.tr <- tr[i, -1]   # w/o Y variable 
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

dig.pd2 <- predict(dig.m1)

barplot(table(dig.pd2))
# low accuracy because distribution is too differing

caret::confusionMatrix(xtabs(~dig.pd2 + dig.ts))
# for explanations, see PDF
# Kappa is more reliable especially for imbalanced datasets
# also in this case: lower, but the data is not imbalanced (9.8 to 11.6% prevalance)

# Kappa or Cohens Kappa is like classification accuracy, 
# except that it is normalized at the baseline of random chance 
# on your dataset. It is a more useful measure to use on problems 
# that have an imbalance in the classes (e.g. 70-30 split for classes 
# 0 and 1 and you can achieve 70% accuracy by predicting all instances 
# are for class 0). Learn more about Kappa here.

# ------ BUILDING MORE ADVANCED MODELS 

dig.m2 <- caret::train(x = dig.tr, y = dig.ts,
                       method = "nnet",
                       tuneGrid = expand.grid(
                         .size = c(10),
                         .decay = 0.1),
                       trControl = trainControl(method = "none"),
                       MaxNWts = 50000,
                       maxit = 100)


dig.pd2 <- predict(dig.m2)
barplot(table(dig.pd2))
caret::confusionMatrix(xtabs(~dig.pd2 + dig.ts))

dig.m3 <- mlp(as.matrix(dig.tr),
              decodeClassLabels(dig.ts),
              size = 10,
              learnFunc = "Rprop",
              shufflePatterns = F,
              maxit = 60)

dig.pd3 <- fitted.values(dig.m3)
dig.pd3 <- encodeClassLabels(dig.pd3)
barplot(table(dig.pd3))

caret::confusionMatrix(xtabs(~I(dig.pd3 -1) + dig.ts))
# reminder from Predictive Analytics: using I for operations (to the power of, subtracttion)

# str(dig.pd3 - 1)
# class(I(dig.pd3))

# model did run through faster and with more accurate results
# Q: coincidence or based on the better algorithm (esp. the learning function)


