library(RCurl)
library(jsonlite)
library(caret)
library(e1071)
library(dplyr)
library(tidyr)
library(readr)
library(skimr)
library(patchwork) # easy beside plots

library(statmod)
library(MASS)
library(parallel)

# install.packages(c("nnet","neuralnet", "RSNNS", "h2o", "deepnet", "darch"))
library(nnet)         # only one hidden layer
library(neuralnet)    # only one hidden layer, advanced training possibs
library(RSNNS)        # variety of NN models

library(deepnet)      # DBN and RBM capabilities 
# library(darch)        # same, but pure R code (slow)
library(h2o)          # java-based, fast
library(doSNOW)

library(glmnet)
