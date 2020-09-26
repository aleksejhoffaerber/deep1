# ---- User Data Tracking

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
library(darch)        # same, but pure R code (slow)
library(h2o)          # java-based, fast


# install.packages(c("parallel", "foreach", "doSNOW"))

library(parallel)
library(foreach)
library(doSNOW)

# ---- DATA LOADING

us.train.x <- read.table("data/UCI HAR Dataset/train/X_train.txt")
us.train.y <- read.table("data/UCI HAR Dataset/train/y_train.txt")[[1]]
# [[1]] needed so that we get a sole vector (as we read only one col)
# important for the expected data model of RSNNS

us.test.x <- read.table("data/UCI HAR Dataset/test/X_test.txt")
us.test.y <- read.table("data/UCI HAR Dataset/test/y_test.txt")[[1]]

us.labels <- read.table("data/UCI HAR Dataset/activity_labels.txt")
barplot(table(us.train.y))

# MODEL AND CLUSTER SET-UP

# tuning param list
tuning <- list(
  size = c(40,20,20,50,50), # hidden neurons
  maxit = c(60,100,100,100,100), # iterations
  shuffle = c(FALSE,FALSE,TRUE,FALSE,FALSE), # changes the order of rows for the selected variables
  params = list(FALSE, FALSE, FALSE, FALSE, c(0.1, 20, 3)) # 0.1 indicates the weight decay
)

# cluster setup
c2 <- makeCluster(4)
clusterEvalQ(c2, {
  library(RSNNS)
})

clusterExport(c2,
              c("tuning", "us.train.x", "us.train.y",
                "us.test.x", "us.test.y")
)
registerDoSNOW(c2)

# MODEL ARCHITECTURE
# build models
us.models <- foreach(i = 1:5, .combine = "c") %dopar% {
  if (tuning$params[[i]][1]) {
    set.seed(1234)
    list(Model = mlp(
      as.matrix(us.train.x),
      decodeClassLabels(us.train.y),
      size = tuning$size[[i]],
      learnFunc = "Rprop",
      shufflePatterns = tuning$shuffle[[i]],
      learnFuncParams = tuning$params[[i]],
      maxit = tuning$maxit[[i]])
    )
  } else {
    set.seed(1234)
    list(Model = mlp(
      as.matrix(us.train.x),
      decodeClassLabels(us.train.y),
      size = tuning$size[[i]],
      learnFunc = "Rprop",
      shufflePatterns = tuning$shuffle[[i]],
      maxit = tuning$maxit[[i]])
    )
  }
}

clusterExport(c2, "us.models")
us.yhat <- foreach(i = 1:5, .combine = "c") %dopar% {
  list(list(
    Insample = encodeClassLabels(fitted.values(us.models[[i]])), # Insample as list name, fitted.values to access insample fits
    Outsample = encodeClassLabels(predict(us.models[[i]], # predict for actual test predicitons
                                          newdata = as.matrix(us.test.x))) 
  ))
}

# INSAMPLE PERFORMANCE
us.insample <- cbind(Y = us.train.y,
                     do.call(cbind.data.frame, lapply(us.yhat, `[[`, "Insample"))) 
# `[[` needed to access the specifically named list

colnames(us.insample) <- c("Y", paste0("Yhat", 1:5))


# difference between substitute and quote
# expr <- substitute(x + y, list(x = 1))
# print(expr) # 1 + y
# eval(expr, list(y = 2)) # 3


performance.insample <- do.call(rbind, lapply(1:5, function(i) { # rbind because I want to have the models below each other 
  us.dat <- us.insample[us.insample[,paste0("Yhat", i)] != 0, ] # needed to throw out the zeros as those are uncertain predictions
  us.dat$Y <- factor(us.dat$Y, levels = 1:6) # to factor
  us.dat[, paste0("Yhat", i)] <- factor(us.dat[, paste0("Yhat", i)], levels = 1:6) # factor
  f <- substitute(~ Y + x, list(x = as.name(paste0("Yhat", i)))) 
  # create the expression "~Y + Yhat1 .... needed for the confusionMatrix later
  # as.name needed for the substitute function
  
  res <- caret::confusionMatrix(xtabs(f, data = us.dat)) 
  
  cbind(Size = tuning$size[[i]],
        Maxit = tuning$maxit[[i]],
        Shuffle = tuning$shuffle[[i]],
        as.data.frame(t(res$overall[c("AccuracyNull", "Accuracy", "AccuracyLower", "AccuracyUpper")]))) 
  # access acc information with [c()], transpose
}))



# OUTSAMPLE PERFORMANCE

us.outsample <- cbind(Y = us.test.y,
                      do.call(cbind.data.frame, lapply(us.yhat, `[[`, "Outsample")))
colnames(us.outsample) <- c("Y", paste0("Yhat", 1:5))


performance.outsample <- do.call(rbind, lapply(1:5, function(i) { 
  us.dat <- us.outsample[us.outsample[,paste0("Yhat", i)] != 0, ] 
  us.dat$Y <- factor(us.dat$Y, levels = 1:6) 
  us.dat[, paste0("Yhat", i)] <- factor(us.dat[, paste0("Yhat", i)], levels = 1:6) 
  f <- substitute(~ Y + x, list(x = as.name(paste0("Yhat", i)))) 

    
  res <- caret::confusionMatrix(xtabs(f, data = us.dat)) 
  
  cbind(Size = tuning$size[[i]],
        Maxit = tuning$maxit[[i]],
        Shuffle = tuning$shuffle[[i]],
        as.data.frame(t(res$overall[c("AccuracyNull", "Accuracy", "AccuracyLower", "AccuracyUpper")]))) 
}))

# Next Steps:

# 1) compare to RF and XGBoost Model Performance (worth it?)
# 2) look at other UCI datasets to predict user behavior 
# 3) introduce shap values 
# 4) combine features with PCA - NN approach

