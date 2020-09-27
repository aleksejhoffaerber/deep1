source("packages.R")

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

digits.test <- read.csv("data/test.csv")

i <- 1:5000
digits.x <- digits.train[i, -1]
digits.y <- digits.train[i, 1]

c1 <- h2o.init(max_mem_size = "3G", nthreads = 2)

cl <- makeCluster(7)
clusterEvalQ(cl, {
  source("packages.R") # load packages into h2o cluster
})

registerDoSNOW(cl)

set.seed(1234)

digits.decay.m1 <- lapply(c(100, 150), function(its) {
  caret::train(digits.x, digits.y,
        method = "nnet",
        tuneGrid = expand.grid(
          .size = c(10),
          .decay = c(0, .1)), # weight decay (L2 penalty)
        trControl = trainControl(method = "cv", number = 5, repeats = 1),
        MaxNWts = 10000,
        maxit = its)
})

digits.decay.m1[[1]] # comparing reg. and non.reg model or 100 iterations
digits.decay.m1[[2]] # 150 iterations, reg. performed way better

digits.decay.m1[[1]]

# cannot test OOS performance, because no labels in test.csv
digits.pd1 <- predict(digits.decay.m1[[1]], newdata = digits.test) # automatically uses the best model = besttune (w/0 reg)
m1.pred <- ggplot(as.data.frame(table(digits.pd1)), 
                  aes(x = digits.pd1, y = Freq)) +
  geom_bar(stat = "identity") +
  ggtitle(paste("100 Iterations Model,","Training Accuracy:", round(digits.decay.m1[[1]]$results$Accuracy[1], digits = 3))) +
  scale_y_continuous(limits = c(0,5500), 
                     breaks = c(1000, 2000, 3000, 4000, 5000)) + 
  theme_minimal()
  
  
digits.pd2 <- predict(digits.decay.m1[[2]], newdata = digits.test)
m2.pred <- ggplot(as.data.frame(table(digits.pd2)), 
                  aes(x = digits.pd2, y = Freq)) +
  geom_bar(stat = "identity") +
  ggtitle(paste("150 Iterations Model,","Training Accuracy:", round(digits.decay.m1[[2]]$results$Accuracy[2], digits = 3))) +
  scale_y_continuous(limits = c(0,5500), 
                     breaks = c(1000, 2000, 3000, 4000, 5000)) + 
  theme_minimal()

m1.pred / m2.pred

# DROPOUT POWERED DNN (no L1/L2 regularization)

nn.models <- foreach(k = 1:4, .combine = "c") %dopar% {
  set.seed(1234)
    list(nn.train( # grid to train multiple models from "deepnet"
      x = as.matrix(digits.x),
      y = model.matrix(~ 0 + digits.y), # 0 to drop the intercept (that is believed to be a factor but isnt)
      hidden = c(40, 80, 40, 80)[k],
      activationfun = "tanh", 
      learningrate = 0.8,
      momentum = 0.5, # momentum for gradient descent, 0.5 is the standard
      numepochs = 150, # number of iteration samples
      output = "softmax", # outputs become normalized on 0 to 1 --> probabilities
      hidden_dropout = c(0, 0, .5, .5)[k],
      visible_dropout = c(0, 0, .2, .2)[k]
    ))
}

nn.yhat <- lapply(nn.models, function(obj) {
  encodeClassLabels(nn.predict(obj, as.matrix(digits.x))) # most DNN need data in matrix form
})

perf.train <- do.call(cbind, lapply(nn.yhat, function(yhat) {
  caret::confusionMatrix(xtabs(~ I(yhat - 1) + digits.y))$overall # xtabs to create factor-based table
}))

colnames(perf.train) <- c("N40", "N80", "N40_Reg", "N80_Reg")
options(digits = 4)
perf.train

# OOS performance

i2 <- 5001:10000
test.x <- digits.train[i2,-1]
test.y <- digits.train[i2,1]

nn.yhat.test <- lapply(nn.models, function(obj) {
  encodeClassLabels(nn.predict(obj, as.matrix(test.x))) # most DNN need data in matrix form
})

perf.test <- do.call(cbind, lapply(nn.yhat.test, function(yhat) {
  caret::confusionMatrix(xtabs(~ I(yhat - 1) + test.y))$overall # xtabs to create factor-based table
}))

colnames(perf.test) <- c("N40", "N80", "N40_Reg", "N80_Reg")
options(digits = 4)
perf.test

# NORMAL TRAIN/TEST 

split <- 0.7

train <- digits.train %>% 
  slice(1:as.integer(nrow(digits.train) * split))

test <- digits.train %>% 
  slice(
    (as.integer(nrow(digits.train) * split)+1):(as.integer(nrow(digits.train+1))))

train.x <- train[,-1]
train.y <- train[,1]

test.x <- test[,-1]
test.y <- test[,1]

sapply(list(train.x, train.y, test.x, test.y), dim) # check correctness



nn.models.tr <- foreach(k = 1:4, .combine = "c") %dopar% {
  set.seed(123)
  list(nn.train(
    x = as.matrix(train.x),
    y = model.matrix(~ 0 + train.y),
    hidden = c(40, 80, 40, 80)[k],
    activationfun = "tanh",
    learningrate = 0.8,
    momentum = 0.5,
    numepochs = 100,
    output = "softmax", 
    hidden_dropout = c(0,0,.5,.5)[k],
    visible_dropout = c(0,0,.2,.2)[k]
  ))
}

yhat.tr <- lapply(nn.models.tr, function(o) {
  encodeClassLabels(nn.predict(o, as.matrix(test.x)))
})

perf.test.tr <- do.call(cbind, lapply(yhat.tr, function(u) # cbind because $overall will be a vector
  {
  caret::confusionMatrix(xtabs(~I(u - 1) + test.y))$overall # basic important statistics
  }
))

colnames(perf.test.tr) <- c("N40", "N80", "N40_Reg", "N80_Reg")
perf.test.tr



