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

digits.decay.m1[[1]]$modelInfo

# cannot test OOS performance, because no labels in test.csv
digits.pd1 <- predict(digits.decay.m1[[1]], newdata = digits.test)
m1.pred <- ggplot(as.data.frame(table(digits.pd1)), aes(x = digits.pd1, y = Freq)) +
  geom_bar(stat = "identity") +
  ggtitle("100 Iterations Model") +
  scale_y_continuous(limits = c(0,5500), breaks = c(1000, 2000, 3000, 4000, 5000)) + 
  theme_minimal()
  
  
digits.pd2 <- predict(digits.decay.m1[[2]], newdata = digits.test)
m2.pred <- ggplot(as.data.frame(table(digits.pd2)), aes(x = digits.pd2, y = Freq)) +
  geom_bar(stat = "identity") +
  ggtitle("150 Iterations Model") +
  scale_y_continuous(limits = c(0,5500), breaks = c(1000, 2000, 3000, 4000, 5000)) + 
  theme_minimal()

m1.pred / m2.pred


