source("packages.R")

digits.train <- read.csv("data/train.csv") %>% 
  mutate(label = as.factor(label))


c2 <- h2o.init(max_mem_size = "20G", nthreads = 10)
h2odigits <- as.h2o(digits.train, # data upload
                    destination_frame = "h2odigits")

i <- 1:20000
h2odigits.train <- h2odigits[i,-1]

it <- 20001:30000
h2odigits.test <- h2odigits[it,-1]
xnames <- colnames(h2odigits.train)

m1 <- h2o.deeplearning(
  x = xnames,
  training_frame = h2odigits.train,
  validation_frame = h2odigits.test,
  activation = "TanhWithDropout",
  autoencoder = TRUE,
  hidden = c(50),
  epochs = 20,
  sparsity_beta = 0,
  input_dropout_ratio = 0,# first run
  hidden_dropout_ratios = 0,
  l1 = 0,
  l2 = 0
)
