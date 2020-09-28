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

# MODEL TRAINING -----
# 1) base model (AE, 50 Hidden Neurons, 20 Epochs, no Reg)
# 2a) AE, 100 HN, 20 E, no reg
# 2b) AE, 100 HN, 20 E, .5 beta (sparsity regularization)
# 2c) AE, 100 HN, 20 E, .2 input dropout

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

m2a <- h2o.deeplearning(
  x = xnames,
  training_frame= h2odigits.train,
  validation_frame = h2odigits.test,
  activation = "TanhWithDropout",
  autoencoder = TRUE,
  hidden = c(100),
  epochs = 20,
  sparsity_beta = 0,
  input_dropout_ratio = 0,
  hidden_dropout_ratios = c(0),
  l1 = 0,
  l2 = 0
)
m2b <- h2o.deeplearning(
  x = xnames,
  training_frame= h2odigits.train,
  validation_frame = h2odigits.test,
  activation = "TanhWithDropout",
  autoencoder = TRUE,
  hidden = c(100),
  epochs = 20,
  sparsity_beta = .5, # important to apply grid_search here
  # relative importance of sparsity loss (imagine the construction of numbers, or: loss of details)
  input_dropout_ratio = 0,
  hidden_dropout_ratios = c(0),
  l1 = 0,
  l2 = 0
)

m2c <- h2o.deeplearning(
  x = xnames,
  training_frame= h2odigits.train,
  validation_frame = h2odigits.test,
  activation = "TanhWithDropout",
  autoencoder = TRUE,
  hidden = c(100),
  epochs = 20,
  sparsity_beta = 0,
  input_dropout_ratio = .2,
  hidden_dropout_ratios = c(0),
  l1 = 0,
  l2 = 0
)

error1 <- as.data.frame(h2o.anomaly(m1, h2odigits.train)) # detects anomalies in model if autoencoders were used
error2a <- as.data.frame(h2o.anomaly(m2a, h2odigits.train)) # based on percentiles and probabilities
error2b <- as.data.frame(h2o.anomaly(m2b, h2odigits.train))
error2c <- as.data.frame(h2o.anomaly(m2c, h2odigits.train))

error <- as.data.table(rbind(
  cbind.data.frame(Model = 1, error1),
  cbind.data.frame(Model = "2a", error2a),
  cbind.data.frame(Model = "2b", error2b),
  cbind.data.frame(Model = "2c", error2c)))


percentile <- error[, .( # dot to keep column name
  Percentile = quantile(Reconstruction.MSE, probs = .99)
), by = Model] # create new df, transformed .99 percentiles of the MSE3

p <- ggplot(error, aes(Reconstruction.MSE)) +
  geom_histogram(binwidth = .001, fill = "grey50") +
  geom_vline(aes(xintercept = Percentile), data = percentile, linetype = 2) +
  theme_bw() +
  facet_wrap(~Model)
print(p)

error.cr <- cbind(error1, error2a, error2b, error2c) 
colnames(error.cr) <- c("M1", "M2a", "M2b", "M2c")
plot(error.cr)

# ADVANCED DNN MODEL ----

m3 <- h2o.deeplearning(
  x = xnames,
  training_frame = h2odigits.train,
  validation_frame = h2odigits.test,
  activation = "TanhWithDropout",
  autoencoder = T,
  hidden = c(100,10),
  epochs = 30,
  sparsity_beta = 0,
  input_dropout_ratio = 0,
  hidden_dropout_ratios = c(0,0),
  l1 = 0,
  l2 = 0
)

error3 <- as.data.frame(h2o.anomaly(m3, h2odigits.train))
percentile3 <- quantile(error3$Reconstruction.MSE, probs = .99)

f3 <- as.data.frame(h2o.deepfeatures(m3, h2odigits.train, 2)) # 2nd layer
f3$label <- digits.train$label[i]
f4 <- melt(f3, id.vars = "label")


colnames(f3) <- c(rep(1:10), "label")

f3 %>% group_by(label) %>% 
  summarise_all(mean) %>% 
  pivot_longer(c(2:11)) %>% 
  as.data.frame() %>% 
  mutate(name = I(as.integer(name) -1)) -> f3c 
  # important to keep name otherwise no line possible 


f3c %>% 
  ggplot(aes(x = name, y = value, 
             col = label, linetype = label)) +
  geom_line(lwd = 2) +
  scale_x_continuous("Deep Features eq. to 2nd Hidden Layer", 
                     breaks = c(1:10), n.breaks = 10) +
  labs(title = "Probability Assignment",
       subtitle = "Despite structure, 2nd layer did not result in a clear classification seperation") +
  theme_minimal() +
  theme(legend.position = "bottom", legend.key.width = unit(1, "cm"))
