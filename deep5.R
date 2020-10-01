source("packages.R")
options(width = 70, digits = 3)

c1 <- h2o.init(max_mem_size = "14G",
               nthreads = 6)


# NEW DATA:
download.file("http://archive.ics.uci.edu/ml/machine-learning-databases/00203/
YearPredictionMSD.txt.zip", destfile = "YearPredictionMSD.txt.zip")

unzip("YearPredictionMSD.txt.zip")

d <- fread("YearPredictionMSD.txt", sep = ",") # speedup

ggplot(d[,.(V1)], aes(V1)) +
  geom_histogram(binwidth = 1) +
  xlab("Year of Release")

d %>% 
  group_by(V1) %>% 
  add_count() %>% 
  summarise(mean = mean(n)) -> da

# Cleaning for "out of quantile" observations
d %<>% filter(V1 >= quantile(d$V1, probs = c(0.005, 0.995))[1] &
                V1 <= quantile(d$V1, probs = c(0.005, 0.995))[2]) 


# Test/train
split <- 0.9

dtr <- d %>% slice(1:(as.integer(nrow(d)) * split))
dte <- d %>% slice((as.integer(nrow(d) * split + 1)):(as.integer(nrow(d) + 1)))

# h2o clustering
h2o.tr <- as.h2o(dtr, destination_frame = "h2o.tr")
h2o.te <- as.h2o(dte, destination_frame = "h2o.te")

# model fitting
summary(m0 <- lm(V1 ~., data = dtr)) # 24% in variance
cor(dte$V1,
    predict(m0, newdata = dte))^2 # 23% variance in test set


m1 <- h2o.deeplearning(
  x = colnames(d)[-1],
  y = "V1",
  training_frame= h2o.tr,
  validation_frame = h2o.te, # for instant test performance output
  activation = "RectifierWithDropout",
  hidden = c(50),
  epochs = 100,
  input_dropout_ratio = 0,
  hidden_dropout_ratios = c(0),
  score_training_samples = 0,
  score_validation_samples = 0, # performance scoring on full data set
  diagnostics = TRUE,
  export_weights_and_biases = TRUE,
  variable_importances = TRUE
)

h2o.saveModel(
  object = m1,
  path = paste0(getwd(),"\\m1"),
  force = T)

# residuals
yhat <- as.data.frame(h2o.predict(m1, h2o.tr))
yhat <- cbind(as.data.frame(h2o.tr[["V1"]]), yhat)

p.resid <- ggplot(yhat, aes(factor(V1), predict - V1)) +
  geom_boxplot(fill = "cornflowerblue",
               outlier.colour = "darkred",
               outlier.size = 1) +
  geom_hline(yintercept = 0) +
  theme_minimal() +
  theme(axis.text.x = element_text(
    angle = 90, vjust = 1, hjust = 1)) +
  labs(title = "Prediction Residuals",
       subtitle = "Model results indicate bad prediction fits in the first years, because of skewed data") +
  xlab("Year of Release") +
  ylab("Difference in Predicted Years of Release")

print(p.resid)

# variable importance
imp <- as.data.frame(h2o.varimp(m1))

p.imp <- imp %>% 
  ggplot(aes(x = factor(variable, levels = variable), y = percentage)) +
  geom_point(color = "cornflowerblue") +
  geom_hline(yintercept=0.01) +
  geom_hline(yintercept=0.02) +
  theme_minimal() +
  theme(axis.text.x = element_text(
    angle = 90, vjust = 1, hjust = 1, size = 5)) +
  labs(title = "Variable Importance",
       subtitle = "Clear importance distinction for first 11 predictors") +
  xlab("Variables") +
  ylab("Score")

print(p.imp)
  
