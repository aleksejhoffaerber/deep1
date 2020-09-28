source("packages.R")
options(width = 70, digits = 3)

c1 <- h2o.init(max_mem_size = "14G",
               nthreads = 6)


# NEW DATA:
download.file("http://archive.ics.uci.edu/ml/machine-learning-databases/00203/
YearPredictionMSD.txt.zip", destfile = "YearPredictionMSD.txt.zip")

unzip("YearPredictionMSD.txt.zip")

d <- fread("YearPredictionMSD.txt", sep = ",") # speedup
