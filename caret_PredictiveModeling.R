#######################################################
#      Streamline Predictive Modeling with caret       #
#           http://topepo.github.io/caret/             #
########################################################

# caret, short for _C_lassification _A_nd _RE_gression _T_raining
# Available models in caret: http://topepo.github.io/caret/available-models.html
# author Max Kuhn, PhD in Biostatistics, Software Engineer at RStudio, former non-clinical Statistician at Pfizer, making order out of chaos
# Short introduction to caret: https://cran.r-project.org/web/packages/caret/vignettes/caret.html
# Kuhn, Max. "Caret package." Journal of statistical software 28.5 (2008): 1-26.
# Book Applied Predictive Modeling http://appliedpredictivemodeling.com/

set.seed(666)

#### Getting started ####
# install caret package for the fist time
install.packages("caret")
# Load packages
library(caret)
library(mlbench)
library(ggplot2)
# load dataset https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html
data(BostonHousing)
# Explore data
summary(BostonHousing)
dim(BostonHousing)
ggplot(data = BostonHousing, aes(x = medv)) + geom_histogram()
# Visualization via lattice
theme1 <- trellis.par.get()
theme1$plot.symbol$col = rgb(.2, .2, .2, .4)
theme1$plot.symbol$pch = 16
theme1$plot.line$col = rgb(1, 0, 0, .7)
theme1$plot.line$lwd <- 2
trellis.par.set(theme1)
featurePlot(x = BostonHousing[, c("age", "lstat", "tax", "rad")], 
            y = BostonHousing$medv, 
            plot = "scatter",
            type = c("p", "smooth"),
            span = .5,
            layout = c(4, 1))
featurePlot(x = BostonHousing[,c("medv", "nox", "dis", "crim")], 
            y = BostonHousing$chas, 
            plot = "box",
            scales = list(x = list(relation="free"), 
                          y = list(relation="free")))

#### Pre-processing ####
# corr
descrCor <- cor(BostonHousing[,-4])
findCorrelation(descrCor, cutoff = .8) #tax & rad
featurePlot(x = BostonHousing$tax, 
            y = BostonHousing$rad, 
            plot = "scatter",
            type = c("p", "smooth"))
# Putting it all together
pp <- preProcess(BostonHousing[, -14],  method = c("center", "scale"))
pp
transformed <- predict(pp, newdata = BostonHousing[, -14])

#### Splitting the data ####
# Simple split: continuous
inTrain <- createDataPartition(
  y = BostonHousing$medv,  ## the outcome data are needed
  p = .75,  ## The percentage of data in the training set
  list = FALSE
)
training <- cbind(transformed[ inTrain,], medv = BostonHousing[ inTrain, 14])
testing  <- cbind(transformed[-inTrain,], medv = BostonHousing[-inTrain, 14])
nrow(training)
nrow(testing)

# Simple split: binary
BostonHousing2 = BostonHousing
BostonHousing2$medv = as.numeric(BostonHousing$medv > 22)
inTrain2 <- createDataPartition(
  y = BostonHousing2$medv,  ## the outcome data are needed
  p = .75,  ## The percentage of data in the training set
  list = FALSE
)
training2 <- cbind(transformed[ inTrain2,], medv = BostonHousing2[ inTrain2, 14])
testing2  <- cbind(transformed[-inTrain2,], medv = BostonHousing2[-inTrain2, 14])
nrow(training2)
nrow(testing2)

#### trainning classification ####
# Tuning control
fitControl <- trainControl(## 10-fold CV repeated ten times
  method = "repeatedcv",
  number = 10,
  repeats = 10)

# Penalized Logistic regression
plrFit <- train(as.factor(medv) ~ ., data = training2, 
                method = "plr", 
                trControl = fitControl)
plrFit

#alternate tuning grids: parameter vs hyperparameter
plrGrid <-  expand.grid(lambda = c(1:9 %o% 10^(-3:1)), 
                        cp = "bic")
plrFit2 <- train(as.factor(medv) ~ ., data = training2, 
                method = "plr",
                trControl = fitControl,
                tuneGrid = plrGrid)
plrFit2

# Nearest Neighbor
knnFit <- train(as.factor(medv) ~ ., data = training2, 
                method = "knn", 
                trControl = fitControl,
                tuneLength = 10)
knnFit
plot(knnFit)

# Random Forest
rfFit <- train(as.factor(medv) ~ ., data = training2, 
               method = "rf", 
               trControl = fitControl)
rfFit

# Support Vector Machines with Linear kernel
svmFit <- train(as.factor(medv) ~ ., data = training2, 
                method = "svmLinear", 
                trControl = fitControl)
svmFit

# Extract predictions and measure performance
predict(rfFit, newdata = testing2[1:10, 1:13])
confusionMatrix(data = predict(rfFit, testing2[, 1:13]), reference = as.factor(testing2[, 14]))
predict(rfFit, newdata = testing2[1:10, 1:13], type = "prob")

# Compare models
models_compare <- resamples(list(plr = plrFit, knn=knnFit, RF=rfFit, SVM=svmFit))
summary(models_compare)
scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(models_compare, scales=scales)



#### trainning regression ####

# Lasso Regression
lassoFit <- train(medv ~ ., data = training, 
                  method = "lasso", 
                  trControl = fitControl)
lassoFit

# Nearest Neighbor
knnFit2 <- train(medv ~ ., data = training, 
                method = "knn", 
                trControl = fitControl,
                tuneLength = 10)
knnFit2
plot(knnFit2)

# Random Forest
rfFit2 <- train(medv ~ ., data = training, 
                method = "rf", 
                trControl = fitControl)
rfFit2

# Support Vector Machine
svmFit2 <- train(medv ~ ., data = training, 
                 method = "svmLinear", 
                 trControl = fitControl)
svmFit2

# Extract Prediction and measure performance
predict(rfFit2, testing[1:10, -14])
postResample(pred = predict(rfFit2, testing[, -14]), obs = testing$medv)

# Compare models
models_compare2 <- resamples(list(lasso = lassoFit, knn=knnFit2, RF=rfFit2, SVM=svmFit2))
summary(models_compare2)
scales <- list(x=list(relation="free"), y=list(relation="free"))
bwplot(models_compare2, scales=scales)


#### Variable importance ####
rfImp <- varImp(rfFit)
plot(rfImp)



