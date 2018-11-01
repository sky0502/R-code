set.seed(666)
library(caret)
library(AppliedPredictiveModeling)
library(mlbench)

data(BostonHousing)
summary(BostonHousing)
dim(BostonHousing)
ggplot(data = BostonHousing, aes(x = medv)) + geom_histogram()

theme1 <- trellis.par.get()
theme1$plot.symbol$col = rgb(.2, .2, .2, .4)
theme1$plot.symbol$pch = 16
theme1$plot.line$col = rgb(1, 0, 0, .7)
theme1$plot.line$lwd <- 2
trellis.par.set(theme1)
featurePlot(x = BostonHousing[, c("age", "lstat", "tax")], 
            y = BostonHousing$medv, 
            plot = "scatter",
            type = c("p", "smooth"),
            span = .5,
            layout = c(3, 1))

# corr
descrCor <- cor(BostonHousing[,-4])
findCorrelation(descrCor, cutoff = .8) #tax & rad
featurePlot(x = BostonHousing$tax, 
            y = BostonHousing$rad, 
            plot = "scatter",
            type = c("p", "smooth"))
pp <- preProcess(BostonHousing[, -14],  method = c("center", "scale"))
pp
transformed <- predict(pp, newdata = BostonHousing[, -14])

# split continuous
inTrain <- createDataPartition(
  y = BostonHousing$medv,  ## the outcome data are needed
  p = .75,  ## The percentage of data in the training set
  list = FALSE
)
training <- cbind(transformed[ inTrain,], medv = BostonHousing[ inTrain, 14])
testing  <- cbind(transformed[-inTrain,], medv = BostonHousing[-inTrain, 14])
nrow(training)
nrow(testing)

# split binary
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
# Logistic regression
# Nearest Neighbor
# Random Forest
# Support Vector Machine

#### trainning regression ####
# Regression
# Random Forest
# Support Vector Machine

fitControl <- trainControl(## 10-fold CV repeated ten times
  method = "repeatedcv",
  number = 10,
  repeats = 10)

lassoFit <- train(medv ~ ., data = training, 
                 method = "lasso", 
                 trControl = fitControl, 
                 verbose = FALSE)
lassoFit

#### Variable importance ####
