library(AppliedPredictiveModeling)
library(MASS)
library(caret)
library(elasticnet)
library(lars)
library(pls)

data(permeability)

#########################
# question 6.2(b) 
######################### 
cat("Before Non-Zero Variance, number of predictors in fingerprints is 1107: \n")
print(str(fingerprints))
cat("\n\n")

cat("After Non-Zero Variance, number of predictors in fingerprints is 388: \n")
NZVfingerprints <- nearZeroVar(fingerprints)
noNZVfingerprints <- fingerprints[,-NZVfingerprints]
print(str(noNZVfingerprints))
cat("\n\n")

#########################
#########################

#########################
# question 6.2(c) 
######################### 

# stratified random sample splitting with 75% training and 25% testing

set.seed(12345)
trainingRows =  createDataPartition(permeability, p = .75, list= FALSE)

trainFingerprints <- noNZVfingerprints[trainingRows,]
trainPermeability <- permeability[trainingRows,]

testFingerprints <- noNZVfingerprints[-trainingRows,]
testPermeability <- permeability[-trainingRows,]

set.seed(12345)

ctrl <- trainControl(method = "repeatedcv", repeats=5, number = 4)


# PLS Model
permeabiltyPLS <- train(x = trainFingerprints , y = trainPermeability,preProcess = c("center","scale"), method = "pls", tuneGrid = expand.grid(ncomp = 1:15), trControl = ctrl)
print(permeabiltyPLS)
plot(permeabiltyPLS, metric ="Rsquared", main = "PLS Tuning Parameter for Permeability Data")

cat("\n")

# # Ridge Regression Method
# permeabiltyRg <- train(x = trainFingerprints , y = trainPermeability, method = "ridge",
#                 trControl = ctrl,
#                 preProcess = c("center","scale"),
#                 tuneGrid = expand.grid(lambda = seq(0,1,length=15)))
# 
# print(permeabiltyRg)
# plot(permeabiltyRg, metric ="Rsquared", main = "Ridge Regression Tuning Parameter for Permeability Data")
# 
# 
# # Lasso Regression Method
# meatLasso <- train(x = trainFingerprints , y = trainPermeability, method = "lasso",
#                    trControl = ctrl,
#                    preProcess = c("center","scale"),
#                    tuneGrid = expand.grid(fraction = seq(0.1,1,length=20)))
# 
# print(meatLasso)
# plot(meatLasso)
# cat("\n")
# 
# 
# # Elastic Net Method
# meatEls <- train(x = trainAbsorption , y = trainFat, method = "enet",
#                  trControl = ctrl,
#                  preProcess = c("center","scale"),
#                  tuneGrid = expand.grid(lambda = c(0,.001,.01,.1,1), 
#                                         fraction = seq(0.05,1,length=20)))
# 
# print(meatEls)
# plot(meatEls)
# cat("\n")
# 
# prediction<-predict(meatEls,testAbsorption)
# accuracy1<-data.frame(obs=testFat,pred=prediction)
# defaultSummary(accuracy1)
# plot(accuracy1)
# 
# #########################
# #########################
# 
