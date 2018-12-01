library(AppliedPredictiveModeling)
library(mlbench)
library(caret)
library(earth)
library(MASS)
library(elasticnet)
library(lars)
library(pls)
library(doParallel)
library(nnet)

data(permeability)


cat("After Non-Zero Variance, number of predictors in fingerprints is 388: \n")
NZVfingerprints <- nearZeroVar(fingerprints)
noNZVfingerprints <- fingerprints[,-NZVfingerprints]
print(str(noNZVfingerprints))
cat("\n\n")

# stratified random sample splitting with 75% training and 25% testing

set.seed(12345)
trainingRows =  createDataPartition(permeability, p = .75, list= FALSE)
trainFingerprints <- noNZVfingerprints[trainingRows,]
trainPermeability <- permeability[trainingRows,]

testFingerprints <- noNZVfingerprints[-trainingRows,]
testPermeability <- permeability[-trainingRows,]

set.seed(12345)

ctrl <- trainControl(method = "repeatedcv", repeats=5, number = 4)


# # For neuralnetwork, find the correlation and delete the correlated data
tooHigh <- findCorrelation(cor(trainFingerprints), cutoff = .75)
# 
# #  the tooHigh gives 99 correlated datas
trainXnnet = trainFingerprints[,-tooHigh]
testXnnet = testFingerprints[,-tooHigh]
# 
# set.seed(12344)

nnetGrid <- expand.grid(.decay = c(0, 0.01, .1),
                        .size = c(1:10),
                        ## The next option is to use bagging (see the
                        ## next chapter) instead of different random
                        ## seeds.
                        .bag = FALSE)


nnetTune <- train(trainXnnet, trainFat,
                  method = "avNNet",
                  tuneGrid = nnetGrid,
                  trControl = ctrl,
                  ## Automatically standardize data prior to modeling
                  ## and prediction
                  preProc = c("center", "scale"),
                  linout = TRUE,
                  trace = FALSE,
                  MaxNWts = 10 * (ncol(trainXnnet) + 1) + 10 + 1,
                  maxit = 500)

prediction<-predict(nnetTune,testXnnet)
accuracy<-data.frame(obs=testPermeability,pred=prediction)
defaultSummary(accuracy)
plot(accuracy)


# # For MARS, using resampling method to tune the model  Selection Using GCV
set.seed(12345)
marsFit <- earth(trainFingerprints,trainPermeability)
summary(marsFit)

set.seed(12345)
permeabilitymarsGrid <- expand.grid(degree = 1:2,nprune = 2:13)
permeabilitymarsTuned <- train(trainFingerprints, trainPermeability,
                   method="earth",
                   tuneGrid = permeabilitymarsGrid,
                   trControl = ctrl)
# 

prediction<-predict(permeabilitymarsTuned,testFingerprints)
accuracy<-data.frame(obs=testPermeability,pred=prediction[,1])
defaultSummary(accuracy)
plot(accuracy)

#

# # For SVM, using radial function is automatic and if the data are linear in regression should use
# linear svm, otherwise radial SVM is good
set.seed(12345)
permeabilitysvmRTuned <- train(trainFingerprints, trainPermeability,
                               method="svmRadial",
                               tuneLength = 14,
                               preProc = c("center", "scale"),
                               trControl = ctrl)
# 
prediction<-predict(permeabilitysvmRTuned,testFingerprints)
accuracy<-data.frame(obs=testPermeability,pred=prediction)
defaultSummary(accuracy)
plot(accuracy)


# # For KNN, remove the near-zero-variance predictors 
# # And, do the centering and scaling 
permeabilityknnDescr <- trainFingerprints[ ,-nearZeroVar(trainFingerprints)]
set.seed(12345)
permeabilityknnTuned <- train(permeabilityknnDescr,trainPermeability,
                 method="knn",
                 preProc = c("center","scale"),
                 tuneGrid = data.frame(k=1:20),
                 trControl = ctrl)

prediction<-predict(permeabilityknnTuned,testFingerprints)
accuracy<-data.frame(obs=testPermeability,pred=prediction)
defaultSummary(accuracy)
plot(accuracy)
