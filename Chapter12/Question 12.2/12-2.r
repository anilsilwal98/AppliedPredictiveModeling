library(caret)
library(AppliedPredictiveModeling)

data(oil)
# use ?hepatic to see more details


library(MASS)
set.seed(975)

barplot(table(oilType),col=c("yellow"), main="Class Distribution")



#this gives 0 predictor with zero-variance
nearZeroVar(fattyAcids,saveMetrics =TRUE)

#remove the correlation between the predictors
highCorM<-findCorrelation(cor(fattyAcids),cutoff = .75)
filteredCorFatty <- fattyAcids[,-highCorM]

# after removing the highly correlated predictor, we split the data using 
# stratified random sampling

# splitting data into 80% and 20% based on oilType response

set.seed(975)
trainingRows =  createDataPartition(oilType, p = .80, list= FALSE)

trainFattyAcids <- filteredCorFatty[ trainingRows, ]
testFattyAcids <- filteredCorFatty[-trainingRows, ]

trainOilType <- oilType[trainingRows]
testOilType <- oilType[-trainingRows]

ctrl <- trainControl(summaryFunction = defaultSummary)

############ Logistic Regression Analysis #############
# logistic regression

library(caret)
set.seed(975)
lrFattyAcids <- train(x=trainFattyAcids,
               y = trainOilType,
               method = "multinom",
               metric = "Accuracy",
               trControl = ctrl)


predictionLRFattyAcids<-predict(lrFattyAcids,testFattyAcids)

confusionMatrix(data =predictionLRFattyAcids,
                reference = testOilType)

#######################################################
############ Linear Discriminant Analysis #############

# LDA Analysis
library(MASS)
set.seed(975)


ldaFattyAcids <- train(x = trainFattyAcids,
                y = trainOilType,
                method = "lda",
                metric = "Accuracy",
                trControl = ctrl)

predictionLDAFattyAcids <-predict(ldaFattyAcids,testFattyAcids)
confusionMatrix(data =predictionLDAFattyAcids,
                reference = testOilType)
##########################################################################

############## Partial Least Squares Discriminant Analysis ###############
library(MASS)
set.seed(975)
plsFattyAcids <- train(x = trainFattyAcids,
                y = trainOilType,
                method = "pls",
                tuneGrid = expand.grid(.ncomp = 1:4),
                # preProc = c("center","scale"),
                metric = "Accuracy",
                trControl = ctrl)

predictionPLSFattyAcids <-predict(plsFattyAcids,testFattyAcids)
confusionMatrix(data =predictionPLSFattyAcids,
                reference = testOilType)

#######################################################
########### Penalized Models ###########

########### Penalized Models for Logistic Regression ###########
# glmnGrid <- expand.grid(.alpha = c(0, .1, .2, .4),
#                         .lambda = seq(.01, .2, length = 10))
glmnGrid <- expand.grid(.alpha = c(0, .1, .2, .4, .6, .8, 1),
                        .lambda = seq(.01, .2, length = 10))
set.seed(476)

glmnTunedLRFattyAcids<- train(x=trainFattyAcids,
                        y =trainOilType,
                        method = "glmnet",
                        tuneGrid = glmnGrid,
                        # preProc = c("center", "scale"),
                        metric = "Accuracy",
                        trControl = ctrl)

predictionGlmnetFattyAcids <-  predict(glmnTunedLRFattyAcids,testFattyAcids)
confusionMatrix(data =predictionGlmnetFattyAcids,
                reference = testOilType)


########### Penalized Models for LDA ###########
library(sparseLDA)
set.seed(975)
sparseLdaModelFattyAcids <- sda(x=trainFattyAcids,
                         y =trainOilType,
                         lambda = 0.01,
                         stop = -7)
## the ridge parameter called lambda.

predictionSparseLDAFattyAcids <-  predict(sparseLdaModelFattyAcids,testFattyAcids)
confusionMatrix(data =predictionSparseLDAFattyAcids$class,
                reference = testOilType)



#######################################################
########### Nearest Shrunken Centroids ###########

library(pamr)
nscGridFattyAcids <- data.frame(.threshold = seq(0,4, by=0.1))
set.seed(975)
nscTunedFattyAcids <- train(x = trainFattyAcids, 
                     y = trainOilType,
                     method = "pam",
                     # preProc = c("center", "scale"),
                     tuneGrid = nscGridFattyAcids,
                     metric = "Accuracy",
                     trControl = ctrl)

predictionNSCFattyAcids <-predict(nscTunedFattyAcids,testFattyAcids)
confusionMatrix(data =predictionNSCFattyAcids,
                reference = testOilType)

