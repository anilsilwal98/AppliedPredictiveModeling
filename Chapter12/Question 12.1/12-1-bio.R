library(caret)
library(AppliedPredictiveModeling)

data(hepatic)
# use ?hepatic to see more details


library(MASS)
set.seed(975)

barplot(table(injury),col=c("yellow","red","green"), main="Class Distribution")

#------------------------------------------------------------------------
# Use the biological predictors:
#------------------------------------------------------------------------


#this gives Z114 predictor has zero-variance
nearZeroVar(bio)

#remove the Z114 predictor and then find the correlation between the predictors
noZVbio <- bio[,-114]

#remove the correlation between the predictors
highCorBio<-findCorrelation(cor(noZVbio),cutoff = .75)
filteredCorBio <- noZVbio[,-highCorBio]



# splitting data into 75% and 25% based on injury response
set.seed(975)
trainingRows =  createDataPartition(injury, p = .75, list= FALSE)

trainBio <- filteredCorBio[ trainingRows, ]
testBio <- filteredCorBio[-trainingRows, ]


trainInjury <- injury[trainingRows]
testInjury <- injury[-trainingRows]


ctrl <- trainControl(summaryFunction = defaultSummary)

############ Logistic Regression Analysis #############
# logistic regression

library(caret)
set.seed(975)
lrBio <- train(x=trainBio,
               y = trainInjury,
               method = "multinom",
               metric = "Accuracy",
               trControl = ctrl)


predictionLRBio<-predict(lrBio,testBio)

confusionMatrix(data =predictionLRBio,
                reference = testInjury)

#######################################################
############ Linear Discriminant Analysis #############

# LDA Analysis
library(MASS)
set.seed(975)


ldaBio <- train(x = trainBio,
                y = trainInjury,
                method = "lda",
                metric = "Accuracy",
                trControl = ctrl)

predictionLDABio <- predict(ldaBio,testBio)
confusionMatrix(data =predictionLDABio,
                reference = testInjury)
##########################################################################

############## Partial Least Squares Discriminant Analysis ###############
library(MASS)
set.seed(975)
plsBio <- train(x = trainBio,
                y = trainInjury,
                method = "pls",
                tuneGrid = expand.grid(.ncomp = 1:1),
                # preProc = c("center","scale"),
                metric = "Accuracy",
                trControl = ctrl)

predictionPLSBio <-predict(plsBio,testBio)
confusionMatrix(data =predictionPLSBio,
                reference = testInjury)

#######################################################
########### Penalized Models ###########

########### Penalized Models for Logistic Regression ###########
glmnGrid <- expand.grid(.alpha = c(0, .1, .2, .4),
                        .lambda = seq(.01, .2, length = 10))
set.seed(975)

glmnTunedLRBio <- train(x=trainBio,
                      y =trainInjury,
                      method = "glmnet",
                      tuneGrid = glmnGrid,
                      # preProc = c("center", "scale"),
                      metric = "Accuracy",
                      trControl = ctrl)

predictionGlmnetBio <-  predict(glmnTunedLRBio,testBio)
confusionMatrix(data =predictionGlmnetBio,
                reference = testInjury)


########### Penalized Models for LDA ###########
library(sparseLDA)
set.seed(975)
sparseLdaModelBio <- sda(x=trainBio,
                          y =trainInjury,
                          lambda = 0.01,
                          stop = -146)
## the ridge parameter called lambda.

predictionSparseLDABio <-  predict(sparseLdaModelBio,testBio)
confusionMatrix(data =predictionSparseLDABio$class,
                reference = testInjury)



#######################################################
########### Nearest Shrunken Centroids ###########

library(pamr)
nscGridBio <- data.frame(.threshold = seq(0,4, by=0.1))
set.seed(476)
nscTunedBio <- train(x = trainBio, 
                     y = trainInjury,
                     method = "pam",
                     # preProc = c("center", "scale"),
                     tuneGrid = nscGridBio,
                     metric = "Accuracy",
                     trControl = ctrl)

predictionNSCBio <-predict(nscTunedBio,testBio)
confusionMatrix(data =predictionNSCBio,
                reference = testInjury)

