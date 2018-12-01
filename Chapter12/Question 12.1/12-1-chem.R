library(caret)
library(AppliedPredictiveModeling)

data(hepatic)
# use ?hepatic to see more details


library(MASS)
set.seed(975)

barplot(table(injury),col=c("yellow","red","green"), main="Class Distribution")


set.seed(975)

#------------------------------------------------------------------------
# Use the Chemical predictors:
#------------------------------------------------------------------------


# this gives removes near-zero variance 
# this is a categorical predictor and should remove near zero variance for this data
zv_cols = nearZeroVar(chem)
noZVChem = chem[,-zv_cols]


#remove the correlation between the predictors
highCorChem<-findCorrelation(cor(noZVChem),cutoff = .75)
filteredCorChem <- noZVChem[,-highCorChem]



# splitting data into 75% and 25% based on injury response
set.seed(975)
trainingRows =  createDataPartition(injury, p = .75, list= FALSE)

trainChem <- filteredCorChem[trainingRows,]
testChem <- filteredCorChem[-trainingRows, ]

trainInjury <- injury[trainingRows]
testInjury <- injury[-trainingRows]


ctrl <- trainControl(summaryFunction = defaultSummary)

############ Logistic Regression Analysis #############
# logistic regression

library(caret)
set.seed(975)
lrChem <- train(x=trainChem,
               y = trainInjury,
               method = "multinom",
               metric = "Accuracy",
               trControl = ctrl)


predictionLRChem<-predict(lrChem,testChem)

confusionMatrix(data =predictionLRChem,
                reference = testInjury)

#######################################################
############ Linear Discriminant Analysis #############

# LDA Analysis
library(MASS)
set.seed(975)

ldaChem <- train(x = trainChem,
                y = trainInjury,
                method = "lda",
                preProc = c("center","scale"),
                metric = "Accuracy",
                trControl = ctrl)

predictionLDAChem <-predict(ldaChem,testChem)
confusionMatrix(data =predictionLDAChem,
                reference = testInjury)
##########################################################################

############## Partial Least Squares Discriminant Analysis ###############
library(MASS)
set.seed(975)
plsChem <- train(x = trainChem,
                y = trainInjury,
                method = "pls",
                tuneGrid = expand.grid(.ncomp = 1:1),
                preProc = c("center","scale"),
                metric = "Accuracy",
                trControl = ctrl)

predictionPLSChem <-predict(plsChem,testChem)
confusionMatrix(data =predictionPLSChem,
                reference = testInjury)

#######################################################
########### Penalized Models ###########

########### Penalized Models for Logistic Regression ###########

glmnGrid <- expand.grid(.alpha = c(0, .1, .2, .4),
                        .lambda = seq(.01, .2, length = 10))
set.seed(975)
glmnTunedChem <- train(x=trainChem,
                      y =trainInjury,
                      method = "glmnet",
                      tuneGrid = glmnGrid,
                      preProc = c("center", "scale"),
                      metric = "Accuracy",
                      trControl = ctrl)

predictionGlmnetChem <-  predict(glmnTunedChem,testChem)
confusionMatrix(data =predictionGlmnetChem,
                reference = testInjury)


########### Penalized Models for LDA ###########
library(sparseLDA)
set.seed(975)
sparseLdaModelChem <- sda(x=trainChem,
                      y =trainInjury,
                      lambda = 0.01,
                      stop = -73)
## the ridge parameter called lambda.

predictionSparseLDAChem <-  predict(sparseLdaModelChem,testChem)
confusionMatrix(data = predictionSparseLDAChem$class,
                reference = testInjury)

#######################################################
########### Nearest Shrunken Centroids ###########

library(pamr)

nscGridChem <- data.frame(.threshold = seq(0,4, by=0.1))
set.seed(975)
nscTunedChem <- train(x = trainChem, 
                     y = trainInjury,
                     method = "pam",
                     preProc = c("center", "scale"),
                     tuneGrid = nscGridBio,
                     metric = "Accuracy",
                     trControl = ctrl)

predictionNSCChem <-predict(nscTunedChem,testChem)
confusionMatrix(data =predictionNSCChem,
                reference = testInjury)

