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

####### Nonlinear Discriminant Analysis ##########

ctrl <- trainControl(summaryFunction = defaultSummary)
set.seed(476)
mdaFit <- train(x = filteredCorFatty, 
                y = oilType,
                method = "mda",
                metric = "Accuracy",
                tuneGrid = expand.grid(.subclasses = 1:3),
                trControl = ctrl)
mdaPrediction<-predict(mdaFit,filteredCorFatty)
confusionMatrix(mdaPrediction,oilType)

############### Neural Networks #############

library(nnet)
set.seed(476)
nnetGrid <- expand.grid(.size = 1:10, .decay = c(0, .1, 1, 2))

maxSize <- max(nnetGrid$.size)

numWts <- 1*(maxSize * (6 + 1) + maxSize + 1) ## 6 is the number of predictors

nnetFit <- train(x = filteredCorFatty, 
                 y = oilType,
                 method = "nnet",
                 metric = "Accuracy",
                 preProc = c("center", "scale", "spatialSign"),
                 tuneGrid = nnetGrid,
                 trace = FALSE,
                 maxit = 2000,
                 MaxNWts = numWts,
                 trControl = ctrl)
nnetFit

########## Flexible Discriminant Analysis ############

library(MASS)
set.seed(476)
marsGrid <- expand.grid(.degree = 1:2, .nprune = 2:38)
fdaTuned <- train(x = filteredCorFatty, 
                  y = oilType,
                  method = "fda",
                  metric = "Accuracy",
                  # Explicitly declare the candidate models to test
                  tuneGrid = marsGrid,
                  trControl = ctrl)

fdaTuned


############## Support Vector Machines ##########

library(MASS)
set.seed(476)
library(kernlab)
library(caret)

sigmaRangeReduced <- sigest(as.matrix(filteredCorFatty))

## Given a range of values for the "sigma" inverse width parameter 
## in the Gaussian Radial Basis kernel for use with SVM.
## The estimation is based on the data to be used.

svmRGridReduced <- expand.grid(.sigma = sigmaRangeReduced[1],
                               .C = 2^(seq(-4, 6)))
set.seed(476)
svmRModel <- train(x = filteredCorFatty, 
                   y = oilType,
                   method = "svmRadial",
                   metric = "Accuracy",
                   preProc = c("center", "scale"),
                   tuneGrid = svmRGridReduced,
                   fit = FALSE,
                   trControl = ctrl)
svmRModel


############ K-Nearest Neighbors #############
library(caret)
set.seed(476)
knnFit <- train(x = filteredCorFatty, 
                y = oilType,
                method = "knn",
                metric = "Accuracy",
                preProc = c("center", "scale"),
                ##tuneGrid = data.frame(.k = c(4*(0:5)+1, 20*(1:5)+1, 50*(2:9)+1)), ## 21 is the best
                tuneGrid = data.frame(.k = 1:50),
                trControl = ctrl)

knnFit


########## Naive Bayes ##########
library(klaR)
set.seed(476)
nbFit <- train( x = filteredCorFatty, 
                y = oilType,
                method = "nb",
                metric = "Accuracy",
                ## preProc = c("center", "scale"),
                # tuneGrid = data.frame(.k = c(4*(0:5)+1, 20*(1:5)+1, 50*(2:9)+1)), ## 21 is the best
                tuneGrid = data.frame(.fL = 2,.usekernel = TRUE,.adjust = TRUE),
                trControl = ctrl)

nbFit
