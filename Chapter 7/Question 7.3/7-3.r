library(mlbench)
library(caret)
library(earth)
library(doParallel)
library(nnet)

data(tecator)

colName = {}
for (i in 1:100){
  colName[i]<- paste("X",i)
}
colnames(absorp)<-colName


# splitting data into 80% and 20% based on Fat Response
set.seed(12345)

trainingRows =  createDataPartition(endpoints[,2], p = .80, list= FALSE)

trainAbsorption <- absorp[ trainingRows, ]
testAbsorption <- absorp[-trainingRows, ]
trainFat <- endpoints[trainingRows, 2]
testFat <- endpoints[-trainingRows, 2]

ctrl <- trainControl(method = "repeatedcv", repeats=4)

# # For neuralnetwork, find the correlation and delete the correlated data
tooHigh <- findCorrelation(cor(trainAbsorption), cutoff = .80)

#  the tooHigh gives 99 correlated datas
trainXnnet1 = trainAbsorption[,-tooHigh]
testXnnet1 = testAbsorption[,-tooHigh]

set.seed(12344)

library(nnet)
library(caret)

# without using PCA
# to train in parallel to 5 processor
cl <- makePSOCKcluster(5)
registerDoParallel(cl)

nnetGrid1 <- expand.grid(.decay = c(0, 0.01, .1),
                        .size = c(1:10),
                        ## The next option is to use bagging (see the
                        ## next chapter) instead of different random
                        ## seeds.
                        .bag = FALSE)

nnetTune1 <- train(trainAbsorption, trainFat,
                  method = "avNNet",
                  trControl = ctrl,
                  preProc = c("center", "scale"),
                  linout = TRUE,
                  trace = FALSE,
                  MaxNWts = 10 * (ncol(trainAbsorption) + 1) + 10 + 1,
                  maxit = 500,
                  tuneGrid = nnetGrid1)

prediction<-predict(nnetTune1,testAbsorption)
accuracy<-data.frame(obs=testFat,pred=prediction)
defaultSummary(accuracy)
plot(accuracy)
## When you are done:
stopCluster(cl)

# using PCA
# to train in parallel to 5 processor
cl <- makePSOCKcluster(5)
registerDoParallel(cl)

nnetGrid1 <- expand.grid(.decay = c(0, 0.01, .1),
                         .size = c(1:10),
                         ## The next option is to use bagging (see the
                         ## next chapter) instead of different random
                         ## seeds.
                         .bag = FALSE)

nnetTune2 <- train(trainAbsorption, trainFat,
                   method = "avNNet",
                   trControl = ctrl,
                   preProc = c("center", "scale","pca"),
                   linout = TRUE,
                   trace = FALSE,
                   MaxNWts = 10 * (ncol(trainAbsorption) + 1) + 10 + 1,
                   maxit = 500,
                   tuneGrid = nnetGrid1)

prediction<-predict(nnetTune2,testAbsorption)
accuracy<-data.frame(obs=testFat,pred=prediction[-41])
defaultSummary(accuracy)
plot(accuracy)
## When you are done:
stopCluster(cl)



# # For MARS, using resampling method to tune the model  Selection Using GCV
set.seed(12345)
marsFit <- earth(trainAbsorption,trainFat)
summary(marsFit)
# 
set.seed(12345)
marsGrid <- expand.grid(.degree = 1:2, .nprune = 2:18)
marsTuned <- train(trainAbsorption, trainFat, method="earth",
                   tuneGrid = marsGrid,
                   trControl = ctrl)

prediction<-predict(marsTuned,testAbsorption)
accuracy<-data.frame(obs=testFat,pred=prediction[,1])
defaultSummary(accuracy)
plot(accuracy)
# 

# # For SVM, using radial function is automatic and if the data are linear in regression should use
# linear svm, otherwise radial SVM is good
set.seed(12345)
svmRTuned <- train(trainAbsorption, trainFat, method="svmRadial",
                   tuneLength = 14,
                   preProc = c("center", "scale"),
                   trControl = ctrl)
# 
prediction<-predict(svmRTuned,testAbsorption)
accuracy<-data.frame(obs=testFat,pred=prediction)
defaultSummary(accuracy)
plot(accuracy)


# # For KNN, remove the near-zero-variance predictors 
# # And, do the centering and scaling 
knnDescr <- trainAbsorption[ ,-nearZeroVar(trainAbsorption)]
set.seed(12345)
knnTune <- train(trainAbsorption,trainFat,
                 method="knn",
                 preProc = c("center","scale"),
                 tuneGrid = data.frame(k=1:20),
                 trControl = ctrl)
# 
prediction<-predict(knnTune,testAbsorption)
accuracy<-data.frame(obs=testFat,pred=prediction)
defaultSummary(accuracy)
plot(accuracy)

