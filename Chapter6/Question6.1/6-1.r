library(AppliedPredictiveModeling)
library(MASS)
library(caret)
library(elasticnet)
library(lars)
library(pls)

data(tecator)

#########################
# question 6.1(b) 
######################### 

colName = {}
for (i in 1:100){
  colName[i]<- paste("X",i)
}
colnames(absorp)<-colName

## The base R function prcomp can be used for PCA. In the code below,
## the data are centered and scaled prior to PCA.
pcaObject <- prcomp(absorp, center = TRUE, scale. = TRUE)

# The standard deviations for the columns in the data are stored in pcaObject as a sub-object called ad:
# Calculate the cumulative percentage of variance which each component
# accounts for.
percentVariance <- pcaObject$sd^2/sum(pcaObject$sd^2)*100
print(percentVariance[1:5])

# taking only 5 variances set npcs = 5
screeplot(pcaObject, npcs = 5, type = "lines", main = "Scree Plot for PCA Analysis")


#########################
#########################

#########################
# question 6.1(c,d,e) 
######################### 

# Response variable = Fat

# BarPlot of response variable 

counts <- table( endpoints[,2])
barplot(counts, main="Fat  Distribution",
        xlab="Percentage Of Fat ") 


# splitting data into 80% and 20% based on Fat Response
set.seed(12345)

trainingRows =  createDataPartition(endpoints[,2], p = .75, list= FALSE)

trainAbsorption <- absorp[ trainingRows, ]
testAbsorption <- absorp[-trainingRows, ]
trainFat <- endpoints[trainingRows, 2]
testFat <- endpoints[-trainingRows, 2]

ctrl <- trainControl(method = "repeatedcv", repeats=4)

cat("\n")

set.seed(12345)

# simple linear regression
meatln <- train(x = trainAbsorption , y = trainFat, method = "lm", trControl = ctrl)
print(meatln)


prediction<-predict(meatln,testAbsorption)
accuracy3<-data.frame(obs=testFat,pred=prediction)
defaultSummary(accuracy3)
plot(accuracy3)



cat("\n")

# PCR method
meatPCR <- train(x = trainAbsorption , y = trainFat, method = "pcr", trControl = ctrl, tuneLength = 24)
print(meatPCR)
plot(meatPCR)

cat("\n")
cat("\n")

# PLS method
meatPLS <- train(x = trainAbsorption , y = trainFat, method = "pls", trControl = ctrl, tuneLength = 24)
print(meatPLS)
plot(meatPLS)
cat("\n")



# Ridge Regression Method
meatRg <- train(x = trainAbsorption , y = trainFat, method = "ridge",
                 trControl = ctrl,
                 preProcess = c("center","scale"),
                 tuneGrid = expand.grid(lambda = seq(0,1,length=15)))

print(meatRg)
plot(meatRg)
cat("\n")

prediction<-predict(meatRg,testAbsorption)
accuracy2<-data.frame(obs=testFat,pred=prediction)
defaultSummary(accuracy2)
plot(accuracy2)


# Lasso Regression Method
meatLasso <- train(x = trainAbsorption , y = trainFat, method = "lasso",
                trControl = ctrl,
                preProcess = c("center","scale"),
                tuneGrid = expand.grid(fraction = seq(0.1,1,length=20)))

print(meatLasso)
plot(meatLasso)
cat("\n")


prediction<-predict(meatLasso,testAbsorption)
accuracy3<-data.frame(obs=testFat,pred=prediction)
defaultSummary(accuracy3)
plot(accuracy3)



# Elastic Net Method
meatEls <- train(x = trainAbsorption , y = trainFat, method = "enet",
                 trControl = ctrl,
                 preProcess = c("center","scale"),
                 tuneGrid = expand.grid(lambda = c(0,.001,.01,.1,1), 
                                        fraction = seq(0.05,1,length=20)))

print(meatEls)
plot(meatEls)
cat("\n")

prediction<-predict(meatEls,testAbsorption)
accuracy1<-data.frame(obs=testFat,pred=prediction)
defaultSummary(accuracy1)
plot(accuracy1)
