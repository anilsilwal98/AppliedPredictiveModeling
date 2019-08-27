library(caret)
library(kernlab)
library(lattice)
library(ggplot2)

x<-runif(100,min=2,max=10)
y<-sin(x)+rnorm(length(x))*0.25
sinData<-data.frame(x=x,y=y)

# create a data Grid
dataGrid<-data.frame(x=seq(2,10,length=100))

# this is done to divide the graph in 4 columns
par(mfrow = c(2,2))

svmParam1 <- expand.grid(eps = c(0.01,0.05,0.1,0.5), costs = 2^c(-2,0,2,8))
for ( i in 1: nrow(svmParam1)){
  set.seed(121)
  rbfSVM <- ksvm(x=x,y=y, data=sinData,
                 kernel="rbfdot",kpar="automatic",
                 C= svmParam1$costs[i],epsilon = svmParam1$eps[i])
  
  tmp<-data.frame(x=dataGrid$x,y =predict(rbfSVM,newdata= dataGrid), 
                  eps=paste("epsilon",format(svmParam1$eps)[i]),
                  costs=paste("costs",format(svmParam1$costs)[i]))
  
  svmPred1 <- if(i==1) tmp else rbind(tmp,svmPred1)
  
  modelPrediction <- predict(rbfSVM, newdata = dataGrid)
  plot(x,y)
  points(x = dataGrid$x, y = modelPrediction[,1], type = "l", col = "blue")
  
}
svmPred1$costs <- factor(svmPred1$costs, levels=rev(levels(svmPred1$costs)))


svmParam2 <- expand.grid(eps = c(0.01,0.05,0.1,0.5),
                         costs = 2^c(-2,0,2,8),
                         sigma=as.vector(sigest(y~x,data=sinData,frac=.75)))

for ( i in 1: nrow(svmParam2)){
  set.seed(121)
  rbfSVM <- ksvm(x=x,y=y, data=sinData,
                 kernel="rbfdot",
                 kpar=list(sigma=svmParam2$sigma[i]),
                 C= svmParam2$costs[i],
                 epsilon = svmParam2$eps[i]
                 )
  
  tmp<-data.frame(x=dataGrid$x,
                  y =predict(rbfSVM,newdata= dataGrid), 
                  eps=paste("epsilon",format(svmParam2$eps)[i]),
                  costs=paste("costs",format(svmParam2$costs)[i]),
                  sigma=paste("sigma",format(svmParam2$sigma,digits=2)[i])
  )
  svmPred2 <- if(i==1) tmp else rbind(tmp,svmPred2)
  
  modelPrediction <- predict(rbfSVM, newdata = dataGrid)
  plot(x,y)
  points(x = dataGrid$x, y = modelPrediction[,1], type = "l", col = "blue")
  
}

svmPred2$costs <- factor(svmPred2$costs, levels=rev(levels(svmPred2$costs)))
svmPred2$sigma <- factor(svmPred2$sigma, levels=rev(levels(svmPred2$sigma)))

