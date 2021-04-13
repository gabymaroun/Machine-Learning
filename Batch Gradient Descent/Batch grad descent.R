library(MASS)
library(ramify)
library(caret) 
library(glmnet)
library(tidyverse)
library(caret)

## Simulate some m samples from true model:

set.seed(10)
m <-5000 ;d <-2 ;w <-c(0.5,-1.5)
x <-matrix(rnorm(m*2),ncol=2,nrow=m)
ptrue <-1/(1+exp(-x%*%matrix(w,ncol=1)))
y <-rbinom(m,size=1,prob = ptrue)
(w.est <-coef(glm(y~x[,1]+x[,2]-1,family=binomial)))

##The cross-entropy loss for this dataset

Cost.fct <-function(w1,w2) {
  w <-c(w1,w2) 
  cost <-sum(-y*x%*%matrix(w,ncol=1)+log(1+exp(x%*%matrix(w,ncol=1))))
  return(cost)
}


w1 <-seq(0, 1, 0.05)
w2 <-seq(-2, -1, 0.05)
cost <-outer(w1, w2, function(x,y)
  mapply(Cost.fct,x,y))

contour(x = w1, y = w2, z = cost)
points(x=w.est[1],y=w.est[2],col="black",lwd=2,lty=2,pch=8)

##Implentation of Batch Gradient Descent
sigmoid <-function(x) 1/(1+exp(-x))

batch.GD <-function(theta,alpha,epsilon,iter.max=500){ 
    tol <-1
    iter <-1
    res.cost <-Cost.fct(theta[1],theta[2])
    res.theta <-theta
    while (tol >epsilon &iter<iter.max) {
      error <-sigmoid(x%*%matrix(theta,ncol=1))-y
      theta.up <-theta-as.vector(alpha*matrix(error,nrow=1)%*%x)
      res.theta <-cbind(res.theta,theta.up)
      tol <-sum((theta-theta.up)**2)^0.5
      theta <-theta.up
      cost <-Cost.fct(theta[1],theta[2])
      res.cost <-c(res.cost,cost)
      iter <-iter +1
    }
    result <-list(theta=theta,res.theta=res.theta,res.cost=res.cost,iter=iter,tol.theta=tol)
    return(result)
}

dim(x);length(y)

theta0 <-c(0,-1); alpha=0.001
test <-batch.GD(theta=theta0,alpha,epsilon =0.0000001)
plot(test$res.cost,ylab="cost function",xlab="iteration",main="alpha=0.01",type="l")
abline(h=Cost.fct(w.est[1],w.est[2]),col="red")

contour(x = w1, y = w2, z = cost)
points(x=w.est[1],y=w.est[2],col="black",lwd=2,lty=2,pch=8)
record <-as.data.frame(t(test$res.theta))
points(record,col="red",type="o")

##Implementation of stochastic gradient descent

Stochastic.GD <-function(theta,alpha,epsilon=0.0001,epoch=50){
  epoch.max <-epoch
  tol <-1
  epoch <-1
  res.cost <-Cost.fct(theta[1],theta[2])
  res.cost.outer <-res.cost
  res.theta <-theta
  while (tol > epsilon & epoch<epoch.max) {
    for (i in 1:nrow(x)){
      errori <-sigmoid(sum(x[i,]*theta))-y[i]
      xi <-x[i,]
      theta.up <-theta-alpha*errori*xi
      res.theta <-cbind(res.theta,theta.up)
      tol <-sum((theta-theta.up)**2)^0.5
      theta <-theta.up
      cost <-Cost.fct(theta[1],theta[2])
      res.cost <-c(res.cost,cost)
    }
    epoch <-epoch +1
    cost.outer <-Cost.fct(theta[1],theta[2])
    res.cost.outer <-c(res.cost.outer,cost.outer)
  }
  result <-list(theta=theta,res.theta=res.theta,res.cost=res.cost,epoch=epoch,tol.theta=tol)
}

test.SGD <-Stochastic.GD(theta=theta0,alpha,epsilon =0.0001,epoch=10)

plot(test.SGD$res.cost,ylab="cost function",xlab="iteration",main="alpha=0.01",type="l")
abline(h=Cost.fct(w.est[1],w.est[2]),col="red")


contour(x = w1, y = w2, z = cost)
points(x=w.est[1],y=w.est[2],col="black",lwd=2,lty=2,pch=8)
record2 <-as.data.frame(t(test.SGD$res.theta))
points(record2,col="red",lwd=0.5)

##Batch
error <- matrix(NA, ncol = 3)
error <- rbind(c(4,4,4),error)

miniBatch1.GD <-function(theta,alpha,epsilon=0.0001,epoch=50){
  epoch.max <-epoch
  tol <-1
  epoch <-1
  res.cost <-Cost.fct(theta[1],theta[2])
  res.cost.outer <-res.cost
  res.theta <-theta
  while (tol > epsilon & epoch<epoch.max) {
    for (i in 1:nrow(x)/10){
      errorbatch <- matrix(NA, ncol=3)
      xbatch <- matrix(NA, ncol = 2)
      for (j in ((i-1)*10):(i*10)) {
        
        #errorj <-sigmoid(sum(x[j,]*theta))-y[j]
        #errorbatch <- rbind(as.matrix(errori),errorbatch)
        
        xbatch <- rbind(x[j,],xbatch)
      }
      error <-sigmoid(xbatch%*%matrix(theta,ncol=1))-y
      theta.up <-theta-as.vector(alpha*matrix(error,nrow=1)%*%x)
      
      res.theta <-cbind(res.theta,theta.up)
      tol <-sum((theta-theta.up)**2)^0.5
      theta <-theta.up
      cost <-Cost.fct(theta[1],theta[2])
      res.cost <-c(res.cost,cost)
    }
    epoch <-epoch +1
    cost.outer <-Cost.fct(theta[1],theta[2])
    res.cost.outer <-c(res.cost.outer,cost.outer)
  }
  result <-list(theta=theta,res.theta=res.theta,res.cost=res.cost,epoch=epoch,tol.theta=tol)
}







xbatch <- matrix(0, ncol = 2)
ybatch <- matrix(0, ncol = 1)




minibatch2.GD <-function(theta,alpha,epsilon,iter.max=500){ 
  tol <-1
  iter <-1
  res.cost <-Cost.fct(theta[1],theta[2])
  res.theta <-theta
  while (tol >epsilon &iter<iter.max) {
    xbatch <- x[iter,]
    ybatch <- y[iter]
    
    for (j in iter+1:iter+9) {
      xbatch <- rbind(xbatch,x[j,])
      ybatch <- rbind(ybatch,y[j])
    }
    error <-sigmoid(xbatch%*%matrix(theta,ncol=1))-ybatch
    theta.up <-theta-as.vector(alpha*matrix(error,nrow=1)%*%xbatch)
    res.theta <-cbind(res.theta,theta.up)
    tol <-sum((theta-theta.up)**2)^0.5
    theta <-theta.up
    cost <-Cost.fct(theta[1],theta[2])
    res.cost <-c(res.cost,cost)
    iter <-iter +10
  }
  result <-list(theta=theta,res.theta=res.theta,res.cost=res.cost,iter=iter,tol.theta=tol)
  return(result)
}

dim(x);length(y)

theta0 <-c(0,-1); alpha=0.002
test <-minibatch2.GD(theta=theta0,alpha,epsilon =0.00001)
plot(test$res.cost,ylab="cost function",xlab="iteration",main="alpha=0.01",type="l")
abline(h=Cost.fct(w.est[1],w.est[2]),col="red")

contour(x = w1, y = w2, z = cost)
points(x=w.est[1],y=w.est[2],col="black",lwd=2,lty=2,pch=8)
record <-as.data.frame(t(test$res.theta))
points(record,col="red",type="o")


# setwd("C:/Users/gabym/Desktop/Semestre 3 UPPA/Machine Learning/MATERIAL_STUDENT/Mnist/")
# setwd(getwd())
setwd("C:/Users/gabym/Desktop/Semestre 3 UPPA/Machine Learning/MATERIAL_STUDENT/Mnist/")
file_training_set_image <- "train-images-idx3-ubyte/train-images.idx3-ubyte"
file_training_set_label <- "train-labels-idx1-ubyte/train-labels.idx1-ubyte"
file_test_set_image <- "t10k-images-idx3-ubyte/t10k-images.idx3-ubyte"
file_test_set_label <- "t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte"
nb_Chiffre <- 1000
nb_Pixel <- 28

extract_images <- function(file, nbimages = NULL) {
  if (is.null(nbimages)) { # We extract all images
    nbimages <- as.numeric(paste("0x", paste(readBin(file, "raw", n = 8)[5:8], collapse = ""), sep = ""))
  }
  nbrows <- as.numeric(paste("0x", paste(readBin(file, "raw", n = 12)[9:12], collapse = ""), sep = ""))
  nbcols <- as.numeric(paste("0x", paste(readBin(file, "raw", n = 16)[13:16], collapse = ""), sep = ""))
  raw <- readBin(file, "raw", n = nbimages * nbrows * nbcols + 16)[-(1:16)]
  return(array(as.numeric(paste("0x", raw, sep="")), dim = c(nbcols, nbrows, nbimages)))
}
extract_labels <- function(file) {
  nbitem <- as.numeric(paste("0x", paste(readBin(file, "raw", n = 8)[5:8], collapse = ""), sep = ""))
  raw <- readBin(file, "raw", n = nbitem + 8)[-(1:8)]
  return(as.numeric(paste("0x", raw, sep="")))
}

# images_training_set <- extract_images(file_training_set_image, nb_Chiffre)
# images_test_set <- extract_images(file_test_set_image, nb_Chiffre)
labels_training_set <- extract_labels(file_training_set_label)
labels_test_set <- extract_labels(file_test_set_label)
# data.frame(images_training_set)



images_training_set <- extract_images(file_training_set_image, 60000)
images_training_set=t(matrix(images_training_set, dim(images_training_set)[1]*dim(images_training_set)[2], dim(images_training_set)[3]))
x <-images_training_set

## Permet d'obtenir la matrice 
temp <- matrix(nrow = 0, ncol = 10)
vrai_y <- matrix(nrow = 0, ncol = 10)
for(i in 1:60000){
  temp <- matrix(-1, 1, 10)
  temp[labels_training_set[i] + 1] <- 1
  vrai_y <- rbind(vrai_y, temp)
}

vrai_y


images_test_set <- extract_images(file_test_set_image, 10000)

images_test_set=t(matrix(images_test_set, dim(images_test_set)[1]*dim(images_test_set)[2], dim(images_test_set)[3]))
x_test <-images_test_set

temp <- matrix(nrow = 0, ncol = 10)
vecY_test <- matrix(nrow = 0, ncol = 10)
for(i in 1:10000){
  temp <- matrix(-1, 1, 10)
  temp[labels_test_set[i] + 1] <- 1
  vecY_test <- rbind(vecY_test, temp)
}

y <- labels_training_set
y_test <- labels_test_set
dim(x_test);length(y_test)

tune <- cv.glmnet(x_test,y_test,alpha=0, family = "multinomial")
plot(tune)

ridge <- glmnet(x,y,lambda=tune$lambda.1se,alpha=0, family = "multinomial")
head(ridge$beta)


tune.lasso <- cv.glmnet(x,y,alpha=1)
plot(tune.lasso)

lasso <- glmnet(x,y,lambda=tune.lasso$lambda.1se,alpha=1)
head(lasso$beta)







probabilities <- ridge %>% predict(x_test)
probabilities <- predict(ridge, x_test)
plot(probabilities)

predicted.classes <- ifelse(probabilities > 0.5, "pos", "neg")
# Model accuracy
observed.classes <- x_test
mean(predicted.classes == observed.classes)

# pred_mult <- function(x,y){
#   res <- predict(ridge,
#                  newdata=data.frame(x=x_test,y=y_test),type="probs")
#   apply(res,MARGIN=1,which.max)
# }
# 
# x_grid <- seq(0,1,length=101)
# y_grid <- seq(0,1,length=101)
# z_grid <- outer(x_grid,y_grid,FUN=pred_mult)
# 
# image(x_grid,y_grid,z_grid,col=clr2)
# points(x,y,pch=19,cex=2,col=clr1[z+1])
# Fit the model
# full.model <- glm(diabetes ~., data = train.data, family = binomial)
# Make predictions
# probabilities <- full.model %>% predict(test.data, type = "response")
set.seed(10)
m <-5000 ;
w <-matrix(rnorm(m*2),ncol=1,nrow=784)
# xdelui <-matrix(rnorm(m*2),ncol=2,nrow=m)

ptrue <-1/(1+exp(-x_test%*%w))
y <-rbinom(m,size=1,prob = ptrue)
vrai_y
tot<-0
for (i in 1:784){
  tot=tot+x_test[,i]
}
View(tot)

temp <- matrix(nrow = 0, ncol = 10)
vecY_test <- matrix(nrow = 0, ncol = 10)
for(i in 1:10000){
  temp <- matrix(-1, 1, 10)
  temp[labels_test_set[i] + 1] <- 1
  vecY_test <- rbind(vecY_test, temp)
}
(w.est <-coef(glm(vecY_test~tot-1,family=binomial)))

##The cross-entropy loss for this dataset

Cost.fct <-function(w1,w2) {
  w <-c(w1,w2) 
  cost <-sum(-y*x%*%matrix(w,ncol=1)+log(1+exp(x%*%matrix(w,ncol=1))))
  return(cost)
}


w1 <-seq(0, 1, 0.05)
w2 <-seq(-2, -1, 0.05)
cost <-outer(w1, w2, function(x,y)
  mapply(Cost.fct,x,y))


xbatch <- matrix(0, ncol = 2)
ybatch <- matrix(0, ncol = 1)

minibatch2.GD <-function(theta,alpha,epsilon,iter.max=500){ 
  tol <-1
  iter <-1
  res.cost <-Cost.fct(theta[1],theta[2])
  res.theta <-theta
  while (tol >epsilon &iter<iter.max) {
    xbatch <- x[iter,]
    ybatch <- y[iter]
    
    for (j in iter+1:iter+9) {
      xbatch <- rbind(xbatch,x[j,])
      ybatch <- rbind(ybatch,y[j])
    }
    error <-sigmoid(xbatch%*%matrix(theta,ncol=1))-ybatch
    theta.up <-theta-as.vector(alpha*matrix(error,nrow=1)%*%xbatch)
    res.theta <-cbind(res.theta,theta.up)
    tol <-sum((theta-theta.up)**2)^0.5
    theta <-theta.up
    cost <-Cost.fct(theta[1],theta[2])
    res.cost <-c(res.cost,cost)
    iter <-iter +10
  }
  result <-list(theta=theta,res.theta=res.theta,res.cost=res.cost,iter=iter,tol.theta=tol)
  return(result)
}

dim(x);length(y)

theta0 <-c(0,-1); alpha=0.002
test <-minibatch2.GD(theta=theta0,alpha,epsilon =0.00001)
plot(test$res.cost,ylab="cost function",xlab="iteration",main="alpha=0.01",type="l")
abline(h=Cost.fct(w.est[1],w.est[2]),col="red")

contour(x = w1, y = w2, z = cost)
points(x=w.est[1],y=w.est[2],col="black",lwd=2,lty=2,pch=8)
record <-as.data.frame(t(test$res.theta))
points(record,col="red",type="o")
