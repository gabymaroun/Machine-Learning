library(spNNGP)


# Ridge vs. OLS in the presence of collinearity

library(MASS)
set.seed(10)
x1 <- rnorm(20)
x2 <- rnorm(20,mean=x1,sd=.01)
## cor(x1,x2) ## 0.9999435
y <- 3+x1+x2+rnorm(20)

lm(y~x1+x2)$coef


lm.ridge(y~x1+x2,lambda=2)


library(bigmemory)

setwd("C:/Users/gabym/Desktop/Semestre 3 UPPA/Machine Learning/WORKSHOP_DATA_SET_LIGHT/WORKSHOP_DATA_SET_LIGHT/")
dataX <- attach.big.matrix(dget("Xda.desc"))
Xdes <- describe(dataX)
Xdes

dataX_train <- dataX[1:120000,]
class(dataX_train)

dim(dataX_train)

col.sup <- which(apply(dataX_train,2,var)==0)
dataX_train <- dataX_train[,-c(col.sup)]
dim(dataX_train)


dataX_train_big <- as.big.matrix(dataX_train,backingfile="dataX_train_big.bin",
                                 descriptorfile ="dataX_train_big.desc")
class(dataX_train_big)

is.filebacked(dataX_train_big)



library(oem)
ylabel <- read.csv(file="emnist_labels_training_set.csv"
                   ,header=FALSE)[,1]
lambda <- 2
yy <- as.numeric(ylabel)[1:120000]
model.oem <- big.oem(x=dataX_train_big,y=yy,penalty=c("elastic.net")
                     ,alpha=0,family="gaussian",lambda=2)
error.train <- mean(predict(model.oem,dataX_train_big)-yy)**2
error.train


y.test <- as.numeric(ylabel)[120001:240000]
error.test <- mean(predict(model.oem,dataX[120001:240000,-c(col.sup)])-y.test)**2
error.test
