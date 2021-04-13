setwd("C:/Users/gabym/Desktop/Semestre 3 UPPA/Machine Learning/MATERIAL_STUDENT/Mnist/")
file_training_set_image <- "train-images-idx3-ubyte/train-images.idx3-ubyte"
file_training_set_label <- "train-labels-idx1-ubyte/train-labels.idx1-ubyte"
file_test_set_image <- "t10k-images-idx3-ubyte/t10k-images.idx3-ubyte"
file_test_set_label <- "t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte"
# file_training_set_image <- "~/Desktop/M2/Machine_learning/Mnist/train-images.idx3-ubyte"
# file_training_set_label <- "~/Desktop/M2/Machine_learning/Mnist/train-labels.idx1-ubyte"
# file_test_set_image <- "~/Desktop/M2/Machine_learning/Mnist/t10k-images.idx3-ubyte"
# file_test_set_label <- "~/Desktop/M2/Machine_learning/Mnist/t10k-labels.idx1-ubyte"

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
labels_training_set <- extract_labels(file_training_set_label)
labels_test_set <- extract_labels(file_test_set_label)
images_training_set <- extract_images(file_training_set_image, 60000)
images_training_set=t(matrix(images_training_set, dim(images_training_set)[1]*dim(images_training_set)[2], dim(images_training_set)[3]))



x <-images_training_set
y <- labels_training_set
images_test_set <- extract_images(file_test_set_image, 1000)
images_test_set=t(matrix(images_test_set, dim(images_test_set)[1]*dim(images_test_set)[2], dim(images_test_set)[3]))


# préparation
temp <- matrix(nrow = 0, ncol = 10)
vecY_test <- matrix(nrow = 0, ncol = 10)
for(i in 1:10000){
  temp <- matrix(-1, 1, 10)
  temp[labels_test_set[i] + 1] <- 1
  vecY_test <- rbind(vecY_test, temp)
}

y <- labels_training_set

## multinomiale
library(glmnet)
# tune <- cv.glmnet(x,y,alpha=0)
# plot(tune)
x
y
temporary <- x[1:1000,]
temporary2<- y[1:1000]
# temporary
# trouver le meilleur lambda grâce à la cross-validation
# Ridge
multinomial_reg <- cv.glmnet(temp, temp2, family = "multinomial", type.measure = "class", alpha = 0)  
plot(multinomial_reg)
pred = predict(multinomial_reg, images_test_set, s=multinomial_reg$lambda.min, type="class")
pred
mean(labels_test_set[1:1000] == pred)

# Lasso
binomimale_reg <- cv.glmnet(temp, temp2, family = "multinomial", type.measure = "class", alpha = 1)  
plot(binomimale_reg)
pred = predict(binomimale_reg, images_test_set, s=binomimale_reg$lambda.min, type="class")
pred
mean(labels_test_set[1:1000] == pred)


#####################
## Binomiale ##
###############

x <- images_training_set
y <- labels_training_set
temp <- matrix(nrow = 0, ncol = 10)
vrai_y <- matrix(nrow = 0, ncol = 10)
for(i in 1:1000){
  temp <- matrix(-1, 1, 10)
  temp[labels_training_set[i] + 1] <- 1
  vrai_y <- rbind(vrai_y, temp)
}



temp0 <-matrix(nrow = 0, ncol=10)
temp1 <-matrix(nrow = 0, ncol=10)
temp2 <-matrix(nrow = 0, ncol=10)
temp3 <-matrix(nrow = 0, ncol=10)
temp4 <-matrix(nrow = 0, ncol=10)
temp5 <-matrix(nrow = 0, ncol=10)
temp6 <-matrix(nrow = 0, ncol=10)
temp7 <-matrix(nrow = 0, ncol=10)
temp8 <-matrix(nrow = 0, ncol=10)
temp9 <-matrix(nrow = 0, ncol=10)


temp0 <- cv.glmnet(x[1:1000,], vrai_y[1:1000,1], family = "binomial", alpha = 1)    
temp1 <- cv.glmnet(x[1:1000,], vrai_y[1:1000,2], family = "binomial", alpha = 1)    
temp2 <- cv.glmnet(x[1:1000,], vrai_y[1:1000,3], family = "binomial", alpha = 1)    
temp3 <- cv.glmnet(x[1:1000,], vrai_y[1:1000,4], family = "binomial", alpha = 1)    
temp4 <- cv.glmnet(x[1:1000,], vrai_y[1:1000,5], family = "binomial", alpha = 1)    
temp5 <- cv.glmnet(x[1:1000,], vrai_y[1:1000,6], family = "binomial", alpha = 1)    
temp6 <- cv.glmnet(x[1:1000,], vrai_y[1:1000,7], family = "binomial", alpha = 1)    
temp7 <- cv.glmnet(x[1:1000,], vrai_y[1:1000,8], family = "binomial", alpha = 1)    
temp8 <- cv.glmnet(x[1:1000,], vrai_y[1:1000,9], family = "binomial", alpha = 1)    
temp9 <- cv.glmnet(x[1:1000,], vrai_y[1:1000,10], family = "binomial", alpha = 1)    


probabilities0 <- predict(temp0, x[1:1000,], family = "binomial", s=temp1$lambda.min, type="class")
probabilities1 <- predict(temp1, x[1:1000,], family = "binomial", s=temp2$lambda.min, type="class")
probabilities2 <- predict(temp2, x[1:1000,], family = "binomial", s=temp2$lambda.min, type="class")
probabilities3 <- predict(temp3, x[1:1000,], family = "binomial", s=temp2$lambda.min, type="class")
probabilities4 <- predict(temp4, x[1:1000,], family = "binomial", s=temp2$lambda.min, type="class")
probabilities5 <- predict(temp5, x[1:1000,], family = "binomial", s=temp2$lambda.min, type="class")
probabilities6 <- predict(temp6, x[1:1000,], family = "binomial", s=temp2$lambda.min, type="class")
probabilities7 <- predict(temp7, x[1:1000,], family = "binomial", s=temp2$lambda.min, type="class")
probabilities8 <- predict(temp8, x[1:1000,], family = "binomial", s=temp2$lambda.min, type="class")
probabilities9 <- predict(temp9, x[1:1000,], family = "binomial", s=temp2$lambda.min, type="class")


# Model accuracy
observed.classes <- vrai_y[1:1000]
mean(probabilities0 == observed.classes)
mean(probabilities1 == observed.classes)
mean(probabilities2 == observed.classes)
mean(probabilities3 == observed.classes)
mean(probabilities4 == observed.classes)
mean(probabilities5 == observed.classes)
mean(probabilities6 == observed.classes)
mean(probabilities7 == observed.classes)
mean(probabilities8 == observed.classes)
mean(probabilities9 == observed.classes)


# mean(labels_test_set[1:1000] == probabilities1)
predicted.classes <- probabilities2 #ifelse(probabilities1 > 0, "1", "-1")
#probabilities <- ridge %>% predict(x[1:1000,])
probabilities <- predict(ridge, x[1:1000,])
head(probabilities)

tune <- cv.glmnet(x[1:1000,],y[1:1000,1],alpha=0)
plot(tune)

ridge <- glmnet(x[1:1000,],y[1:1000],lambda=tune$lambda.1se,alpha=0)
head(ridge$beta)


