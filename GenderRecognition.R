
### SOLUTION TO PROBLEM1: LOGISTIC REGRESSION ANALYSIS 
#1. LOAD LIBRARIES
library(modelr)
library(pscl)
library(caret)
library(InformationValue)
library(ROCR)
library(car)


#2. IMPORT DATA
#  Load data with variable names into the data frame "voicedata"
voicedata=read.table('voice.csv',header=T,sep=',')

#3. CHECK WHETHER OUR DATA CONTAINS MISSING VALUES OR NOT
sum(is.na(voicedata))

#4. CREATE VARIABLES TO STORE PREDICTORS
meanfreq = voicedata[,1]
sd = voicedata[,2]
median = voicedata[,3]
Q25 = voicedata[,4]
Q75 = voicedata[,5]
IQR = voicedata[,6]
skew = voicedata[,7] 
kurt = voicedata[,8]
sp.ent = voicedata[,9]
sfm = voicedata[,10]
mode = voicedata[,11]
centroid = voicedata[,12]
meanfun = voicedata[,13]
minfun = voicedata[,14]
maxfun = voicedata[,15]
meandom = voicedata[,16]
mindom = voicedata[,17]
maxdom = voicedata[,18]
dfrange = voicedata[,19]
modindx = voicedata[,20]
label = voicedata[,21]

#5. CREATE FOLDS FOR N-FOLD CROSS VALIDATION
set.seed(3)
folds <- crossv_kfold(voicedata, k = 5)

#6. CONVERT RESAMPLE OBJECT TO REGULAR DATA FRAME
train1=as.data.frame(folds$train[[1]])
train2=as.data.frame(folds$train[[2]])
train3=as.data.frame(folds$train[[3]])
train4=as.data.frame(folds$train[[4]])
train5=as.data.frame(folds$train[[5]])
test1=as.data.frame(folds$test[[1]])
test2=as.data.frame(folds$test[[2]])
test3=as.data.frame(folds$test[[3]])
test4=as.data.frame(folds$test[[4]])
test5=as.data.frame(folds$test[[5]])


#7. PERFORM MODEL SELECTION ON DATA SET TRAIN1

#7a. BACKWARD ELIMINATION BY MANUALLY DROPPING X VARIABLE
full=glm(label~.,data=folds$train[[1]],family="binomial")
summary(full)

m1=glm(label~.-IQR-centroid-dfrange-skew,data=folds$train[[1]],family="binomial")
summary(m1)

m2=glm(label~.-IQR-centroid-dfrange-skew-mindom,data=folds$train[[1]],family="binomial")
summary(m2)

m3=glm(label~.-IQR-centroid-dfrange-skew-mindom-meandom,data=folds$train[[1]],family="binomial")
summary(m3)

m4=glm(label~.-IQR-centroid-dfrange-skew-mindom-meandom-maxfun,data=folds$train[[1]],family="binomial")
summary(m4)

m5=glm(label~.-IQR-centroid-dfrange-skew-mindom-meandom-maxfun-maxdom,data=folds$train[[1]],family="binomial")
summary(m5)

m6=glm(label~.-IQR-centroid-dfrange-skew-mindom-meandom-maxfun-maxdom-meanfreq,data=folds$train[[1]],family="binomial")
summary(m6)

m7=glm(label~.-IQR-centroid-dfrange-skew-mindom-meandom-maxfun-maxdom-meanfreq-median,data=folds$train[[1]],family="binomial")
summary(m7)

m8=glm(label~.-IQR-centroid-dfrange-skew-mindom-meandom-maxfun-maxdom-meanfreq-median-sd,data=folds$train[[1]],family="binomial")
summary(m8)

m9=glm(label~.-IQR-centroid-dfrange-skew-mindom-meandom-maxfun-maxdom-meanfreq-median-sd-kurt,data=folds$train[[1]],family="binomial")
summary(m9)

m10=glm(label~.-IQR-centroid-dfrange-skew-mindom-meandom-maxfun-maxdom-meanfreq-median-sd-kurt-modindx,data=folds$train[[1]],family="binomial")
summary(m10)


#7b. BACKWARD ELIMINATION USING STEP() FUNCTION
step(full, direction="backward",trace=F)
backward=glm(formula = label ~ Q25 + Q75 + kurt + sp.ent + sfm + mode + meanfun + minfun + modindx, family = "binomial", data = folds$train[[1]])
summary(backward)

#7c. STEPWISE FORWARD SELECTION
base=glm(label~meanfreq, data=folds$train[[1]],family="binomial")
step(base, scope=list(upper=full,lower=~1),direction="forward",trace=F)
forward=glm(formula = label ~ meanfreq + meanfun + IQR + minfun + skew + sfm + sp.ent + modindx + mode, family = "binomial", data = folds$train[[1]])
summary(forward)


#7d. STEPWISE REGRESSION  
step(base, scope=list(upper=full,lower=~1),direction="both",trace=F)
both=glm(formula = label ~ meanfun + IQR + minfun + sfm + sp.ent + modindx + mode, family = "binomial", data = folds$train[[1]])
summary(both)


#8. COMPUTE AIC and MCFADDEN R2 FOR MODELS BUILD ABOVE

# declare variables for aic
aicm10 = NULL
aicbackward = NULL
aicforward = NULL
aicboth = NULL
# declare variables for mcfadden r2(pseudo r2)
mcr2m10 =NULL
mcr2backward = NULL
mcr2forward = NULL
mcr2both = NULL
# declare and define list of training data sets
trainlist=list(train1,train2,train3,train4,train5)

 for (i in 1:5)
 {
	#fit the model
   modelm10=glm(label~Q25+Q75+sp.ent+sfm+mode+meanfun+minfun, family = "binomial", data = trainlist[[i]])
   modelbackward=glm(formula = label ~ Q25 + Q75 + kurt + sp.ent + sfm + mode + meanfun + minfun + modindx, family = "binomial", data = trainlist[[i]])
   modelforward=glm(formula = label ~ meanfreq + meanfun + IQR + minfun + skew + sfm + sp.ent + modindx + mode, family = "binomial", data = trainlist[[i]])
   modelboth=glm(formula = label ~ meanfun + IQR + minfun + sfm + sp.ent + modindx + mode, family = "binomial", data = trainlist[[i]])
	
	#compute aic
   aicm10[i] = modelm10$aic   
   aicbackward[i] = modelbackward$aic
   aicforward[i] = modelforward$aic
   aicboth[i] = modelboth$aic
   
	# build null model
   nullmod = glm(label~1,data=trainlist[[i]],family="binomial")

    #compute mc fadden r2
   mcr2m10[i] = 1-logLik(modelm10)/logLik(nullmod)
   mcr2backward[i] = 1 -logLik(modelbackward)/logLik(nullmod)
   mcr2forward[i] = 1-logLik(modelforward)/logLik(nullmod)
   mcr2both[i] = 1-logLik(modelboth)/logLik(nullmod)   
 }
# Average AIC of the models
mean(aicm10)
mean(aicbackward)
mean(aicforward)
mean(aicboth)
# Average McFaddenr2 of the models
mean(mcr2m10)
mean(mcr2backward)
mean(mcr2forward)
mean(mcr2both)


#9. COMPUTE ACCURACY FOR MODELS BUILD ABOVE
# declare variables for finding accuracy
accm10 = NULL
accbackward = NULL
accforward = NULL
accboth = NULL
# declare and define list of test data sets
testlist= list(test1,test2,test3,test4,test5)
 for (i in 1:5)
 {
	#fit the model
   modelm10=glm(label~Q25+Q75+sp.ent+sfm+mode+meanfun+minfun, family = "binomial", data = trainlist[[i]])
   modelbackward=glm(formula = label ~ Q25 + Q75 + kurt + sp.ent + sfm + mode + meanfun + minfun + modindx, family = "binomial", data = trainlist[[i]])
   modelforward=glm(formula = label ~ meanfreq + meanfun + IQR + minfun + skew + sfm + sp.ent + modindx + mode, family = "binomial", data = trainlist[[i]])
   modelboth=glm(formula = label ~ meanfun + IQR + minfun + sfm + sp.ent + modindx + mode, family = "binomial", data = trainlist[[i]])
   
    #predict values
	predm10 = predict(modelm10, newdata=testlist[[i]], type="response")
	predm10acc = prediction(predm10, testlist[[i]]$label)
    
	#compute ACCURACY for predicting gender with the model
	perfm10acc <- performance(predm10acc, measure = "acc")
	
	#find index with maximum accuracy 
	ind = which.max( slot(perfm10acc, "y.values")[[1]] )
	accm10[i] = slot(perfm10acc, "y.values")[[1]][ind]		

	#Repeat the same for other models
	predbackward = predict(modelbackward, newdata=testlist[[i]], type="response")
	predbackwardacc = prediction(predbackward, testlist[[i]]$label)
    perfbackwardacc <- performance(predbackwardacc, measure = "acc")
	ind = which.max( slot(perfbackwardacc, "y.values")[[1]] )
	accbackward[i] = slot(perfbackwardacc, "y.values")[[1]][ind]		
	
	predforward = predict(modelforward, newdata=testlist[[i]], type="response")
	predforwardacc = prediction(predforward, testlist[[i]]$label)
    perfforwardacc <- performance(predforwardacc, measure = "acc")
	ind = which.max( slot(perfforwardacc, "y.values")[[1]] )
	accforward[i] = slot(perfforwardacc, "y.values")[[1]][ind]
	
	predboth = predict(modelboth, newdata=testlist[[i]], type="response")
	predbothacc = prediction(predboth, testlist[[i]]$label)
    perfbothacc <- performance(predbothacc, measure = "acc")
	ind = which.max( slot(perfbothacc, "y.values")[[1]] )
	accboth[i] = slot(perfbothacc, "y.values")[[1]][ind]
}
# Average accuracy of the models
mean(accm10)
mean(accbackward)
mean(accforward)
mean(accboth)


#10. PLOT ROC CURVE AND COMPUTE AUC FOR MODELS BUILD ABOVE
predm10 = predict(m10, newdata=test1, type="response")
predaucm10 <- prediction(predm10, test1$label)
perfm10 <- performance(predaucm10, measure = "tpr", x.measure = "fpr")
plot(perfm10, main="ROC plot of m10 model")
aucm10 <- performance(predaucm10, measure = "auc")
aucm10 <- aucm10@y.values[[1]]
aucm10

predbackward = predict(backward, newdata=test1, type="response")
predaucbackward <- prediction(predbackward, test1$label)
perfbackward <- performance(predaucbackward, measure = "tpr", x.measure = "fpr")
plot(perfbackward, main="ROC plot of backward model")
aucbackward <- performance(predaucbackward, measure = "auc")
aucbackward <- aucbackward@y.values[[1]]
aucbackward

predforward = predict(forward, newdata=test1, type="response")
predaucforward <- prediction(predforward, test1$label)
perfforward <- performance(predaucforward, measure = "tpr", x.measure = "fpr")
plot(perfforward, main="ROC plot of forward model")
aucforward <- performance(predaucforward, measure = "auc")
aucforward <- aucforward@y.values[[1]]
aucforward

predboth = predict(both, newdata=test1, type="response")
predaucboth <- prediction(predboth, test1$label)
perfboth <- performance(predaucboth, measure = "tpr", x.measure = "fpr")
plot(perfboth, main="ROC plot of both model")
aucboth <- performance(predaucboth, measure = "auc")
aucboth <- aucboth@y.values[[1]]
aucboth


#11. EXAMINE MULTICOLLINEARITY PROBLEM IN BEST MODEL
#compute vif statistics 
vif(backward)
#print correlation matrix
cor(cbind(Q25,Q75,kurt,sp.ent,sfm,mode,meanfun,minfun,modindx))


#12. ELIMINATE MULTICOLLINEARITY PROBLEM IN BEST MODEL
#Rebuild model removing sfm variable 
backward2=glm(formula = label ~ Q25 + Q75 + kurt + sp.ent + mode + meanfun + minfun + modindx, family = "binomial", data = train1)

#Rebuild model removing sp.ent variable
backward3=glm(formula = label ~ Q25 + Q75 + kurt + sfm + mode + meanfun + minfun + modindx, family = "binomial", data = train1)

#COMPUTE AIC and ACCURACY FOR NEW MODELS
#declare and define variables 
aicb2 = NULL
aicb3 = NULL
accb2 = NULL
accb3 = NULL
trainlist=list(train1,train2,train3,train4,train5)
testlist= list(test1,test2,test3,test4,test5)
 for ( i in 1:5)
 {
	#fit the model
   modelb2=glm(formula = label ~ Q25 + Q75 + kurt + sp.ent + mode + meanfun + minfun + modindx, family = "binomial", data = trainlist[[i]])
   modelb3=glm(formula = label ~ Q25 + Q75 + kurt + sfm + mode + meanfun + minfun + modindx, family = "binomial", data = trainlist[[i]])

   #compute aic
   aicb2[i] = modelb2$aic   
   aicb3[i] = modelb3$aic   
   
   #compute accuracy 
	predb2 = predict(modelb2, newdata=testlist[[i]], type="response")
	predb2acc = prediction(predb2, testlist[[i]]$label)
    perfb2acc <- performance(predb2acc, measure = "acc")
	ind = which.max( slot(perfb2acc, "y.values")[[1]] )
	accb2[i] = slot(perfb2acc, "y.values")[[1]][ind]
	
	predb3 = predict(modelb3, newdata=testlist[[i]], type="response")
	predb3acc = prediction(predb3, testlist[[i]]$label)
    perfb3acc <- performance(predb3acc, measure = "acc")
	ind = which.max( slot(perfb3acc, "y.values")[[1]] )
	accb3[i] = slot(perfb3acc, "y.values")[[1]][ind]
 }
#Average AIC of the models
mean(aicb2)
mean(aicb3)
#Average Accuracy of the models
mean(accb2)
mean(accb3)

#verify multicollinearity in new model
vif(backward2)
vif(backward3)


#13. EXAMINE INFLUENTIAL POINTS IN DATA SET
#compute cooks distance
cook = cooks.distance(backward3)
#plot cook distance plot
plot(cook,ylab="Cooks distances")
#combine data set with cooks distance
b=cbind(train1,cook)
#print all influential point observations
b[cook > 0.02, ]

#14. ELIMINATE INFLUENTIAL POINTS FROM DATA SET
train1Inf1=b[cook < 0.02, ]
#Re-build model on new data set w/o influential points
modelb3Inf1=glm(formula = label ~ Q25 + Q75 + kurt + sfm + mode + meanfun + minfun + modindx, family = "binomial", data = train1Inf1)
summary(modelb3Inf1)

# Compute Accuracy for new model
	predb3Inf = predict(modelb3Inf1, newdata=test1, type="response")
	predb3Infacc = prediction(predb3Inf, test1$label)
    perfb3Infacc <- performance(predb3Infacc, measure = "acc")
	ind = which.max( slot(perfb3Infacc, "y.values")[[1]] )
	accb3Inf = slot(perfb3Infacc, "y.values")[[1]][ind]		
accb3Inf


#########################################################################################################


### SOLUTION TO PROBLEM2: CLASSIFICATION AND REGRESSION TREE (CART) MODEL
#1. LOAD LIBRARIES
library(rpart)
library(rpart.plot)

#2. SHUFFLE THE DATA & SPLIT IT INTO TRAINING AND TESTING DATA SET
set.seed(7)
voicedata=voicedata[sample(nrow(voicedata)),]
#80% training and 20% testing
select.data= sample (1:nrow(voicedata), 0.8*nrow(voicedata))
train.data= voicedata[select.data,]
test.data= voicedata[-select.data,]
 
#3. BUILD THE CART MODEL
genderCART <- rpart(label ~ ., data=train.data, method='class')

#4. PLOT CLASSIFICATION TREE
rpart.plot(genderCART,type=3,digits=3,fallen.leaves=TRUE)

#5. COMPUTE CART MODEL ACCURACY
#compute predictions
predictCART = predict(genderCART, newdata = test.data, type = "class")
#generate confusion matrix
gender_CART<-table(test.data$label, predictCART)
gender_CART
#calculate Accuracy
CART_Accuracy=(gender_CART[1,1] + gender_CART[2,2])/sum(gender_CART)
CART_Accuracy


#########################################################################################################
### SOLUTION TO PROBLEM3: BOX PLOT ANALYSIS AND HYPOTHESIS TESTING

#1. LOAD LIBRARIES
library(psych)
library(BSDA)

#2. IMPORT DATA
#  Load data with variable names into the data frame "voicedata"
voicedata=read.table('voice.csv',header=T,sep=',')

#3. GENERATE BOX PLOT
y=voicedata$meanfreq
gender=voicedata$label
plot(y~gender)
plot(y~gender,xlab="Gender",ylab="Mean frequency")

#4. GET MALE AND FEMALE DATA 
attach(voicedata)
male=voicedata[which(label=='male'),]
female=voicedata[which(label=='female'),]

#5. FIND SAMPLE MEAN AND SAMPLE STANDARD DEVIATION
summary(male)
summary(female)
describe(male)
describe(female)

#6. PERFORM HYPOTHESIS TESTING ON TWO SAMPLE MEANS 
z.test(male$meanfreq,female$meanfreq, alternative = "greater", mu = 0, sigma.x= 0.03,sigma.y= 0.03, conf.level= 0.95)
