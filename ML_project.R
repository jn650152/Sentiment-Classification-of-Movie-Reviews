Needed <- c("tm", "SnowballCC", "RColorBrewer", "ggplot2", "wordcloud", "biclust", "cluster", "igraph", "fpc")   
#install.packages(Needed, dependencies=TRUE)   
#install.packages("Rcampdf", repos = "http://datacube.wu.ac.at/", type = "source")   
require(NLP)
library(tm) 
library(SnowballC)
library(qdap)
library(ggplot2)  
library(wordcloud)
require(MASS)
library(klaR)
library(caret)
library(e1071)
library(verification)
library(ROCR)
#read data
cname1 <- file.path("/Users/jluo/Downloads/IMDB\ Review\ Data/imdb1/neg")
text1 <- Corpus(DirSource(cname1)) 
cname2 <- file.path("/Users/jluo/Downloads/IMDB\ Review\ Data/imdb1/pos") 
text2 <- Corpus(DirSource(cname2)) 
text <- c(text1,text2)


# preprocessing: 
text <- tm_map(text, removePunctuation)
text <- tm_map(text, removeNumbers) 
text <- tm_map(text, removeWords, stopwords("english")) 
text <- tm_map(text, stemDocument) #? not complete
text <- tm_map(text, stripWhitespace) 
text <- tm_map(text, removeWords, c("film","movie","one","two","charact","time","get","place","thing"))   
text <- tm_map(text, PlainTextDocument)

dtm <- DocumentTermMatrix(text) 
m <- as.matrix(dtm) 

#data discovery
#organize words according to the frequency 
freq1 <- sort(colSums(as.matrix(dtm)), decreasing=TRUE) #data discovery for sparse corpus
freq1

#plot data discovery
wf <- data.frame(word=names(freq1), freq=freq1)  
head(wf,10)
p <- ggplot(subset(wf, freq>1850), aes(word, freq))   # part of data discovey 
p <- p + geom_bar(stat="identity")   
p <- p + theme(axis.text.x=element_text(angle=45, hjust=1))   
p   

#Relationships Between Terms
findAssocs(dtm, "like", corlimit=0.90) # specifying a correlation limit of 0.9

#Word Clouds
set.seed(142)   
dark2 <- brewer.pal(6, "Dark2")   
wordcloud(names(freq1), freq1, max.words=99, rot.per=0.2, colors=dark2)  

#feature engineering
#select features 
dtms <- removeSparseTerms(dtm, 0.8)#remove features with less frequency
m1 <- as.matrix(dtms) 

#Applying the learning algorithm
attitude <- c(rep(0,1000), rep(1,1000))
attitude <- as.factor(attitude)

#knn
fit1 <- train(m1, attitude,
              method = "knn",
              trControl = trainControl(method = "cv",number=10,savePredictions = TRUE))
predict1 <- predict(fit1, m1)
fit1$resample
fit1_sd = sd(fit1$resample$Accuracy)
fit1_mean = mean(fit1$resample$Accuracy)
#NB
fit2 <- train(m1, attitude,
                 method = "nb",
                 trControl = trainControl(method = "cv",number=10,savePredictions = TRUE))
predict2 <- predict(fit2, m1)
fit2$resample
fit2_sd = sd(fit2$resample$Accuracy)
fit2_mean = mean(fit2$resample$Accuracy)

#Linear Discriminant Analysis
fit3 <- train(m1, attitude,
              method = "lda",
              trControl = trainControl(method = "cv",number=10,savePredictions = TRUE))
predict3 <- predict(fit3, m1)
fit3$resample
fit3_sd = sd(fit3$resample$Accuracy)
fit3_mean = mean(fit3$resample$Accuracy)

#svm
fit4 <- svm(m1, attitude, kernel = "linear", cross = 10);
predict4 <- predict(fit4, m1)
print(fit4)
summary(fit4)
c <- c(0.73, 0.73, 0.75, 0.755, 0.74, 0.75, 0.705, 0.81, 0.715, 0.805)
fit4_sd = sd(c)
fit4_mean = mean(c)
#evaluation 
# t-test set 0.05 as significance threshold 
t.test(fit1$resample$Accuracy,fit2$resample$Accuracy)# p-value = 0.01255, thus there is a difference between fit1 and fit2 
t.test(fit1$resample$Accuracy,fit3$resample$Accuracy)# p-value = 1.98e-06, thus there is a difference between fit1 and fit3 
t.test(fit1$resample$Accuracy,fit4$resample$Accuracy)# p-value = 5.897e-12, thus there is a difference between fit1 and fit4
t.test(fit2$resample$Accuracy,fit3$resample$Accuracy)# p-value =7.351e-05, thus there is a difference between fit2 and fit3 
t.test(fit2$resample$Accuracy,fit4$resample$Accuracy)# p-value = 3.34e-14, thus there is a difference between fit2 and fit4
t.test(fit3$resample$Accuracy,fit4$resample$Accuracy)# p-value =1.168e-13, thus there is a difference between fit3 and fit4

#roc
#fit1
mod1 <- verify(attitude==1, as.numeric(predict1)-1)
roc.plot(mod1, plot.thres = NULL)
#fit2
mod2 <- verify(attitude==1, as.numeric(predict2)-1)
roc.plot(mod2, plot.thres = NULL)
#fit3
mod3 <- verify(attitude==1, as.numeric(predict3)-1)
roc.plot(mod3, plot.thres = NULL)
#fit4
mod4 <- verify(attitude==1, as.numeric(predict4)-1)
roc.plot(mod4, plot.thres = NULL)

#confusion matrix
table1 <- table(attitude, predict1)
table1
table2 <- table(attitude, predict2)
table2
table3 <- table(attitude, predict3)
table3
table4 <- table(attitude, predict4)
table4

#AUC
pred1 <- prediction(as.numeric(predict1), attitude)
auc1.tmp <- performance(pred1,"auc")
auc1 <- as.numeric(auc1.tmp@y.values)
auc1

pred2 <- prediction(as.numeric(predict2), attitude)
auc2.tmp <- performance(pred2,"auc")
auc2 <- as.numeric(auc2.tmp@y.values)
auc2

pred3 <- prediction(as.numeric(predict3), attitude)
auc3.tmp <- performance(pred3,"auc")
auc3 <- as.numeric(auc3.tmp@y.values)
auc3

pred4 <- prediction(as.numeric(predict4), attitude)
auc4.tmp <- performance(pred4,"auc")
auc4 <- as.numeric(auc4.tmp@y.values)
auc4
#draw histogram
accuracies <- c(fit1_mean, fit2_mean, fit3_mean, fit4_mean)
sds <- c(fit1_sd, fit2_sd, fit3_sd, fit4_sd)
methods <- c("knn", "NB", "LDA", "SVM")
evaluation <- data.frame(methods, accuracies, sds)

ggplot(evaluation, aes(x = factor(methods), y=accuracies)) + geom_bar(stat="identity", position="dodge", fill="red") + geom_errorbar(aes(ymin=accuracies-sds, ymax=accuracies+sds), width=.3, color="darkblue")




evaluation <- matrix( c(fit1_mean, fit1_sd, fit2_mean, fit2_sd,fit3_mean, fit3_sd,fit4_mean, fit4_sd), nrow=2, ncol=4, byrow=FALSE)
evaluation <- data.frame(fit1_mean, fit1_sd, fit2_mean, fit2_sd,fit3_mean, fit3_sd,fit4_mean, fit4_sd);
barplot(evaluation, main="different algorithms", ylab= "mean and standard devariation",
        beside=TRUE, col=rainbow(5))
legend("topleft", c("mean","sd"), cex=0.6,bty="n", fill=rainbow(5));

