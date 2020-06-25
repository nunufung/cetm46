# Online Shoppers Intention

library(ggplot2)
library(tidyverse)
library(gmodels)
library(ggmosaic)
library(corrplot)
library(caret)
library(rpart)
library(rpart.plot)
library(fpc)
library(data.table)
library(cluster)
library(gridExtra)
library(GGally)
library(caret)
library(data.table)
library(ggpubr)
library(ROSE)
library(class)
library(tree)
library(dtree)
library(randomForest)
library(mltools)
library(rsample)
library(e1071)
library(pheatmap)
library(keras)
library(dummies)
library(mlbench)
library(reticulate)
library(dplyr)
library(infotheo)
library(praznik)
set.seed(2019)

## Loading the dataset
df <- read.csv("https://github.com/nunufung/cetm46/raw/master/online_shoppers_intention.csv")
df2 <- read.csv("https://github.com/nunufung/cetm46/raw/master/online_shoppers_intention.csv")

# Checking number of rows and columns.
ncol(df)
nrow(df)

# Summary of data structure with HEAD(), STR(), and SUMMARY().
head(df,5)
str(df)
summary(df)

# Review distribution of target variable.
summary(df$Revenue)
CrossTable(df$Revenue)

# Creating a binary dependent variable for potential regression models.
df <- df %>%
  mutate(Revenue_binary = ifelse(Revenue == "FALSE",0,1))

# Review and check distribution of target variable.
hist(df$Revenue_binary)

summary(df$Revenue_binary)

# Checking the missing values:
colSums(is.na(df))

colSums(df == "")

# Because these are data from website visits, we see no missing values. 
# Each data point is created by a person interacting in some way with the website, 
# with no interactions being a '0'

## default theme for ggplot2
theme_set(theme_gray())

## setting default parameters for mosaic plots
mosaic_theme = theme(axis.text.x = element_text(angle = 90,
                                                hjust = 1,
                                                vjust = 0.5),
                     axis.text.y = element_blank(),
                     axis.ticks.y = element_blank())


# We see that 76.7 percent of our guests come on the weekday, 
# a five-day cycle, with a 14.9 percent chance of purchasing something 
# and 23.3 percent on weekends, a two-day cycle, 
# with a 17.4 percent risk of purchasing.

# EDA Summary, In our empirical variables we see very little variance, 
# and in our categorical mild variability. 
# Based on this, we will attempt a classification through a decision tree 
# algorithm and clustering through an algorithm k-means.


################################
# Data Preparation for Analysis#
################################

# Converting our Categorical Variables to Ordinal Factors.
# Converting our variables to factors with ordered levels 
# (ordinal variables) for use with various algorithms:

df$OperatingSystems <- factor(df$OperatingSystems, order = TRUE, levels = c(6,3,7,1,5,2,4,8))
df$Browser <- factor(df$Browser, order = TRUE, levels = c(9,3,6,7,1,2,8,11,4,5,10,13,12))
df$Region <- factor(df$Region, order = TRUE, levels = c(8,6,3,4,7,1,5,2,9))
df$TrafficType <- factor(df$TrafficType, order = TRUE, levels = c(12,15,17,18,13,19,3,9,1,6,4,14,11,10,5,2,20,8,7,16))

# Changing Month and Visitor Type to ordinal variables and assigning numbers to the levels for clustering.
library(plyr)
df$Month <- factor(df$Month, order = TRUE, levels =c('Feb', 'Mar', 'May', 'June','Jul', 'Aug', 'Sep','Oct', 'Nov','Dec'))
df$Month_Numeric <-mapvalues(df$Month, from = c('Feb', 'Mar', 'May', 'June','Jul', 'Aug', 'Sep','Oct', 'Nov','Dec'), to = c(1,2,3,4,5,6,7,8,9,10))


# Creating Appropriate Dummy Variables, 
# We convert the variable weekend to a dummy, 
# with weekend being a ‘1’ and a weekday being a ‘0’
df <- df %>%
  mutate(Weekend_binary = ifelse(Weekend == "FALSE",0,1))

# Normalizing Numerical Data

# Certain machine learning algorithms 
# (such as SVM and K-means) are more sensitive to the scale of data 
# than others since the distance between the data points is very important.

# Running this formula through the data in the column does the following: 
# it takes every observation one by one, subtracts the smallest value 
# from the data. Then this difference is divided by the difference between 
# the largest data point and the smallest data point, which in turn scales 
# it to a range [0;1].

# Logically, the rescaled value of the smallest data point will be 0 and 
# the rescaled value of the largest data point will be 1.

normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}

## Creating a copy of the original data.
shopper_data_norm <- df

## Normalizing our 10 variables.
shopper_data_norm$Administrative <- normalize(df$Administrative)
shopper_data_norm$Administrative_Duration <- normalize(df$Administrative_Duration)
shopper_data_norm$Informational <- normalize(df$Informational_Duration)
shopper_data_norm$Informational_Duration <- normalize(df$Administrative)
shopper_data_norm$ProductRelated <- normalize(df$ProductRelated)
shopper_data_norm$ProductRelated_Duration <- normalize(df$ProductRelated_Duration)
shopper_data_norm$BounceRates <- normalize(df$BounceRates)
shopper_data_norm$ExitRates <- normalize(df$ExitRates)
shopper_data_norm$PageValues <- normalize(df$PageValues)
shopper_data_norm$SpecialDay <- normalize(df$SpecialDay)

# Finalizing our normalized dataframe for clustering models:
shopper_data_clust <- shopper_data_norm[-c(11,16:19)]

# Creating Test and Train Data
# Splitting the data into training and test datasets (80-20 split) for classification:
shopper_data_class <- df[-c(19:21)]

set.seed(1984)
training <- createDataPartition(shopper_data_class$Revenue, p = 0.8, list=FALSE)

train_data <- shopper_data_class[training,]
test_data <- shopper_data_class[-training,]

# We now have two specific data sets:
# shopper_data_class for our classification algorithms
# shopper_data_clust for our clustering algorithms

# Clustering
# Our data visualization suggests that there are no clear 
# distribution patterns among our variables and hence, 
# clustering might a good sorting Algorithm for our needs. 
# It will look at the data and try to find groupings.

# K-Means Clustering, Data we are feeding to our clustering models:
summary(shopper_data_clust)

str(shopper_data_clust)

# Running the K-Means model:
# We are asking the model to group our data into two groups (or centers) 
# to be able to predict ‘TRUE’ and ‘FALSE’ Revenue.

k_mean_clust <- kmeans(shopper_data_clust, centers = 2, iter.max = 100)

# Our findings:
## Size of our clusters
k_mean_clust$size

# ## Our cluster centers (means)
k_mean_clust$centers

# ## Between cluster sum of squares
k_mean_clust$betweenss

# ## Total cluster sum of squares
k_mean_clust$totss

# ## Whithin clusters sum of squares
k_mean_clust$betweenss / k_mean_clust$totss

# Within cluster sum of squares by cluster: (between_SS / total_SS = 24.0 %)
# Suggests the model is not very accurate at prediction.

#############################################
# Lets look at our K-Means Confusion Matrix:#
#############################################
t1 <- table(k_mean_clust$cluster, shopper_data_norm$Revenue)
t1


# We see that this iteration of the model did a good job with the 
# ‘TRUE’ values, with most of them being in group 2, 
# but did not sort the ‘FALSE’ values correctly.


# Visualizing our K-means Clusters
# Preforming a PCA
# We are going to create components from our existing data:
pca_cluster_data <- prcomp(shopper_data_clust[c(1:10)], scale. = TRUE)
plot(pca_cluster_data, main = "Principal Components")

# Picking out and plotting the first two components against each other:

shopper_components_data <- as.data.frame(pca_cluster_data$x)

## Show first two PCs for out shoppers
head(shopper_components_data[1:2], 5)

## Plotting
plot(PC1~PC2, data=shopper_components_data,
     cex = .1, lty = "solid")
text(PC1~PC2, data=shopper_components_data, 
     labels=rownames(shopper_data_clust[c(1:10)]),
     cex=.8)

# Finally comparing how our derived clusters compare:
plot(PC1~PC2, data=shopper_components_data, 
     main= "Online Shopper Intent: PC1 vs PC2 - K-Means Clusters",
     cex = .1, lty = "solid", col=k_mean_clust$cluster)
text(PC1~PC2, data=shopper_components_data, 
     labels=rownames(shopper_data_clust[c(1:10)]),
     cex=.8, col=k_mean_clust$cluster)

# We can see that our two components, have a lot of overlap (the red overlaps the black). 
# Indicating that perhaps our model isn't really reliable. 
# Next, we mathematically check the accuracy.

# Precision, Recall, and F1 Score
# Precision attempts to answer the following question:
# What proportion of positive identifications was actually correct?

# While Recall attempts to answer the following question:
# What proportion of actual positives was identified correctly?

# In order to fully assess a model 's effectiveness, 
# both precision and recall must be examined. Unfortunately, 
# there's often tension between precision and recall. 
# That is, precision enhancement typically reduces recall, 
# and vice versa. Different metrics have been established 
# that depend both on precision and one such metric is the recall F1 value. 
# The F1 score (also F-score or F-calculation) is a calculation of the 
# accuracy of a test in the statistical analysis of binary classification. 
# In calculating the score, it considers both the precision and 
# the recall of the test.

# Prediction
# Let’s look at the predictive power of this model.
# Below are our accuracy measures:

presicion_kmeans<- t1[1,1]/(sum(t1[1,]))
recall_kmeans<- t1[1,1]/(sum(t1[,1]))

## Precision
presicion_kmeans

## Recall
recall_kmeans

# And our K-Means F-Score:
F1_kmeans<- 2*presicion_kmeans*recall_kmeans/(presicion_kmeans+recall_kmeans)
F1_kmeans

# K-Medoids Clustering
# Since our data has many '0's, running a median-based clustering model may make sense. 
# Running Code K-Medoids:  In order to predict 'TRUE' and 'Fake' sales, 
# we ask the model to divide our data into two classes (or centres).

k_med_clust <- pam(x = shopper_data_clust, k = 2)

# Our findings: 
## Size of our clusters
k_med_clust$id.med

## Centers of our clusters (medians)
k_med_clust$mediods

## Objective Function
k_med_clust$objective

## Summary of our cluster
k_med_clust$clusinfo

################################################
# Let’s look at our K-Medoids Confusion Matrix:#
################################################
t1b <- table(k_med_clust$clustering, shopper_data_norm$Revenue)
t1b


# We do not see any clear sorting.
# Maybe visualizing can give us a better picture.

# Visualizing our clusters
# Preforming a PCA
# Again, visualizing against our principal components (derived in section 6.1):
plot(PC1~PC2, data=shopper_components_data, 
     main= "Online Shopper Intent: PC1 vs PC2 - K-Medoids Clusters",
     cex = .1, lty = "solid", col=k_med_clust$clustering)
text(PC1~PC2, data=shopper_components_data, 
     labels=rownames(shopper_data_clust[c(1:10)]),
     cex=.8, col=k_med_clust$clustering)

# We see, as with our K-means, a lot of overlap between our 
# two components (the red overlaps the black one). 
# Although the second main component tends to be best divided 
# (along the x-axis when PC1 is '0' or below). 
# This indicates the model may not be very accurate but may be significantly better than our clustering of K-means. 
# We step on mathematically to test the accuracy.


# Prediction
# Let’s look at the predictive power of this model.
# Below are our accuracy measures:

presicion_kmed<- t1b[1,1]/(sum(t1b[1,]))
recall_kmed<- t1b[1,1]/(sum(t1b[,1]))
## Precision
presicion_kmed

## Recall
recall_kmed

# K-Medoids F-Score:
F1_kmed<- 2*presicion_kmed*recall_kmed/(presicion_kmed+recall_kmed)
F1_kmed

# Summary

# We see that clustering by K-means give us a somewhat precise model (0.89) that has bad recall (0.51). This results in a low F-score of about 0.65.

# In contrast, clustering by K-medoids give us similarly precise model (0.89) though it still has a bad recall of 0.59 (this is slightly better than our K-means recall of 0.51) resulting in a low F-score of about 0.72 (though it is better than our k-means F-score of 0.65).

# We conclude that given our imbalanced data with only about 12000 observations (10422 FALSE against 1908 TRUE) we need more data to get a better F-Score from clustering algorithms.

# Classification
# Decision Tree, Data we are feeding to our clustering models:
summary(shopper_data_class)

str(shopper_data_class)

# Running the decision tree algorithm from the “rpart” library:
model_dt<- rpart(Revenue ~ . , data = train_data, method="class")
rpart.plot(model_dt)


# Our predictive model indicates page values greater than 0.99 contribute to 57 percent of the time being a TRUE. In addition, an efficient bounce rate above 0 boosts our TRUE to 75% and as an added point pages of form '5' or below (0,1,2,3,4,5) results in a Real 83% of the time. 
# The October and November months are good months for conversions by shoppers. 
# We should be focused on these three metrics as a web developer and designer; increasing the page ranking, decreasing the bounce rate and paying attention to the types of items listed on administrative form '5' pages and below. 
# Marketing can either 'double-down' to boost sales on October and November, or concentrate on other months t

# Prediction 
# Let’s look at the prediction of this model on the test dataset (test_data):
pred.train.dt <- predict(model_dt,test_data,type = "class")

######################################
# Our Decision Tree Confusion Matrix:#
######################################
t2<-table(pred.train.dt,test_data$Revenue)
t2

# Our accuracy measures:
presicion_dt<- t2[1,1]/(sum(t2[1,]))
recall_dt<- t2[1,1]/(sum(t2[,1]))
## Precision
presicion_dt

## Recall
recall_dt

#  F-Score:
F1_dt<- 2*presicion_dt*recall_dt/(presicion_dt+recall_dt)
F1_dt


# Summary
# We see that classifying by decision tree gave us a very precise model 
# (0.92) that also has good recall (0.95). This resulting F-Score of 0.94 
# suggests a high predictive power for our decision tree model.


# Concluding 
# For our specific set of variables, we found that due to the limitations of our datasets, 
#the Decision tree was better able to predict the shoppers purchasing intent than the Clustering models. 

# The Decision Tree had a higher F-score of 0.94, despite the small number of observations and variables,
#while the Clustering model had an F-score of 0.72. With the Decision Tree we were able to predict that, 
#during October and November, the customer was more likely to make a purchase. We can also increase 
#the chances of sales by concentrating on three metrics; increase the relevance of the website, 
#decrease the bounce rate and pay attention to the types of products listed on administrative pages