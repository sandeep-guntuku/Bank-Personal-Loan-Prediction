# Bank-Personal-Loan-Prediction

## Project Description:

This project is about predicting likeliness of converting liability customers to personal loan customers using Logistic Regression, Random forest, KNN, Neural Networks, and Ensemble Models using “Bank Personal Loan Modelling”.
<br />Dataset from Kaggle (https://www.kaggle.com/krantiswalke/bank-personal-loan-modelling) based on 5000 observations with 14 explanatory variables.

## Goal: 

The project is aimed at implementing a model(s) to predict likeliness of converting liability customers to personal loan customers.
<br />•	Remove variables, build what is needed.
<br />•	Models: Logistic Regression, KNN techniques, RandomForest, Ensemble Learning & Neural Networks.
<br />•	Choose the best model having best accuracy.

## Business Problem:

This case is about a bank whose management wants to explore ways of converting its liability customers to personal loan customers (while retaining them as depositors). A campaign that the bank ran last year for liability customers showed a healthy conversion rate of over 9% success. This has encouraged the retail marketing department to devise campaigns with better target marketing to increase the success ratio with minimal budget.

## Data Exploration and Preprocessing:

<br />The Dataset contains data of 5000 customers with 14 explanatory variables. The data includes:
<br />•	Customer demographic information (age, income, etc.), 
<br />•	The customer's relationship with the bank (mortgage, securities account, etc.) 
<br />•	Customer response to the last personal loan campaign (Personal Loan).
<br />•	Among these 5000 customers, only 480 (= 9.6%) accepted the personal loan that was offered to them in the earlier campaign.

## Data Cleaning:

<br />•	Removed rows which were having unknown values for features like Zip code and ID.
<br />•	Dropped rows with Nan/Null values.
<br />•	Dropped index column
<br />•	Checking for outliers, data entry errors
<br />•	Apply abs for “Experience”

## Models and their comparison:
We have implemented the below models:
<br />•	Logistic Regression
<br />•	RandomForest
<br />•	Classification using K-Nearest Neighbors 
<br />•	Neural Networks
<br />•	Ensemble method 

## Reasons for specific Model Selection:
## Logistic Regression:

Since we are dealing with a classification problem and expect some linear relationships 
between variables, we will use a logistic regression model to classify our data. 

The Logistic Regression model on the testing data gives an accuracy value of 90.6%. 
 
## Classification using K-Nearest Neighbors:

KNN stands for K-Nearest Neighbors. It is a supervised learning algorithm. It is often used as a benchmark for more complex classifiers such as Artificial Neural Networks (ANN) and Support Vector Machines (SVM). 
We have used 14 independent features for KNN implementation. A robust implementation must consider feature engineering, data cleaning, and cross-validation. 
<br />•	K means clustering
<br />•	K = 3
<br />•	Sampling 80% of data for training the algorithms using random sampling 

We have implemented KNN with different optimal weights by changing k values and this time the accuracy we achieved is 99.2%. 
 
## Random Forest:

Random forests or random decision forests is an ensemble learning method for classification, regression and other tasks that operates by constructing a multitude of decision trees at training time.
<br />•	Type of randomforest: classification
<br />•	Number of trees: 500
<br />•	No. of variables tried at each split: 3

Accuracy: 99.2%

## Neural Network:

Neural networks are a class of machine learning algorithms used for complex patterns in datasets using multiple hidden layers and non-linear activation functions. They are also known as artificial neural networks (ANNs) or simulated neural networks (SNNs). 

We have implemented in our scenario and the accuracy we achieved for the testing set is 91.2%.

## Ensemble: Voting & Weighted

Ensemble methods are techniques that create multiple models and then combine them to produce improved results. Ensemble methods usually produces more accurate solutions than a single model would. We have implemented ensemble techniques with three models: Logistic Regression, Neural Network and KNN in this project. 

The accuracy we attained 99.2% for both voted and weighted models but the sensitivity is 99.45 for weighted Ensemble.

## Results:
Below is the accuracy for all the five models implemented in the project:
MODEL	ACCURRACY
<br />• Logistic Regression	90.6 %
<br />• KNN	99.2%
<br />• RandomForest	99.2 %
<br />• Neural Networks	91.2 %
<br />• Ensemble	Voting: 99.2 %
<br />• Weighted: 99.2 %

KNN and RandomForest performed the best with an accuracy of 99.2% followed by Ensemble and with an accuracy of 99.2%.

## Summary:

The aim of the bank is to convert their liability customers into loan customers. They want to set up a new marketing campaign; hence, they need information about the connection between the variables given in the data. Four classification algorithms were used in this project. From the implementation, it seems like KNN have the highest accuracy and we can choose that as our final model.
