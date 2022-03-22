# Supervised Machine Learning Homework - Predicting Credit Risk
In this project I built a machine learning model that attempts to predict whether a loan from LendingClub will become high risk or not.

## Background
LendingClub is a peer-to-peer lending services company that allows individual investors to partially fund personal loans as well as buy and sell notes backing the loans on a secondary market. LendingClub offers their previous data through an API.
I used this data to create machine learning models to classify the risk level of given loans. Specifically, I compared the Logistic Regression model and Random Forest Classifier.

## Process

### Retrieve the data
In the Generator folder in Resources, there is a GenerateData.ipynb notebook that will download data from LendingClub and output two CSVs:

2019loans.csv
2020Q1loans.csv

I used an entire year's worth of data (2019) to predict the credit risk of loans from the first quarter of the next year (2020).
Note: these two CSVs have been undersampled to give an even number of high risk and low risk loans. In the original dataset, only 2.2% of loans are categorized as high risk. To get a truly accurate model, special techniques need to be used on imbalanced data. Undersampling is one of those techniques. Oversampling and SMOTE (Synthetic Minority Over-sampling Technique) are other techniques that are also used.

### Preprocessing: Convert categorical data to numeric
I created a training set from the 2019 loans using pd.get_dummies() to convert the categorical data to numeric columns. Similarly, I create a testing set from the 2020 loans, also using pd.get_dummies(). Note! There are categories in the 2019 loans that do not exist in the testing set. 

### Consider the models
I created and compared two models on this data: a logistic regression, and a random forests classifier. Before I create the models, fit, and score the models, I was asked to make a prediction as to which model I think will perform better. 

### Fit a LogisticRegression model and RandomForestClassifier model
Create a LogisticRegression model, fit it to the data, and print the model's score. Do the same for a RandomForestClassifier. 

### Revisit the Preprocessing: Scale the data
The data going into these models was never scaled, an important step in preprocessing. Use StandardScaler to scale the training and testing sets. Before re-fitting the LogisticRegression and RandomForestClassifier models on the scaled data, I made another prediction about how I think scaling will affect the accuracy of the models. Write your predictions down and provide justification.
Fit and score the LogisticRegression and RandomForestClassifier models on the scaled data. How do the model scores compare to each other, and to the previous results on unscaled data? How does this compare to your prediction? 