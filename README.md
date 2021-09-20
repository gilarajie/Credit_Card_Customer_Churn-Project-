# Credit Card Customer Churn Prediction
In this repository, My team and I build a machine learning model that is used to predict which customers will churn on credit card services. we use the BankChurn.csv dataset obtained from Kaggle.
(link: https://www.kaggle.com/sakshigoyal7/credit-card-customers?select=BankChurners.csv. 
The case in this dataset consists of 10,000 customers specifying age, salary, marital status, credit card limit, credit card category, etc. There are almost 18 features. There are 16.07% for the negative class of customers who have churn which makes a binary classification with an unbalanced proportion of target variables. In this dataset there are several features that are used as parameters to make predictions, including:
*	***CLIENTNUM***: customer account number.
*	***Attrition_Flag***: customer status (Existing and Attrited).
*	***Customer_Age***: age of the customer.
*	***Gender***: gender of customer (M for male and F for female).
*	***Dependent_count***: number of dependents of customers.
*	***Education_Level***: customer education level (Uneducated, High School, Graduate, College, Post-Graduate, Doctorate, and Unknown).
*	***Marital_Status***: customer's marital status (Single, Married, Divorced, and Unknown).
*	***Income_Category***: customer income interval category (Less than $40K, $40K-$60k, $60K-$80K, $80K-$120K, $120K +, and Unknown).
*	***Card_Category***: type of card used (Blue, Silver, Gold, and Platinum).
*	***Months_on_Book***: period of being a customer (in months).
*	***Total_Relationship_Count***: the number of products used by customers in the bank.
*	***Months_Inactive_12_mon***: period of inactivity for the last 12 months.
*	***Contacts_Count_12_mon***: the number of interactions between the bank and the customer in the last 12 months.
*	***Credit_Limit***: credit card transaction nominal limit in one period.
*	***Total_Revolving_Bal***: total funds used in one period.
*	***Avg_Open_To_Buy***: the difference between the credit limit set for the cardholder's account and the current balance.
*	***Total_Amt_Chng_Q4_Q1***: increase in customer transaction nominal between quarter 4 and quarter 1.
*	***Total_Trans_Amt***: total nominal transaction in the last 12 months.
*	***Total_Trans_Ct***: the number of transactions in the last 12 months.
***

### STEP 1
The first stage, we perform Exploratory Data Analysis by using several functions and graphical visualization. At this stage the dataset didn’t find any invalid data and duplicate data. Missing values in this data set have been accounted for with the status "**Unknown**". We found that the scale data wasn’t the same between one feature and another, where the range of minimum and maximum values between one feature and another wasn’t the same. In addition, in this dataset there are several features that have outlier values that can be seen through the boxplot. There are some features that have abnormal data distribution, for this we have to do a normality test on numeric variables. In this dataset there are features that have a very strong correlation, namely **Credit_Limit** and **Average_Open_to_Buy**.
***

### STEP 2
The next stage is Data Preprocessing with reference to the Exploration Data Analysis stage, the steps taken are as follows:

* Delete the Credit Limit feature
  
  Based on the heatmap results, there are two features that have a very strong correlation, namely the **Credit_Limit** and **Average_Open_to_Buy** features. All we have to do is delete one of the features is **Credit_Limit**. The assumption of this **Credit_Limit** feature is obtained based on bank regulations, while **Average_Open_to_Buy** is obtained based on customer behavior. In this project, we want to create a machine learning model to predict customer churn based on customer behavior, therefore we decided to remove **Credit_Limit** and include **Average_Open_to_Buy** in the modeling process.

* Data Scaling
  
  Based on the EDA results, there are several data features that aren’t normally distributed. We perform features that are normally distributed, namely **Customer_Age**, **Months_on_Book**, **Total_Trans_Ct**, and **Total_Ct_Chng_Q4_Q1**. Then normalize the features that aren’t normally distributed. Data scaling can also overcome features that have outliers.

* Categorical Coding
  
  At this stage we change the categorical feature to numeric, this is done because machine learning cannot read categorical data. There are two techniques that will be used, namely **Label Encoding** and **One Hot Encoding**. Label Encoding for features with ordinal measurement scales and One Hot Encoding for implementing features with nominal measurement scales.

* Split Data into Training and Test sets
  
  At this stage, we divide the dataset into training data and test data with a proportion of 80% for training data and 20% for test data. we use this training data to train the model and we will use the test data to test the performance of the model.

* Dataset Balancing
  
  In this dataset, there is a condition where the proportion between classes on the dependent variable of **Existing Customers** and **Attributed Customers** is not balanced. This will affect the model's ability to predict customer status on new data, where the possibility of the model when making predictions for the minority class is not as good as the majority class. Therefore, to balance the data classes we use the **SMOTE** technique, where this technique will add data in the minority class to a number of data in the majority class based on the nearest neighbor of the data point (this technique is similar to the KNN concept), so by using this technique the dataset is The resulting data is more varied and does not reduce the information from the dataset.
***

### STEP 3
The last stage, we do modeling using 3 algorithms, namely **Logistic Regression**, **Random Forest Classifier**, and **Gradient Boosting Classifier**. The parameters that we use to see the performance of the model are the values of Accuracy, Precision, Recall, and Cross Entropy Loss. We focus on the Accuracy Score to see the ability of the model to make predictions correctly for all the data being tested. The Precision Score is used to see the model's ability to correctly predict the positive class when compared to the overall result that is predicted as a positive class. The Recall Score is used to see the model's ability to correctly predict the positive class when compared to the overall data that actually has a positive class. We use Cross Entropy Loss to compute a score that summarizes the mean difference between the actual and predicted probability distributions for the positive prediction class. The ideal model has a Cross Entropy Loss value below the value of logK basis e (for the case of binary classification it has a value of K=2), where the value is 0.6931. If a model in the case of binary classification has a Cross Entropy Loss value below 0.6931, which indicates that our machine learning model has a lot of information, so the model has a greater chance of accurately predicting new data.
***
