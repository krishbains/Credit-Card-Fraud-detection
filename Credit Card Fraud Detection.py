# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# %%

# Load the dataset to pandas data frame
credit_card_data = pd.read_csv('creditcard.csv')

# %%

#printing first 5 rows of the dataset
credit_card_data.head(5)
#printing last 5 rows of the dataset
credit_card_data.tail(5)
# %%
#Dataset information
credit_card_data.info()
#%%
#Check for missing values in the dataset in each column
credit_card_data.isnull().sum()
#%%
#check the distribution of legit transactions and  fraudulent transactions 0 is normal and 1 is fraudulent (we can't feed this data to the model yet, since 90% of data is 0)
credit_card_data['Class'].value_counts()

#This Dataset is highly unbalanced
#%%
# Separating the data for analysis
legit = credit_card_data[credit_card_data["Class"] == 0]
fraud = credit_card_data[credit_card_data["Class"]==1]

# Checking the distribution of classes in both datasets
print("Legitimate transactions: ", legit.shape)
print("Fraudulent transactions: ", fraud.shape)

#%%
# statistical measures of the data (refers to Amount col. in csv file)
legit.Amount.describe()
#%%
fraud.Amount.describe()
#%%
# Compare the values for both transactions
credit_card_data.groupby('Class').mean()
#%%
# Under-Sampling:- Build a sample dataset containing similar distribution of normal transactions and Fraudulent Transactions
#Number of fraudlent transactions --> 492
legit_sample = legit.sample(n=492)
#%%
# Concatenating two data frames (fraudulent and legit_sample) axis = 0 adds each value row wise from the datasets alternatively. if not mentioned, it'll add all legit_sample entries first and then the fraud ones.
new_dataset = pd.concat([legit_sample, fraud] ,axis = 0)
#%%
new_dataset.head()
#%%
new_dataset.tail()
#%%

new_dataset['Class'].value_counts()
#%%
new_dataset.groupby('Class').mean()
#%%
#Splitting data into features & Targets, axis = 1 refers to columns (Y axis)
X = new_dataset.drop(columns = 'Class', axis=1)
Y = new_dataset['Class']
#%%
print(X)
#%%
print(Y)
#%%
# Split data into Training data and Testing Data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2 ,stratify=Y, random_state = 2)
print(X.shape, X_train.shape,X_test.shape)
#%%
# Model Training

#we are going to use Logistic Regression

model = LogisticRegression()
#%%
#training the Logistic Reg model with Training data
model.fit(X_train, Y_train)
#%%
#Model Evaluation based on the Accuracy score
#accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
#%%
print('Accuracy score on training data : ', training_data_accuracy)
#%%
#Accuracy on test data
X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score on test :  ', testing_data_accuracy)
#%%
