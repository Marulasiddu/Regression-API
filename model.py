import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import requests
import json

# importing the dataset
dataset = pd.read_csv('Employee_info.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# splitting the dataset into training set and testing set.
# from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)


# Fitting Simple Linear Regression to the training set.
# from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# Predicting the test set results.
y_pred = regressor.predict(X_test)

pickle.dump(regressor, open('model.pkl', 'wb'))
model = pickle.load(open('model.pkl', 'rb'))
print(model.predict([[1.8]]))









