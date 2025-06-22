'''
practice scikit learn
https://www.youtube.com/watch?v=pqNCD_5r0IU
https://www.youtube.com/watch?v=7eh4d6sabA0
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import openpyxl as ox
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import accuracy_score

boston = pd.read_csv('C:/Users/erwin/PycharmProjects/Practice ScikitLearn/boston_house_prices.csv')

#features
X = boston[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']]
y = boston[['MEDV']]

print("X: ", X)
print(X.shape)
print("y: ", y)
print(y.shape)


#plot
plt.scatter(boston['CRIM'], y)
#plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#model
model = linear_model.LinearRegression()

#train
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("Predictions: ", predictions)
print("R^2: ", model.score(X, y))
print("coedd: ", model.coef_)
print("intercept: ", model.intercept_)

