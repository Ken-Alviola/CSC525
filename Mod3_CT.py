#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings("ignore")

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures



# Load dataset
df = pd.read_csv('Salary_Data.csv')

X = df[['YearsExperience']]
y = df[['Salary']]

#splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)


# Visualizing the Linear Regression results
lin_reg = LinearRegression()
lin_reg.fit(X, y)
def viz_linear():
    plt.scatter(X, y, color='red')
    plt.plot(X, lin_reg.predict(X), color='blue')
    plt.title('Linear Regression')
    plt.xlabel('Years Experience')
    plt.ylabel('Salary')
    plt.show()
    return
viz_linear()


# Visualizing the Polymonial Regression results
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
pol_reg = LinearRegression()
pol_reg.fit(X_poly, y)

def viz_polymonial():
    plt.scatter(X, y, color='red')
    plt.plot(X, pol_reg.predict(poly_reg.fit_transform(X)), color='blue')
    plt.title('Polynomial 4 degree')
    plt.xlabel('Years Experience')
    plt.ylabel('Salary')
    plt.show()
    return
viz_polymonial()


#Baseline RMSE using mean or median

# 1. Predict pred_mean
y_train_mvp = y_train.copy()
Salary_pred_mean = y_train_mvp.Salary.mean()
y_train_mvp['Salary_pred_mean'] = Salary_pred_mean

# 2. compute pred_median
Salary_pred_median = y_train_mvp.Salary.median()
y_train_mvp['Salary_pred_median'] = Salary_pred_median

# 3. RMSE of pred_mean
rmse_train = mean_squared_error(y_train_mvp.Salary, y_train_mvp.Salary_pred_mean) ** .5
print("RMSE using Mean\nTrain/In-Sample: ", round(rmse_train, 2))

# 4. RMSE of pred_median
rmse_train = mean_squared_error(y_train_mvp.Salary, y_train_mvp.Salary_pred_median) ** .5
print("RMSE using Median\nTrain/In-Sample: ", round(rmse_train, 2))
print()

# Linear model
lm = LinearRegression()
lm.fit(X_train, y_train)

# Predictions
y_train_mvp['Salary_pred_lm'] = lm.predict(X_train)

# evaluate: rmse
rmse_train = mean_squared_error(y_train_mvp.Salary, y_train_mvp.Salary_pred_lm) ** (1/2)
print("RMSE for OLS using LinearRegression\nTraining/In-Sample: ", round(rmse_train,2))

# R2 for train
R2_train = explained_variance_score(y_train_mvp.Salary, y_train_mvp.Salary_pred_lm)
print("R2 score for LinearRegresssion\nIn-Sample Performance: ",round(R2_train,2))
print()

# 4 degree Polynomial model
pf4 = PolynomialFeatures(degree=4)

# fit and transform X_train_scaled
X_train_degree4 = pf4.fit_transform(X_train)

# create the model object
lm4 = LinearRegression()

# fit the model to our training data
lm4.fit(X_train_degree4, y_train)

# predict train
y_train_mvp['Salary_pred_lm4'] = lm4.predict(X_train_degree4)

# evaluate: rmse
rmse_train = mean_squared_error(y_train_mvp.Salary, y_train_mvp.Salary_pred_lm4) ** (1/2)
print("RMSE for Polynomial Model, degrees=4\nTraining/In-Sample: ", round(rmse_train,2))

# R2 for train
poly_R2 = explained_variance_score(y_train_mvp.Salary, y_train_mvp.Salary_pred_lm4)
print("R2 score for 4 degree Polynomial model\nIn-Sample Performance: ",round(poly_R2, 2))
print()

# Using poly model on test data
y_test_mvp = y_test.copy()

# predict on test
y_test_mvp['Salary_pred_lm4'] = lm4.predict(pf4.fit_transform(X_test))

# evaluate: rmse
rmse_test = mean_squared_error(y_test_mvp.Salary, y_test_mvp.Salary_pred_lm4)**(1/2)
print("RMSE for 4 degree Polynomial model on test data\nOut-of-Sample Performance: ", round(rmse_test,2))

# R2 for test
test_R2 = explained_variance_score(y_test_mvp.Salary, y_test_mvp.Salary_pred_lm4)
print("R2 score for 4 degree Polynomial model\nOut-of-Sample Performance: ", round(test_R2,2))
print()

# Prompt the user to enter a salary
print('Salary Prediction')
print('-----------------------')
try:
    Salary = float(input("Enter experience (in years): "))
except ValueError:
    print("Invalid input. Please enter numeric values.")
    sys.exit()

# Input as numpy array
input_features = np.array([[Salary]])
input_transform = pf4.fit_transform(input_features)


# Make prediction
prediction = lm4.predict(input_transform)
prediction = prediction[0][0]
print(f"The predicted Salary is: ${prediction:.2f}")






