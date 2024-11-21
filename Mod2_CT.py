#!/usr/bin/env python
# coding: utf-8

import sys
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def train_validate_test_split(df, target, seed=123):
    '''
    This function takes in a dataframe, the name of the target variable
    (for stratification purposes), and an integer for a setting a seed
    and splits the data into train, validate and test. 
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''
    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=seed, 
                                            stratify=df[target])
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=seed,
                                       stratify=train_validate[target])
    return train, validate, test

def get_metrics(model, X, y):
    '''
    get_metrics_bin will take in a sklearn classifier model, an X and a y variable and utilize
    the model to make a prediction and then gather accuracy, class report evaluations

    return:  a classification report as a pandas DataFrame
    '''
    y_pred = model.predict(X)
    accuracy = model.score(X, y)
    conf = confusion_matrix(y, y_pred)
    print('confusion matrix: \n', conf)
    print()
    class_report = pd.DataFrame(classification_report(y, y_pred, output_dict=True)).T
    tpr = conf[1][1] / conf[1].sum()
    fpr = conf[0][1] / conf[0].sum()
    tnr = conf[0][0] / conf[0].sum()
    fnr = conf[1][0] / conf[1].sum()
    print(f'''
    The accuracy for our model is {accuracy:.4}
    The True Positive Rate is {tpr:.3}, The False Positive Rate is {fpr:.3},
    The True Negative Rate is {tnr:.3}, and the False Negative Rate is {fnr:.3}
    ''')
    return class_report


# Load dataset
data = pd.read_csv('data.csv')

# Data exploration
sns.countplot(data, x='Genre', hue='Gender', hue_order = [1, 0])
plt.title('Counts of each Genre grouped by Gender (1=M, 0=F)')
plt.show()

sns.heatmap(data.corr(),annot=True)
plt.title('Correlation Heatmap')
plt.show()

sns.pairplot(data, hue='Gender', hue_order = [1,0])
plt.title('Pairplot between feature grouped by Gender (1=M, 0=F)')
plt.show()

sns.pairplot(data, hue='Genre')
plt.title('Pairplot between each feature grouped by Genre')
plt.show()


# Splitting data into train, validate, and test
train, validate, test = train_validate_test_split(data,'Genre', seed=123)

features = ['Age', 'Height', 'Weight', 'Gender']

X_train = train[features]
y_train = train.Genre

X_validate = validate[features]
y_validate = validate.Genre

X_test = test[features]
y_test = test.Genre

# Standardize features
scaler = StandardScaler()
Xtr_scaled = scaler.fit_transform(X_train)
Xv_scaled = scaler.fit_transform(X_validate)
Xtest_scaled = scaler.fit_transform(X_test)

print()
print('Value Counts (training data)')
print('-------------')
print(train.Genre.value_counts())
print()

#Setting baseline accuracy using most frequent class
train['baseline_prediction'] = 'Strategy'
baseline_accuracy = (train.Genre == train.baseline_prediction).mean()
print(f'Baseline accuracy: {round(baseline_accuracy, 2)}%')
print()
input('Press Enter')
print()

# Initializing and fitting the KNN model on training data
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(Xtr_scaled, y_train)

#training data report
print('Training data report: ')
print('---------------------------------------------------------------')
knn_report = get_metrics(knn,Xtr_scaled,y_train)
print(knn_report)
print()
input('Press Enter')
print()

#validate data report
print('Validate data report: ')
print('---------------------------------------------------------------')
knn_report = get_metrics(knn,Xv_scaled,y_validate)
print(knn_report)
print()
input('Press Enter')
print()

#test data report
print('Test data report: ')
print('---------------------------------------------------------------')
knn_report = get_metrics(knn,Xtest_scaled,y_test)
print(knn_report)
print()
input('Press Enter')
print()

# Prompt the user to enter each value individually
print('Favorite Video Game Genre prediction: ')
print('--------------------------------------')
try:
    age = float(input("Enter age (in years): "))
    height = float(input("Enter height (in inches): "))
    weight = float(input("Enter weight (in lbs): "))
    gender = float(input("Enter gender (0 for female, 1 for male): "))
except ValueError:
    print("Invalid input. Please enter numeric values.")
    sys.exit()

# Input as numpy array
input_features = np.array([[age, height, weight, gender]])
input_scaled = scaler.transform(input_features)

# Make prediction
prediction = knn.predict(input_scaled)
print()
print(f"The predicted favorite video game genre is: {prediction[0]}")



