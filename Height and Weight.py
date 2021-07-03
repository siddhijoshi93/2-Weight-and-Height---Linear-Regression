# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 09:24:02 2021

@author: ADMIN
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv("C:/Users/ADMIN/Downloads/weight-height.csv")

#EDA
data.columns
data.info()
data.describe()

#checking missing values
data.isna().sum()

#checking outliers
plt.boxplot(data[['Height', 'Weight']])

#data visualization
#checking the count of male
sns.countplot(data['Gender'])

#relation between geder,height and weight
plt.figure(figsize=(16,10))
sns.scatterplot(x = data['Height'], y = data['Weight'], hue = data['Gender']);

#checking relation between height and gender
plt.figure(figsize=(16,10))
sns.kdeplot(x = data['Height'], hue = data['Gender'])

##checking relation between weight and gender
plt.figure(figsize=(16,10))
sns.kdeplot(x = data['Weight'], hue = data['Gender'])

df_encode = pd.get_dummies(data)
df_encode.head()

#splitting data
X = df_encode.drop('Weight', axis=1)
y = df_encode['Weight']

#splitting model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=100)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#predicting
y_pred= regressor.predict(X_test)
print(y_pred)

#accuracy

print(regressor.coef_)
print(regressor.intercept_)

from sklearn.metrics import r2_score,mean_squared_error
r2_score(y_pred,y_test)
#accuracy=0.8911397725943436

