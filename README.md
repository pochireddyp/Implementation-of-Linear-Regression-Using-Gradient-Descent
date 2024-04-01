# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import NumPy, pandas, and StandardScaler for numerical operations, data handling, and feature scaling, respectively.
2. Create a linear regression function using gradient descent to iteratively update parameters, minimizing the difference between predicted and actual values.
3.Load the dataset, extract features and target variable, and standardize both using StandardScaler for consistent model training. 
4. Apply the defined linear regression function to the scaled features and target variable, obtaining optimal parameters for the model.
5. Display the predicted value for the target variable based on the linear regression model applied to the new data.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: pochireddy.p
RegisterNumber:  212223240115
*\
```
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1, y, learning_rate=0.01, num_iters=1000):
    # Add a column of ones to X for the intercept term 
  X = np.c_[np.ones(len(X1)), X1]
  # Initialize theta with zeros
  theta = np.zeros(X.shape[1]).reshape(-1,1)
  for _ in range(num_iters):
    predictions = (X).dot(theta).reshape(-1, 1)
    errors = (predictions - y).reshape(-1,1)
    theta -= learning_rate* (1 / len(X1)) * X.T.dot(errors)
  return theta

data = pd.read_csv('50_Startups.csv', header=None)
print(data.head())
# Assuming the last column is your target variable 'y' and the preceding column 
X = (data.iloc[1:, :-2].values)
print(X)
X1=X.astype(float)
scaler = StandardScaler()
y = (data.iloc[1:,-1].values).reshape(-1,1)
print(y)
X1_Scaled = scaler.fit_transform(X1)
Y1_Scaled = scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)
theta = linear_regression(X1_Scaled, Y1_Scaled)

# Predict target value for a new data point
new_data = np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled = scaler.fit_transform(new_data)
prediction = np.dot(np.append(1, new_Scaled), theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")
```

## Output:
![image](https://github.com/pochireddyp/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150232043/55c769b5-d011-4cbe-aa58-de01312e2686)
![image](https://github.com/pochireddyp/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150232043/a6e8ca60-dbd6-49df-8d3f-bbc062a3cdcd)
![image](https://github.com/pochireddyp/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150232043/36a6d2b6-dc2e-4c67-bfa6-d8818f4be629)
![image](https://github.com/pochireddyp/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150232043/f6d69a22-32cd-4b20-a993-9623325f2dfe)
![image](https://github.com/pochireddyp/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150232043/ee83b3eb-d4a8-4eb0-9527-03f54a64a51d)
![image](https://github.com/pochireddyp/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150232043/028ff31f-f3e2-4ced-93a5-9c20b5975181)
![image](https://github.com/pochireddyp/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150232043/0926222a-049b-4176-aebd-2001ff6394d0)
![image](https://github.com/pochireddyp/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150232043/dafce160-17f5-47a5-8d0b-43c2521055a0)
![image](https://github.com/pochireddyp/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150232043/12dd1bcd-faf7-4af0-ba05-412a47e9c921)
![image](https://github.com/pochireddyp/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/150232043/3cc0b1d5-a1ea-4952-bda7-b0bef79e4f69)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
