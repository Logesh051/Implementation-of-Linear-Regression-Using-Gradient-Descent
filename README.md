# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Logesh.N.A
RegisterNumber:  212223240078
/*
Program to implement the linear regression using gradient descent.
Developed by: POZHILAN V D
RegisterNumber:  212223240118
*/
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def linear_regression(X1,y,learning_rate=0.01,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1,1)
        errors = (predictions-y).reshape(-1,1)
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
    
data=pd.read_csv('50_Startups.csv',header=None)
data.head()
X = (data.iloc[1:,:-2].values)
print(X)

X1=X.astype(float)
scaler = StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)

X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)

theta = linear_regression(X1_Scaled,Y1_Scaled)

new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled = scaler.fit_transform(new_data)
prediction = np.dot(np.append(1,new_Scaled),theta)
prediction = prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted Value:{pre}")
*/
```

## Output:
![Screenshot 2024-03-14 215110](https://github.com/Logesh051/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/144979188/4821fd89-a5cf-4667-8a2d-82bcb3020e1a)
![Screenshot 2024-03-14 215128](https://github.com/Logesh051/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/144979188/50af3f02-87ac-4271-9f7a-5618448579d3)
![Screenshot 2024-03-14 215144](https://github.com/Logesh051/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/144979188/d31ed1c8-12b0-4db1-a80b-1cd98badd1ef)
![Screenshot 2024-03-14 215201](https://github.com/Logesh051/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/144979188/691e2df6-6173-4768-8e35-7d3f79782608)
![Screenshot 2024-03-14 215219](https://github.com/Logesh051/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/144979188/534a060d-56ed-4d3d-bff4-4d13ef16cac6)
![Screenshot 2024-03-14 215234](https://github.com/Logesh051/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/144979188/155b78a7-3824-403d-84c9-0b143c026765)
![Screenshot 2024-03-14 215341](https://github.com/Logesh051/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/144979188/bb7c5790-4fa7-4956-b1a6-2d0fc25cac68)
![Screenshot 2024-03-14 215354](https://github.com/Logesh051/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/144979188/a42cba51-e51f-4bca-b6b9-fd64bf53820e)
![Screenshot 2024-03-14 215407](https://github.com/Logesh051/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/144979188/f6e5cdf2-2369-476a-bbf4-fb755fe8b22c)
![Screenshot 2024-03-14 215419](https://github.com/Logesh051/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/144979188/588a1d6c-e266-4727-9916-f9c518b93b4a)
![Screenshot 2024-03-14 215428](https://github.com/Logesh051/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/144979188/8d443858-1f3d-4fe8-bd3b-154206567952)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
