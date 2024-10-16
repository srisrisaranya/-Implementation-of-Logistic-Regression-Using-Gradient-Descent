# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Load the dataset.
3. Define X and Y array.
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary.
6. Define a function to predict the Regression value.


## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: SARANYA S
RegisterNumber:  212223110044
*/
```
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
data=pd.read_csv('/content/Placement_Data_Full_Class (1).csv')
data
```
![image](https://github.com/user-attachments/assets/10340e55-c103-4e7e-982e-3123ad2ff57e)
```
data.head()
```
![image](https://github.com/user-attachments/assets/88902f8c-9258-435d-a08d-8fb10631147b)
```
data.tail()
```
![image](https://github.com/user-attachments/assets/d490db12-64a8-48d0-8b2d-4013daa8f948)
```
data.info()
```
![image](https://github.com/user-attachments/assets/63859368-e6f7-45a0-b675-89e609449170)
```
data.drop('sl_no',axis=1)
```
![image](https://github.com/user-attachments/assets/9c09222c-45e1-4fd6-815a-693e1224f13c)

```
data["gender"]=data["gender"].astype('category')
data["ssc_b"]=data["ssc_b"].astype('category')
data["hsc_b"]=data["hsc_b"].astype('category')
data["hsc_s"]=data["hsc_s"].astype('category')
data["degree_t"]=data["degree_t"].astype('category')
data["workex"]=data["workex"].astype('category')
data["specialisation"]=data["specialisation"].astype('category')
data["status"]=data["status"].astype('category')
data.info()
data.dtypes
```
![image](https://github.com/user-attachments/assets/45f427fc-9cc4-4783-86ff-aa269c3975aa)
```
data["gender"]=data["gender"].cat.codes
data["ssc_b"]=data["ssc_b"].cat.codes
data["hsc_b"]=data["hsc_b"].cat.codes
data["hsc_s"]=data["hsc_s"].cat.codes
data["degree_t"]=data["degree_t"].cat.codes
data["workex"]=data["workex"].cat.codes
data["specialisation"]=data["specialisation"].cat.codes
data["status"]=data["status"].cat.codes
data
```
![image](https://github.com/user-attachments/assets/02a8ca2a-efd9-449a-9ae8-ed76dbd16447)

```
x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values
x
y
```
![image](https://github.com/user-attachments/assets/eb95a00e-4991-482c-bbe9-f0ca315095cb)
```
import numpy as np
theta=np.random.randn(x.shape[1])
y=y
def sigmoid(z):
  return 1/(1+np.exp(-z))
def loss(theta,x,y):
  h=sigmoid(x.dot(theta))
  return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))
def gradient_descent(theta,x,y,alpha,num_iterations):
  m=len(y)
  for i in range(num_iterations):
    h=sigmoid(x.dot(theta))
    gradient=x.T.dot(h-y)/m
    theta -= alpha*gradient
  return theta
theta=gradient_descent(theta,x,y,alpha=0.01,num_iterations=1000)
def predict(theta,x):
  h=sigmoid(x.dot(theta))
  y_pred=np.where(h>=0.5,1,0)
  return y_pred
y_pred=predict(theta,x)
accuracy=np.mean(y_pred.flatten()==y)
print("accuracy:",accuracy)
```
![image](https://github.com/user-attachments/assets/f0d2fb75-d724-4cf5-b597-819adebc9319)

```
xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0,3]])
y_prednew=predict(theta,xnew)
print(y_prednew)
```
![image](https://github.com/user-attachments/assets/7e0a97d8-c135-40fa-8d7d-81d4afcfd951)

```
print(theta)
```
![image](https://github.com/user-attachments/assets/251c0fc4-14bc-4c9f-9b9f-dcfa3ae68dea)








## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

