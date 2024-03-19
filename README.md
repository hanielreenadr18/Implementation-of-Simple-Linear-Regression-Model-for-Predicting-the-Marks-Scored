# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
`````python
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Haniel Reena D R
RegisterNumber:  2305001008
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
*/
`````

## Output:
## Head:
![Screenshot (39)](https://github.com/hanielreenadr18/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/155225915/b3b64c80-391b-4438-8e04-9d65a7c70ab2)
## Tail:
![Screenshot (42)](https://github.com/hanielreenadr18/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/155225915/29f455d1-ff6a-4523-b5e5-789722ec2d8a)
## X and Y values split:
![Screenshot (41)](https://github.com/hanielreenadr18/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/155225915/d115a32d-eed2-4638-a89a-1b07c410d80d)
## Info:
![Screenshot (40)](https://github.com/hanielreenadr18/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/155225915/5586c735-74d6-4e41-8497-fe8b695dbd69)
## Y pred & X test:
![Screenshot (43)](https://github.com/hanielreenadr18/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/155225915/359e3c3d-8ef5-415c-bdb3-533e8bca2a48)
## MSE, MAE, RMSE:
![Screenshot (44)](https://github.com/hanielreenadr18/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/155225915/8f2a47fa-6ab0-453c-abc3-81ea9ed86ba2)








## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
