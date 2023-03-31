# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages.
2. Display the values predicted using scatter plot and predict.
3. Plot the graph according to the given input.
4. End the program.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: J.Archana priya
RegisterNumber:  212221230007
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error 
df=pd.read_csv('/student_scores.csv')
df.head()

x = df.iloc[:,:-1].values
x
#segregating data to variables

y= df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)

y_pred
y_test

plt.scatter(x_train,y_train,color="green")
plt.plot(x_train,reg.predict(x_train),color="brown")
plt.title("Hours Vs Scores(Training set")
plt.xlabel ("Hours")
plt.ylabel ("Scores")
plt.show()

plt.scatter(x_test,y_test,color="black")
plt.plot(x_test,reg.predict(x_test),color="magenta")
plt.title("Hours Vs Scores(Test set")
plt.xlabel ("Hours")
plt.ylabel ("Scores")
plt.show()

mse = mean_squared_error(y_test,y_pred)
print("MSE = ",mse)

mae = mean_absolute_error(y_test,y_pred)
print("MAE = ",mae)

rmse = np.sqrt(mse)
print("RMSE = ",rmse)
```

## Output:
![e2](https://user-images.githubusercontent.com/93427594/229067440-558a9593-b90c-4b41-a091-53ca3dcf8814.png)

![e21](https://user-images.githubusercontent.com/93427594/229067570-aeb46c9d-fd9b-4106-ae71-eea1c8123e02.png)

![Screenshot 2023-03-31 135446](https://user-images.githubusercontent.com/93427594/229067891-edafa1ce-7c2a-4a0f-823c-192b53a77320.png)

![Screenshot 2023-03-31 135507](https://user-images.githubusercontent.com/93427594/229067912-d328278c-559e-4162-bef4-b40d7cdd8a68.png)

![Screenshot 2023-03-31 135522](https://user-images.githubusercontent.com/93427594/229067932-bbf5c1f0-4e31-4c8c-8342-cc2db058d596.png)

![Screenshot 2023-03-31 135539](https://user-images.githubusercontent.com/93427594/229067959-2059b5f5-9243-4f21-8117-c7430b01f0ad.png)


![Screenshot 2023-03-31 135557](https://user-images.githubusercontent.com/93427594/229067984-38643270-05d9-4336-a93b-31045702736e.png)

![Screenshot 2023-03-31 135617](https://user-images.githubusercontent.com/93427594/229067999-07af7d3b-62f9-4f10-a407-719b29037dab.png)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
