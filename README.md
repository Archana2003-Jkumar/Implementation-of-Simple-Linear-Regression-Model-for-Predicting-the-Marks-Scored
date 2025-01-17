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
### df.head()
![image](https://user-images.githubusercontent.com/93427594/235605833-a77c219f-4bed-4722-b52f-ea7ebdcc81cb.png)
### df.tail()
 ![image](https://user-images.githubusercontent.com/93427594/235606104-3e2877a0-5f5a-44ae-9c78-3ddce1c67868.png)
### Array value of x
 ![image](https://user-images.githubusercontent.com/93427594/235606413-a520d770-0509-41d9-a45c-c0ab484796cf.png)
### Array value of y
![image](https://user-images.githubusercontent.com/93427594/235606918-989c06c6-98a3-487f-b042-e57391c394bc.png)
### Values of y prediction
![image](https://user-images.githubusercontent.com/93427594/235607035-d284d20a-3e50-45c4-834c-5e4f5dadfb9b.png)
### Array values of y test
![image](https://user-images.githubusercontent.com/93427594/235607218-f06babf8-551f-444d-92b2-5b03a623059b.png)

### Training set graph
![image](https://user-images.githubusercontent.com/93427594/235607437-7b61c240-3621-46a1-a903-5104c9d52ed3.png)
### Test set graph
![image](https://user-images.githubusercontent.com/93427594/235607509-58f42675-0610-4c73-99c1-8a0ac14cf50c.png)
### Values of MSE,MAE,RMSE
![image](https://user-images.githubusercontent.com/93427594/235607565-01973870-3a90-4ee8-af80-b5e1963ea8cc.png)




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
