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
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("/content/ex1.txt",header = None)

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of city(10,000s")
plt.ylabel("profit ($10,000")
plt.title("Profit Prediction")

def computeCost(x,y,theta):
  m=len(y)
  h=x.dot(theta)
  square_err=(h-y)**2

  return 1/(2*m) * np.sum(square_err)#returning
  
  data_n = data.values
m=data_n[:,0].size
x=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))

computeCost(x,y,theta)#call function

def gradientDescent(x,y,theta,alpha,num_iters):
  m=len(y)
  j_history=[]
  for i in range(num_iters):
    preds = x.dot(theta)
    error = np.dot(x.transpose(),(preds -y))
    descent = alpha * 1/m * error
    theta-=descent
    j_history.append(computeCost(x,y,theta))
  return theta,j_history


theta,j_history = gradientDescent(x,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+" +"+str(round(theta[1,0],2))+"x1")

plt.plot(j_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function")

def predict(x,theta):
  pred = np.dot(theta.transpose(),x)
  return pred[0]
pred1 = predict(np.array([1,3.5]),theta)*10000
print("Population = 35000 , we predict a profit of $"+str(round(pred1,0)))

pred2 = predict(np.array([1,7]),theta)*10000
print("Population = 70000 , we predict a profit of $"+str(round(pred2,0)))
```

## Output:
![Screenshot 2023-03-31 133122](https://user-images.githubusercontent.com/93427594/229062297-1ef196a6-b281-4bcc-a24e-742d241bb381.png)
![Screenshot 2023-03-31 133209](https://user-images.githubusercontent.com/93427594/229062328-89efc909-cc65-48e4-aff4-a94a9f7e4bfc.png)
![Screenshot 2023-03-31 133229](https://user-images.githubusercontent.com/93427594/229062349-ae652182-c682-4f09-947d-f20efbc91ac7.png)
![Screenshot 2023-03-31 133309](https://user-images.githubusercontent.com/93427594/229062393-fe5dff40-d3b2-4d66-8ba7-674aac8730ff.png)
![Screenshot 2023-03-31 133345](https://user-images.githubusercontent.com/93427594/229062411-fc4b4ba4-2fee-4271-96ed-b0e663073ca6.png)
![Screenshot 2023-03-31 133402](https://user-images.githubusercontent.com/93427594/229062167-1a79aef2-7c60-4098-9bbc-0e123542978d.png)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
