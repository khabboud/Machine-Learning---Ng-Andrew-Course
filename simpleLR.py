# -*- coding: utf-8 -*-
"""
Created on Wed Sept 28 17:26 2016

@author: khabboud 
Implementing simple Linear Regression (LR): NG Andrea Course 
"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
#import pandas as pd 
from matplotlib import cm
import matplotlib.pyplot as plt
import os  
#======================Parameters========================================================
ulfa=0.1;
Theta=np.array([[0,0,0]])
#Theta.shape += (1,)
#Theta=Theta.T
Iteration=50;
accuracy=0.000001; # to check the difference between costs at consecutive iterations
#==============================================================================
path = os.getcwd() + '\LinearRegression-ex1\ex1\ex1data2.txt'  
#data = pd.read_csv(path, header=None, names=['Profit', 'Population'])  
#'\NYC collision DataSets\NYPD_Motor_Vehicle_Collisions.csv'
data = np.genfromtxt(path, delimiter=',')
#================Load raw data - One Dimentional Linear Regression ====================================
#x=data[:,0] #Population
#x.shape += (1,)
#y=data[:,1]  #profit
#y.shape += (1,)
#plt.ylabel("Profit")
#plt.xlabel("Population")
#m= max(x.shape) # len(x) not necessarly true
##colors = np.random.rand(m)
##area = np.pi * (15 * np.random.rand(m))**2  # radius
##plt.scatter(x, y, s=area, c=colors, alpha=0.5 )
#plt.plot(x, y, 'rx')
#============Load raw data - Multivariant Linear Regression
xRaw=data[:,[0,1]] # size and number of bedrooms
y=data[:,2]  #price of the house
y.shape += (1,)
m= max(xRaw.shape) # len(x) not necessarly true
Ident=np.ones([m,1])

# 3D Scatter plot for 2 dimensional LR  (2 features)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xRaw[:,0], xRaw[:,1], y, zdir='z', marker='x')
ax.set_xlabel('Size')
ax.set_xlim(-30,np.max(xRaw[:,0]))
ax.set_ylabel('Number of rooms')
ax.set_ylim(0, np.max(xRaw[:,1]))
ax.set_zlabel('Price')
ax.set_zlim(0, np.max(y))
#========Feature Scaling========================
x=(xRaw-np.mean(xRaw, axis=0))/(np.sqrt(np.var(xRaw, axis=0)))
#x=(xRaw-np.mean(xRaw, axis=0))/(np.max(xRaw, axis=0)-np.min(xRaw, axis=0))
XRaw=np.concatenate((Ident, xRaw),axis=1)
#======================Gradient Descent ==============================================
def GD(ulfa,X,Y,theta,iteration,Accuracy):
    m=max(X.shape)  #len(X) is wrong! it will give me 2
    J=np.zeros(iteration)
    for i in range (0,iteration):
        J[i]=costFun(X,Y,theta)        
        theta=theta-(ulfa*(np.dot(X.T,(np.dot(X,theta.T)-Y))/m)).T
        if i>0:
            if J[i]-J[i-1]>Accuracy:
                print("ERROR Cost is increasing.") 
    
    return theta,J
### The cost function (mean square error)
def costFun(X,Y,theta):
    Jx=sum((np.dot(X,theta.T)-Y)**2)/(2*max(X.shape))
    
    return Jx[0]
X=np.concatenate((Ident, x),axis=1)

#X=np.concatenate(([Ident.T], [x.T]), axis=0)

j=costFun(X,y,Theta)
[ThetaRes,Cost]=GD(ulfa,X,y,Theta,Iteration,accuracy)
minCost=costFun(X,y,ThetaRes)

# 3D plot for two dimensional LR (two features )
ax.plot(xRaw[:,0], xRaw[:,1], np.dot(X,ThetaRes.T).T[0], label='Multivariant Linear Regression',zdir='z')
ax.legend()
plt.show()
###plot for one  dimentional LR
#plt.plot(x, np.dot(X,ThetaRes.T))
plt.plot(np.arange(1,Iteration+1,1)-1,Cost)
plt.show()
#plotting the fitted y with the training data Ys
plt.plot(np.dot(X,ThetaRes.T).T[0])
plt.plot(y)
plt.show()
