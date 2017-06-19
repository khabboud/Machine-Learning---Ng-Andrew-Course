# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 09:17:11 2016

@author: khabboud
Implementing Logistic Regression: NG Andrea Course 

"""
import numpy as np
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import cm
import matplotlib.pyplot as plt
import scipy.optimize as opt  
import os
#==============================================================================
path = os.getcwd() + '\LogisticRegression-ex2\ex2\ex2data1.txt'  
data = np.genfromtxt(path, delimiter=',')
#================Load raw data ===============================================
x=data[:,[0,1]] # Exam1 Score Exam2 Score
y=data[:,2]  # Pass or Fail class
y.shape += (1,)
plt.ylabel("Exam2 Score")
plt.xlabel("Exam1 Score")
plt.scatter(x[np.where(y==0)[0],0], x[np.where(y==0)[0],1], s=100, c="red", alpha=0.5, label='Rejected' )
plt.scatter(x[np.where(y==1)[0],0], x[np.where(y==1)[0],1], s=60, marker="s", c="yellow", alpha=0.5, label='Admitted' )
#======================Parameters========================================================
m, n = x.shape # len(x) not necessarly true
Ident=np.ones([m,1])
X=np.concatenate((Ident, x),axis=1)
initial_theta=np.zeros(n+1) 
Iteration=1000;
#========================= Define The sigmoid function ===================================
def sigmoid(z):   
    sig=1./(1+np.exp(-z))
    sig.shape +=(1,)
    return sig
#==========================The cost function =============================================
def costFun(theta,X,Y):
    Jx=sum(-Y*np.log(sigmoid(np.dot(X,theta.T)))- \
       (1-Y)*np.log(1-sigmoid(np.dot(X,theta.T))))/(max(X.shape))
    return Jx[0]
#==========================The Gradient =============================================    
def gradient(theta,X,Y):
    m=max(X.shape)  
    grad=(np.dot(X.T,(sigmoid(np.dot(X,theta.T))-Y))/m).T  
    return grad

result = opt.fmin_tnc(func=costFun, x0=initial_theta, fprime=gradient, args=(X, y))  
thetaRes=result[0]
minCost=costFun(thetaRes,X, y) 
#=====================Plot decision Boundary======================
# The decision Boundary is where the points belong equally to either class, 
#i.e. sigmoid(thetaT X)=0.5 which correspond to Theta.T X =0  (Only need 2 points to define a line, so choose two endpoints)
x_vals = np.array([min(X[:,1])-2,  max(X[:,1])+2]);
#Calculate the decision boundary line from  equating Theta.T X to zero
# x2= -1/theta3 * (theta1+theta2 x2)
y_vals = (-1/thetaRes[2])*(thetaRes[1]*x_vals + thetaRes[0]);
plt.plot(x_vals, y_vals, label="Decision Boundary")
plt.legend(loc='lower right', numpoints=1, ncol=3, fontsize=8)
#================================================================================
### what's the probability of a student passing if he gets in exam1 45 and exam2 85
x0=np.array([[1,45,85]])
P_passing=sigmoid(np.dot(x0,result[0].T))