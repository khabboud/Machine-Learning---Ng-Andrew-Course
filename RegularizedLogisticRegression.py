# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 10:03:30 2016

@author: khabboud
Implementing Regularized Logistic Regression: NG Andrea Course 

"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import cm
import matplotlib.pyplot as plt
import scipy.optimize as opt  
from scipy.optimize import minimize
import os
#==============================================================================
path = os.getcwd() + '\LogisticRegression-ex2\ex2\ex2data2.txt'  
data = np.genfromtxt(path, delimiter=',')
#================Load raw data==========================================================
x=data[:,[0,1]] # Exam1 Score Exam2 Score
y=data[:,2]  # Pass or Fail class
y.shape += (1,)
plt.ylabel("Test2 Result")
plt.xlabel("Test1 Result")
plt.scatter(x[np.where(y==0)[0],0], x[np.where(y==0)[0],1], s=100, c="red", alpha=0.5, label='Rejected' )
plt.scatter(x[np.where(y==1)[0],0], x[np.where(y==1)[0],1], s=60, marker="s", c="yellow", alpha=0.5, label='Admitted' )
plt.legend(loc='lower right', numpoints=1, ncol=3, fontsize=8) 
#======================Quadratic Map Features ====================================================
# number of terms is equal to the number of ordered 2-restricted integer partition of a number that is less than or equal to power order (6 here) including zero as part
# equals 0 + 1 + 2 +... + (6+1) = 28 
def mapFeature(x,power_order):
    m = max(x.shape)
    n = sum(np.arange(0,power_order+2,1))
    X=np.zeros([m,n])
    count=0
    for i in range(0,power_order+1):
        for j in range(0,i+1):
            X[:,count]=x[:,0]**(i-j)*x[:,1]**(j)
            count=count+1            
    return X,n,m  #outputs the X feature matrix, n the number of features, and m the size of training dataset
#========================= Define The sigmoid function ===================================
def sigmoid(z):   
    sig=1./(1+np.exp(-z))
    sig.shape +=(1,)
    return sig
#==========================The cost function =============================================
def costFun(theta,RegParam,X,Y):
    Jx=sum(-Y*np.log(sigmoid(np.dot(X,theta.T)))- \
       (1-Y)*np.log(1-sigmoid(np.dot(X,theta.T))))/(max(X.shape)) \
       +RegParam*sum(theta[1:len(theta)]**2)/(2*m)  # notice here we do not include the first theta0 by convention
    return Jx[0]
#==========================The Gradient =============================================    
def gradient(theta,RegParam,X,Y):
    m=max(X.shape)  
    grad=(np.dot(X.T,(sigmoid(np.dot(X,theta.T))-Y))/m).T  
    grad[0,1:max(grad.shape)]=grad[0,1:max(grad.shape)] \
                      +RegParam*theta[1:len(theta)]/m
    # very important:  -1 is the index of the last element in array, (i.e., grad[0,-1])
    # but when dealing with intervals, 0:-1 is not the whole size, because intervals in 
    # python is [0,-1) closed, and open at the end, so it won't include the last element.                     
    return grad
#======================Parameters========================================================
power_order=6
RegParam=0.9;
X,n,m=mapFeature(x,power_order)
initial_theta=np.zeros(n)
#=====================Obtain parameters that Minimizes the costfunction =======================================
result = opt.fmin_tnc(func=costFun, x0=initial_theta, fprime=gradient, args=(RegParam,X, y))  
thetaRes=result[0]
fmin = minimize(fun=costFun, x0=initial_theta, args=(RegParam,X, y), method='TNC', jac=gradient)        
theta= fmin.x
minCost1=costFun(theta,RegParam,X, y) 
minCost=costFun(thetaRes,RegParam,X, y) 