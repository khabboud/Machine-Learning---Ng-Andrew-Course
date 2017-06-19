# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 11:15:10 2016

@author: khabboud
Multiclass classification of numbers using oneVsAll Regularized Logistic Regression: NG Andrea Course 

"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D 
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.io import loadmat   # this is to load matlab (.mat) data
import scipy.optimize as opt  
import matplotlib.image as mpimg
import os
from scipy import stats
from DisplayImageFunction import displayData
from scipy.optimize import minimize
from scipy.optimize import check_grad
import numpy.matlib
#==============================================================================
data = loadmat('MultiClassNeuralNet-ex3/ex3/ex3data1.mat')  
#================Load raw data==========================================================
xRaw=data['X']
epslon=100
#meanRep=np.matlib.repmat(np.mean(xRaw, axis=1),xRaw.shape[1],1)
#StdRep=np.matlib.repmat(np.sqrt(np.var(xRaw, axis=1)),xRaw.shape[1],1)
#x=(xRaw-meanRep.T)/(epslon+StdRep.T)
#x=(xRaw-np.mean(xRaw, axis=0))/(epslon+np.sqrt(np.var(xRaw, axis=0)))
x = np.delete(xRaw,np.where( np.std(xRaw,axis =0) == 0), axis = 1)
x = (x -np.mean(x,axis=0) )/(np.std(x,axis=0))
#x=xRaw
#x=stats.zscore(xRaw, axis=0)
y=data['y']  # Pass or Fail class
#y.shape += (1,)
##======================Parameters========================================================
K=10;
m,n =x.shape
RegParam=0.9
X = np.insert(x, 0, values=np.ones(m), axis=1)
#X,n,m=mapFeature(x,power_order)
#initial_theta=np.zeros(n)
#===============Display data using displayData function in DisplayImageFunction file===================================================
rand_rows=np.random.randint(5000,size=100);
Grid_h, X_display_array=displayData(xRaw[rand_rows,0:xRaw.shape[1]],np.array([]))
#displayData(x[1400:1500,0:x.shape[1]],np.array([]))
#========================= Define The sigmoid function ===================================
def sigmoid(z):   
    sig=1/(1+np.exp(-z))
    sig.shape +=(1,)
    return sig
#==========================The cost function =============================================
def costFun(theta,RegParam,X,Y):
    m=X.shape[0]
    h_x=sigmoid(np.dot(X,theta))
    Jx=sum(-Y*np.log(h_x)-(1-Y)*np.log(1-h_x))/m \
       +RegParam*sum(theta[1:theta.shape[0]]**2)/(2*m)  # notice here we do not include the first theta0 by convention
    return Jx[0]
#==========================The Gradient =============================================    
def gradient(theta,RegParam,X,Y):
    m=X.shape[0]
    h_x=sigmoid(np.dot(X,theta))
    grad=((np.dot(X.T,h_x-Y))/m).T    
    reg_part=RegParam*theta[1:theta.shape[0]]/m
    #print(reg_part.shape,grad[0,1:grad.shape[1]].shape)
    grad[0,1:grad.shape[1]]=grad[0,1:grad.shape[1]]+reg_part     
    #grad2=np.sum(X.T*np.squeeze(h_x-Y)),axis=1)/m   
    #grad2[1:grad2.shape[0]]=grad2[1:grad2.shape[0]]+reg_part      
    # very important:  -1 is the index of the last element in array, (i.e., grad[0,-1])
    # but when dealing with intervals, 0:-1 is not the whole size, because intervals in 
    # python is [0,-1) closed, and open at the end, so it won't include the last element.  
    #grad2.shape +=(1,) 
    return grad[0] # grad2  
##=====================Obtain parameters that Minimizes the costfunction =======================================
##======================One VS All regularized Linear Regression ====================================================
#def oneVsAll(X,y,K,RegParam):
m,n =X.shape
theta=np.zeros([K,n])
#initial_theta=np.zeros(n)
initial_theta = np.random.randn(n)
initial_theta.shape +=(1,)    
for k in range(1,K+1):
    y_k=y.copy() #otherwise any change in y_k will change y, because 
                    #y_k and y will be names to the same variable (just pointers to the same memory location)       
    y_k[np.where(y!=k)[0]]=0
    y_k[np.where(y==k)[0]]=1
    #,method='TNC'
    fmin = minimize(fun=costFun, x0=initial_theta, args=(RegParam,X, y_k), options={'maxiter':100},method='Newton-CG', jac=gradient)        
    print(fmin.success,fmin.nit,fmin.message)
    theta[k-1,:] = fmin.x
    result=fmin
    #result = opt.fmin_tnc(func=costFun, x0=initial_theta, fprime=gradient, args=(RegParam,X, y_k))
    #theta[k-1,:]=result[0]
#    return theta, result
#thetaRes, result=oneVsAll(X,y,K,RegParam)
#check_grad(costFun, gradient, [1.5, -1.5],RegParam,X, y)
thetaRes=theta
P_Y=sigmoid(np.dot(X,thetaRes.T))
Y_predict=np.argmax(P_Y, axis=1)+1
predict_correct=np.zeros([m,1])
predict_correct[np.where(Y_predict-y==0)[0]]=1
prediction_accuracy=sum(predict_correct)/m
print ("prediction accuracy = ", prediction_accuracy[0]*100,"%")
#Errorindx=np.where(predict_correct==0)[0]
#Grid_h, X_display_array=displayData(xRaw[Errorindx[0:100],0:xRaw.shape[1]],np.array([]))
