# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 14:25:12 2017

@author: khabboud
Regularized linear regression and checking Bias and Variance through cross validation 
and splitting the data to three sets: training (60%) cross validation (20%), test sets (20%)
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat   # this is to load matlab (.mat) data
from scipy.optimize import minimize
#================Load raw data==========================================================
data = loadmat('BiasVariance-ex5/ex5/ex5data1.mat')  
xRaw=data['X']
m =xRaw.shape[0]
Y=data['y']
XcvRaw=data['Xval']
Ycv=data['yval']
XtestRaw=data['Xtest']
Ytest=data['ytest']
#------plot training data 
plt.plot(xRaw,Y, 'o')
#----- plot all data 
#plt.plot(np.append(np.append(X,Xcv),Xtest),np.append(np.append(Y,Ycv),Ytest),'o')
#======================Parameters========================================================
X = np.insert(xRaw, 0, values=np.ones(m), axis=1)
Xcv= np.insert(XcvRaw, 0, values=np.ones(XcvRaw.shape[0]), axis=1)
Xtest= np.insert(XtestRaw, 0, values=np.ones(XtestRaw.shape[0]), axis=1)
m,n =X.shape
theta_initial=np.reshape(np.ones(2),[1,n])
Max_iteration=50;
RegParam=1
#==========================The Gradient =============================================    
def gradient(theta,RegParam,X,Y):
    m,n =X.shape
    theta=np.reshape(theta,[1,2])
    grad=(np.dot(X.T,(np.dot(X,theta.T)-Y))/m).T    # this has dimension of 1Xn
    grad[0,1:]=grad[0,1:]+RegParam*theta[:,1:]/m
    grad=np.squeeze(grad)
    return grad
###==================Regularized cost function (mean square error) + regularization 
def costFun(theta,RegParam,X,Y): #theta has to be a two dimensional array (NOT (2,))
    m,n =X.shape
    theta=np.reshape(theta,[1,n])
    Jx=np.sum((np.dot(X,theta.T)-Y)**2)/(2*m)+RegParam*np.sum(theta[:,1:]**2)/(2*m)
    return Jx
#======================Train linear regression model ================================================
def LR(theta_initial,RegParam,X,Y,Max_iteration):
    fmin = minimize(fun=costFun, x0=theta_initial, args=(RegParam,X,Y), options={'maxiter':Max_iteration},method='Newton-CG', jac=gradient)        
    print(fmin.success,fmin.nit,fmin.message)
    params_res=fmin.x # resulting parameters from training the network
    return params_res
#======================Training and CrossValidation Cost for different Training sets ==================
def TrainingValidCost(RegParam,X,Y,Xcv,Ycv,Max_iteration,theta_initial):
    Jtrain=np.zeros(m-2+1)
    Jcv=np.zeros(m-2+1)
    print(Jtrain.shape,Jcv.shape)
    for M in range(2,m+1):
        Xtrain=X[0:M,:]
        Ytrain=Y[0:M,:]
        params_res=LR(theta_initial,RegParam,Xtrain,Ytrain,Max_iteration)
        print(M-2)
        #-----------Training cost without regularization 
        Jtrain[M-2]=costFun(params_res,0,Xtrain,Ytrain)
        #-----------Cross validation cost without regularization 
        Jcv[M-2]=costFun(params_res,0,Xcv,Ycv)   
    return Jtrain,Jcv
#===================================================================================
params_res=LR(theta_initial,RegParam,X,Y,Max_iteration)
#----sort the x-axis 
x_axis=np.sort(xRaw.T)
#----get the indices for the sorted x-axis
sort_indx=np.argsort(xRaw.T)
#----predict y based on the trained linear regression model with the optimized parameters
y_predict=np.dot(X,params_res.T)
#----the y-axis should correspond to the sorted x-axis
y_axis=y_predict[sort_indx]
plt.plot(x_axis.T,y_axis.T,'-*')
plt.show()
#========================Plotting the Learning Curve for a certain RegParam ========
#---------------Training and Cross Validation Errors for different training set size ======
Jtrain,Jcv=TrainingValidCost(RegParam,X,Y,Xcv,Ycv,Max_iteration,theta_initial)
plt.plot(np.arange(2,13,1),Jtrain,'o-',label='Training Error')
plt.plot(np.arange(2,13,1),Jcv,'*-',label='Cross Validation Error')
plt.ylabel('Training/Cross Validation Error')
plt.xlabel('Training set size (m)')
plt.title('Regularization parameter= %d'%RegParam)
plt.legend()
plt.show()