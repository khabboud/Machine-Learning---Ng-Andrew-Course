# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 14:25:12 2017

@author: khabboud
Implementing neural networks classification algorithm  Ng Andrew Course 
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat   # this is to load matlab (.mat) data
from scipy.optimize import minimize
#================Load raw data==========================================================
data = loadmat('MultiClassNeuralNet-ex3/ex3/ex3data1.mat')  
weighs= loadmat('NeuralNetLearning-ex4/ex4/ex4weights.mat')

Theta1=weighs['Theta1']
Theta2=weighs['Theta2']
xRaw=data['X']
x=xRaw
#x = np.delete(xRaw,np.where( np.std(xRaw,axis =0) == 0), axis = 1)
#Theta1 = np.delete(Theta1,np.where( np.std(xRaw,axis =0) == 0), axis = 1)
#x = (x -np.mean(x,axis=0) )/(np.std(x,axis=0))
y=data['y']
L=10; # number of labels (classes)
m,n =x.shape
RegParam=1
X = np.insert(x, 0, values=np.ones(m), axis=1)
Y=np.zeros([len(y),L])
indx1=y.astype(int)
#------------ if zero to correspond to the first digit before 1 
#indx1[np.where(indx1==10)[0]]=0
#Y[np.arange(m),indx1[:,0]]=1
#----------- if zero to correspond to the last digit after 9
Y[np.arange(m),indx1[:,0]-1]=1
params_v=np.array(np.append(Theta1.ravel(),Theta2.ravel()))  # append the parameters into one vector
H_size=Theta1.shape[0]
In_size=Theta1.shape[1]-1
#========================= Define The sigmoid function ===================================
def sigmoid(z):   
    sig=1/(1+np.exp(-z))
    return sig
#==========================FeedForward function ==========================================
def feedfwd(params_v,In_size,H_size,L,X):
    m=X.shape[0]
    #---------Unroll the parameters
    Theta1=np.reshape(params_v[0:H_size*(In_size+1)],(H_size,In_size+1))
    Theta2=np.reshape(params_v[H_size*(In_size+1):],(L,H_size+1))
    #---------Feedforward to get activation units
    a_1=X.T
    a_2=sigmoid(np.dot(Theta1,X.T))  # activation units in the first hidden layer result is H_size X m matrix
    a_2= np.insert(a_2, 0, values=np.ones(m), axis=0)  # add the bias unit result is (H_size+1) X m matrix
    a_o=sigmoid(np.dot(Theta2,a_2)) # output layer units result is L X m  
    return a_o,a_2,a_1,Theta1,Theta2
    
#==========================The cost function =============================================
#parameters for the neural network are "unrolled" into the vector params_v and need to be converted back into the weight matrices. 
def costFun(params_v,In_size,H_size,L,X,Y,RegParam):
    [a_o,a_2,a_1,Theta1,Theta2]=feedfwd(params_v,In_size,H_size,L,X)
    h_x=a_o
    Jx = np.sum( np.sum((-Y.T*np.log(h_x)-(1-Y.T)*np.log(1-h_x)),axis=0 ))/m
    regularized_term=RegParam*(np.sum(Theta1[:,1:]**2)+np.sum(Theta2[:,1:]**2))/(2*m)
    return Jx+regularized_term
#========================The sigmoid gradient ============================================
def gradSigmoid(z):
    gradsig=sigmoid(z)*(1-sigmoid(z))
    return gradsig
#=======================Parameters (Weights) initialization ==============================
def randparam(param_size,H_size,In_size):
    E=np.sqrt(6)/np.sqrt(In_size+H_size)
    Theta_init= np.random.uniform(-E,E,param_size)
    return Theta_init    
#======================BackPropagation Algorithm to compute the gradient ========================================
def backprop(params_v,In_size,H_size,L,X,Y,RegParam):
    #---------Feedforward to get activation units
    [a_o,a_2,a_1,Theta1,Theta2]=feedfwd(params_v,In_size,H_size,L,X)
    #--------Computing errors
    Err_o=a_o-Y.T # result is L X m 
    Err_H=np.dot(Theta2.T,Err_o)*a_2*(1-a_2) # result is (H_size +1)X m  because (H_size+1)X L | L X m .* (H_size +1)X m 
    Err_H=np.delete(Err_H, 0, axis=0) # remove the bias error result is (H_size)X m
    #We don't have Err for the input layer!! no Err_x! 
    #--------Computing gradients 
    Delta2=np.dot(Err_o,a_2.T)/m # result is Lx(H_size+1)
    Delta1=np.dot(Err_H,a_1.T)/m # result is H_sizeX (n+1)
    #--------Add regularization term
    Delta2[:,1:]=Delta2[:,1:]+(RegParam*Theta2[:,1:])/m
    Delta1[:,1:]=Delta1[:,1:]+(RegParam*Theta1[:,1:])/m
    grad=np.array(np.append(Delta1.ravel(),Delta2.ravel()))  # append the gradients into one vector 
    return grad # gradient size = size of parameters unrolled vector (H_size * (n+1) )+(L*(H_size+1))
#======================Train the Neural Network ================================================
param_size=(H_size * (n+1) )+(L*(H_size+1))
initial_params= randparam(param_size,H_size,In_size)   
fmin = minimize(fun=costFun, x0=initial_params, args=(In_size,H_size,L,X,Y,RegParam), options={'maxiter':200},method='Newton-CG', jac=backprop)        
print(fmin.success,fmin.nit,fmin.message)
params_res=fmin.x # resulting parameters from training the network
#======================Use the trained neural network model to predict the outcome ========
#-----------------unrol the parameters and feed forward 
[a_o,a_2,a_1,Theta1,Theta2]=feedfwd(params_res,In_size,H_size,L,X)
y_pred = np.argmax(a_o, axis=0) + 1
prediction_accuracy=np.sum(y_pred == y.T)/m
print ("prediction accuracy = ", prediction_accuracy*100,"%")



#### ----------------- results: 
#fmin = minimize(fun=costFun, x0=initial_params, args=(In_size,H_size,L,X,Y,RegParam), options={'maxiter':100},method='Newton-CG', jac=backprop)        
#False 100 Maximum number of iterations has been exceeded.
#prediction accuracy =  99.6 %
#fmin = minimize(fun=costFun, x0=initial_params, args=(In_size,H_size,L,X,Y,RegParam), options={'maxiter':200},method='Newton-CG', jac=backprop)        
#False 74 Warning: CG iterations didn't converge.  The Hessian is not positive definite.
#prediction accuracy =  99.62 %
