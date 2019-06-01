# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 15:35:00 2019

@author: DIKA
"""

   # -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 16:26:20 2019

@author: DIKA
"""

import numpy as np
import math 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing as pre
import tensorflow as tf

dataset = pd.read_csv('latihan.csv')





scale  = pre.MinMaxScaler(feature_range= (0,1))
x_train = scale.fit_transform(dataset.iloc[:12, :-1].values)
x_testing = scale.fit_transform(dataset.iloc[12:, :-1].values)
y_train = scale.fit_transform(dataset.iloc[:12, 10].values)
y_testing  = scale.fit_transform(dataset.iloc[12:, 10].values)
y_train = y_train.reshape(12,1)
y_testing = y_testing.reshape(12,1)



#seed3 = np.random.RandomState(181)
seed3 = np.random
bias = seed3.rand(20)

#seed4 = np.random.RandomState(918)
seed4= np.random
bias2 = seed4.rand(1)

#seed2 = np.random.RandomState(1721)
seed2 = np.random
weight1 = seed2.rand(len(x_train[0]),20) 
#seed1 = np.random.RandomState(121)
seed1 = np.random
weight2 = seed1.rand(20,1)


def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_der(x):
    return sigmoid(x) * (1- sigmoid(x))



error_x= []
lr = 0.2

    
for epoch in range (1):
    #forward pass1 
    fwPass = (np.dot(x_train, weight1) + bias )
    pred = sigmoid(fwPass)
    
    #forward pass2
    fwPass2 = (np.dot(pred, weight2) + bias2)
    output = sigmoid(fwPass2)
    #hittung error
    eror_asli = output - y_train
    errors = 0.5 * (np.square(output - y_train))
    
    
        #hitung rumus backward untuk output->hidden
    der_error =   output - y_train
    der_actv = sigmoid_der(fwPass2)
    der_cost = der_error * der_actv
        
    chain_weight = np.dot(pred.T, der_cost)
    chain_bias = der_error * der_actv * 1 
        
        #chain = []
        #hitung rumus backward untuk hidden -> input
    der_error2 =  np.dot(der_cost,weight2.T)
    der_actv2 = sigmoid_der(fwPass)
        
    chain_weight2 = np.dot(x_train.T, der_error2 * der_actv2 )
    chain_bias2 = der_error2 * der_actv2
        
        #update weight2 & bias2
    weight2 =  weight2 +  ( - lr * chain_weight )
        
       # for num in chain_bias:
    bias  = bias -  lr * chain_bias
        
        #update weight1 & bias1
    weight1 =  weight1 +  ( - lr * chain_weight2 )
        #for num in chain_bias2:
    bias2  = bias2 + (-  lr * chain_bias )
    
       
 
        
def MAPE(x , y):
    return (sum(abs((x-y))/x) *100 / 12)
    
#x_norm = denormalize(y_testing , np.sum(dataset.iloc[12:, 10].values, axis=0))
#y_norm = np.round(denormalize(output_test , np.sum(dataset.iloc[12:, 10].values, axis=0)))
x_norm = scale.inverse_transform(y_testing)
y_norm = scale.inverse_transform(output_test)
MAPE(x_norm , y_norm)

erorr = []
erorr.append(MAPE(x_norm , y_norm))



#testing prediksi
fwPass_test = (np.dot(x_testing, weight1) + bias )
pred_test = sigmoid(fwPass_test)
    
    #forward pass2
fwPass2_test = (np.dot(pred_test, weight2) + bias2)
output_test = sigmoid(fwPass2_test)
    
plt.ylabel("MAPE(%)")
plt.xlabel("EPOCH")
ax = plt.gca()
ax.set_xticks([0,1,2,3,4,5,6,7,8,9])
ax.set_xticklabels([200,400,600,800,1000,1200,1400,1600,1800,2000])
plt.plot(erorr,'ro-')
plt.show()