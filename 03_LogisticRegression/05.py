# -*- coding: utf-8 -*-
"""
Created on Mon May 16 17:16:06 2022

@author: syj47
"""

# 실습 05
# 실습 03에서 구한 모델의 결정경계를 실습 01에서 그린 그래프와 함께 표시하라.
# 결과물: 코드, 그래프


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

raw_data = pd.read_csv('logistic_regression_data.csv', names=['num', 'x', 'y', 'label'])
x = raw_data['x'].values
y = raw_data['y'].values
label = raw_data['label'].values

x_0 = []
x_1 = []
y_0 = []
y_1 = []

for i in range(len(x)):
    if label[i]==0:
        x_0.append(x[i])
        y_0.append(y[i])
    else:
        x_1.append(x[i])
        y_1.append(y[i])
        
w0_list = []
w1_list= []
w2_list = []
CEE_list = []
acc_list = []


# 경사하강법 진행 함수
def GDM(x, y, label, lr, epochs):
    
    w0 = np.random.uniform(low=-1.0, high=1.0)
    w1 = np.random.uniform(low=-1.0, high=1.0)
    w2 = np.random.uniform(low=-1.0, high=1.0)
    
    for i in range(epochs):
        predict = 1/(1+np.exp(-(w0*x + w1*y + w2)))
        CEE = -(np.sum(label*np.log(predict+1e-7)+(1-label)*np.log(1-predict+1e-7)))/len(label)
        
        if CEE < 0.0001:
            break

        w0 = w0 - lr * ((predict-label)*x).mean()
        w1 = w1 - lr * ((predict-label)*y).mean()
        w2 = w2 - lr * (predict-label).mean()
            
   
        if i%10000 == 0:
            print("epoch : {0:2}  w0 = {1:.5f}, w1 = {2:.5f}, w2 = {3:.5f} CEE = {4:.5f}".format(i, w0, w1, w2, CEE))
     

    print("----" * 20)
    print("\nepoch : {0:2}  w0 = {1:.5f}, w1 = {2:.5f}, w2 = {3:.5f} CEE = {4:.5f}".format(i, w0, w1, w2, CEE))
    print('\nfinal CEE : {} \noptimal w0 : {} \noptimal w1 : {} \noptimal w2 : {}'.format(CEE, w0, w1, w2))
    
    decision_boundary = -(w0*x)/w1 - w2/w1
    plt.figure()
    plt.plot(x, decision_boundary, 'k', label='Decision Boundary')
    plt.plot(x_0, y_0, 'ro', label='0')
    plt.plot(x_1, y_1, 'bx', label='1') 
    plt.legend(loc='upper left')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.title('class')
    plt.show()
    
GDM(x, y, label, lr=0.03, epochs=200000)