# -*- coding: utf-8 -*-
"""
Created on Mon May 16 16:08:13 2022

@author: syj47
"""

# 실습 03
# 실습 02에서 경사하강법의 반복 횟수에 따른 매개변수의 변화를 그래프에 
# 표시하고, 수렴한 뒤 최적 매개변수의 값을 구하라.
# 결과물: 코드, 그래프, 최적 매개변수

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
        
        w0_list.append(w0)
        w1_list.append(w1)
        w2_list.append(w2)
        CEE_list.append(CEE)
   
        if i%10000 == 0:
            print("epoch : {0:2}  w0 = {1:.5f}, w1 = {2:.5f}, w2 = {3:.5f} CEE = {4:.5f}".format(i, w0, w1, w2, CEE))
     
      
    print("----" * 20)
    print("\nepoch : {0:2}  w0 = {1:.5f}, w1 = {2:.5f}, w2 = {3:.5f} CEE = {4:.5f}".format(i, w0, w1, w2, CEE))
    print('\nfinal CEE : {} \noptimal w0 : {} \noptimal w1 : {} \noptimal w2 : {}'.format(CEE, w0, w1, w2))
        
    
GDM(x, y, label, lr=0.03, epochs=200000)

plt.figure()
plt.plot(w0_list, 'bx-', markerfacecolor='none', markevery=10000)
plt.plot(w1_list, 'r^-', markerfacecolor='none', markevery=10000)
plt.plot(w2_list, 'go-', markerfacecolor='none', markevery=10000)
plt.legend([r'$w_0$', r'$w_1$', r'$w_2$'])
plt.ylabel(r'$w_0, w_1, w_2$')
plt.grid(True)
plt.title(r'$\alpha = 0.03$')