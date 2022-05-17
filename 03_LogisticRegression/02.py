# -*- coding: utf-8 -*-
"""
Created on Tue May  3 15:10:39 2022

@author: syj47
"""

# 실습 02
# 평균 교차엔트로피오차를 비용함수로 사용하는 경사하강법을 구현하고, 
# 경사하강법의 반복 횟수에 따른 평균 교차엔트로피 오차의 변화를 그래프에 표시하라.
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
plt.plot(CEE_list, 'ko-', markerfacecolor='none', markevery=10000)
plt.legend(['CEE'])
plt.xlabel('Step')
plt.ylabel('CEE')
plt.grid(True)
plt.show()