# -*- coding: utf-8 -*-
"""
Created on Mon May 16 17:09:51 2022

@author: syj47
"""

# 실습 04
# 실습 03에서 구한 최적 매개변수로 구현한 로지스틱 회귀 모델의 훈련 정확도는?
# 결과물: 코드, 정확도

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

# 정확도를 구하는 함수
def accuracy(predict, label):
    y_hat = np.where(predict<0.5, 0, 1)
    cnt = 0
    accuracy = 0
    for j in range(len(y_hat)):
        if y_hat[j] == label[j]:
            cnt += 1
    accuracy = cnt/len(y_hat)
    return accuracy


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
        acc_list.append(accuracy(predict, label))        
   
        if i%10000 == 0:
            print("epoch : {0:2}  w0 = {1:.5f}, w1 = {2:.5f}, w2 = {3:.5f} CEE = {4:.5f}".format(i, w0, w1, w2, CEE))
     
      
    print("----" * 20)
    print("\nepoch : {0:2}  w0 = {1:.5f}, w1 = {2:.5f}, w2 = {3:.5f} CEE = {4:.5f}".format(i, w0, w1, w2, CEE))
    print('\nfinal CEE : {} \noptimal w0 : {} \noptimal w1 : {} \noptimal w2 : {}'.format(CEE, w0, w1, w2))
    print('\nAccuracy : ', accuracy(predict, label))
    
    decision_boundary = -(w0*x)/w1 - w2/w1
    plt.figure()
    plt.plot(x, decision_boundary, 'r')
    plt.plot(x_0, y_0, 'ro', label='0')
    plt.plot(x_1, y_1, 'bx', label='1')
    
    
GDM(x, y, label, lr=0.03, epochs=200000)

plt.figure()
plt.plot(acc_list, 'ko-', markerfacecolor='none', markevery=10000)
plt.legend(['Accuracy'])
plt.xlabel('Step')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()