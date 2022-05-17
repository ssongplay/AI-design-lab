# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 19:00:58 2022

@author: syj47
"""

# 실습 08
# 실습 06을 이용해 5겹 교차검증을 완성하고 최종 매개변수와 일반화 오차를 구하라.
# 이 결과의 의미를 실습 03과 비교하여 설명하라.
# 결과물 : 코드, 매개변수, 일반화 오차


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

raw_data = pd.read_csv('lin_regression_data_03.csv', names=['months', 'height'])
x = raw_data['months'].values
y = raw_data['height'].values

k = 5  #k=5인 교차검증
K = 9  # K = 9인 선형 기저함수 모델

MSE_valid_list = []
w_list = []

def find_w(x, y):
        
    phi = np.ones((len(x), K))
    sigma = (max(x)-min(x)) / (K-1)
    
    for i in range(K):
        mu = min(x) + (max(x)-min(x))*i/(K-1)
        gau_k = np.exp(-0.5*(x-mu)**2 / sigma**2)
        phi[:,i] = gau_k
    phi = np.c_[ phi, np.ones((len(x),)) ]   
    phi_T = phi.transpose()

    w = np.linalg.inv(phi_T@phi)@phi_T@y

    return w


def find_y_hat(x_train, x_valid, w):

    sigma = (max(x_train)-min(x_train)) / (K-1)
    
    y_hat = 0
    for i in range(K):
        mu = min(x_train) + (max(x_train)-min(x_train))*i/(K-1)
        gau_k = np.exp(-0.5*(x_valid-mu)**2 / sigma**2)
        y_hat += w[i]*gau_k 
    y_hat += w[K]
    
    return y_hat


for i in range(k):
    x_train = []
    y_train = []
    x_valid = []
    y_valid = []
     
    for j in range(25):
        if j in range(i*k, (i+1)*k):
            x_valid.append(x[j])
            y_valid.append(y[j])
        else:
            x_train.append(x[j])
            y_train.append(y[j])
    
   
    w = find_w(x_train, y_train)
    w_list.append(w)

    y_hat_valid = find_y_hat(x_train, x_valid, w)
    MSE_valid = np.square(y_hat_valid-y_valid).mean()
    MSE_valid_list.append(MSE_valid)


final_w = np.mean(w_list, axis=0)
print('Final w = ', final_w)
final_MSE = np.mean(MSE_valid_list)
print('Final MSE = ', final_MSE)


# 최종 모델을 적용한 그래프 
x_gp = np.linspace(min(x), max(x), 100)
final_y_hat_gp = find_y_hat(x, x_gp, final_w)
plt.figure()
plt.scatter(x, y, label='data')
plt.plot(x_gp, final_y_hat_gp, 'r', label=r'Final $\hat y$')
plt.legend()
plt.xlabel('months')
plt.ylabel('height')
plt.grid(True)
plt.text(11, 15, '5-Fold Cross Validation', bbox=dict(facecolor='yellow'))
plt.title('Babies\' Height')