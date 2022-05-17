# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 14:48:41 2022

@author: syj47
"""

# 실습 04
# 실습 03에서 구한 선형 기저함수 모델의 평균제곱오차(MSE)를 
# 훈련 집합과 테스트 집합에 대해 각각 구하고 그래프를 그려라.
# 결과물: 코드, 그래프


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

raw_data = pd.read_csv('lin_regression_data_03.csv', names=['months', 'height'])

train = raw_data[:20]
valid = raw_data[20:]

x_train = train['months'].values
y_train = train['height'].values
x_valid = valid['months'].values
y_valid = valid['height'].values


def find_w(x, y, K):
        
    phi = np.ones((len(x), K))
    sigma = (max(x)-min(x)) / (K-1)
    
    for k in range(K):
        mu = min(x) + (max(x)-min(x))*k/(K-1)
        gau_k = np.exp(-0.5*(x-mu)**2 / sigma**2)
        phi[:,k] = gau_k
    phi = np.c_[ phi, np.ones((len(x),)) ]   
    phi_T = phi.transpose()
    
    w = np.linalg.inv(phi_T@phi)@phi_T@y
    
    return w


def find_y_hat(x, w, K):

    sigma = (max(x_train)-min(x_train)) / (K-1)
    
    y_hat = 0
    for k in range(K):
        mu = min(x_train) + (max(x_train)-min(x_train))*k/(K-1)
        gau_k = np.exp(-0.5*(x-mu)**2 / sigma**2)
        y_hat += w[k]*gau_k 
    y_hat += w[K]
    
    return y_hat

# 선형기저함수의 개수 K
total_K = [6, 7, 8, 9, 10, 11, 12, 13]

total_MSE_train = []
total_MSE_valid = []

for k in total_K:
    
    w = find_w(x_train, y_train, k)    
    
    y_hat_train = find_y_hat(x_train, w, k)
    MSE_train = np.square(y_hat_train-y_train).mean()
    total_MSE_train.append(MSE_train)
    
    y_hat_valid = find_y_hat(x_valid, w, k)
    MSE_valid = np.square(y_hat_valid-y_valid).mean()
    total_MSE_valid.append(MSE_valid)
    

print('MSE_train = ', total_MSE_train)
print('\nMSE_valid = ', total_MSE_valid)    

plt.figure()
plt.title('MSE')
plt.plot(total_K,total_MSE_train, 'r-o', total_K, total_MSE_valid, 'b-o')
plt.legend(['MSE_train', 'MSE_valid'])
plt.xlabel("K")
plt.ylabel("MSE")
plt.grid(True)
plt.show()