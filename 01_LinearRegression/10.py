# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 04:03:36 2022

@author: syj47
"""

# 실습 10
# K=3, 5, 8, 10일 때, 훈련 데이터와 선형 기저함수 회귀 모델을 그래프에 표시하라.
# 필수요소: x축, y축 이름, grid, legend
# 결과물: 그래프

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

raw_data = pd.read_csv('lin_regression_data_01.csv', names=['months', 'height'])

x = raw_data['months'].values
y = raw_data['height'].values

def Gaussian(x, mu, sigma):
    return np.exp(-0.5*(x-mu)**2 / sigma**2)

def phi(x, K):
    phi = np.ones((len(x), K))
    sigma = (max(x)-min(x)) / (K-1)
    for k in range(K):
        mu = min(x) + (max(x)-min(x))*k/(K-1)
        gau_k = Gaussian(x, mu, sigma)
        phi[:,k] = gau_k  
    phi = np.c_[ phi, np.ones((len(x),)) ]
    return phi

def find_w(x, y, K):
    phi_K = phi(x, K)
    phi_K_T = phi_K.transpose()
    w = np.linalg.inv(phi_K_T.dot(phi_K)).dot(phi_K_T).dot(y)
    return w

def show_y_hat(x, y, K):
    
    w = find_w(x, y, K)
    sigma = (max(x)-min(x)) / (K-1)
    y_hat = 0
    x_axis = np.linspace(min(x), max(x), 25)
    for k in range(K):
        mu = min(x) + (max(x)-min(x))*k/(K-1)
        y_hat += w[k]*Gaussian(x_axis, mu, sigma)
    y_hat += w[K]
    plt.plot(x, y, 'bo', label='data')
    plt.plot(x_axis, y_hat,'r-', label = 'Gaussian (K={})'.format(K))
    plt.legend()
    plt.xlabel('months')
    plt.ylabel('height')
    plt.grid(True)
    plt.title('Babies\' Height')    
    

plt.subplot(2, 2, 1)    
show_y_hat(x, y, 3)
plt.subplot(2, 2, 2) 
show_y_hat(x, y, 5)
plt.subplot(2, 2, 3) 
show_y_hat(x, y, 8)
plt.subplot(2, 2, 4) 
show_y_hat(x, y, 10)



    
    
    
    