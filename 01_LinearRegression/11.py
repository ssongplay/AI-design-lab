# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 17:12:02 2022

@author: syj47
"""

# 실습 11
# 실습 10에서 각 K에 대한 평균제곱오차를 구하고 x축은 K값, y축은 평균제곱오차를 나타내는 2차원 그래프를 구하라.
# 필수요소: x축, y축 이름, grid, legend
# 결과물: 코드, 그래프

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
    plt.plot(x_axis, y_hat, label = 'Gaussian (K={})'.format(K))
     
def find_MSE(x, y, K):
    w = find_w(x, y, K)
    sigma = (max(x)-min(x)) / (K-1)
    y_hat = 0
    x_axis = np.linspace(min(x), max(x), 25)
    for k in range(K):
        mu = min(x) + (max(x)-min(x))*k/(K-1)
        y_hat += w[k]*Gaussian(x_axis, mu, sigma)
    y_hat += w[K]
    MSE = ((y_hat-y)**2).mean()
    return MSE

for i in range(4):
    MSE = []
    K = [3, 5, 8, 10]
    for k in K:
        MSE.append(find_MSE(x, y, k))
print(MSE)
plt.plot(K, MSE, 'bo-')
plt.legend(['MSE'])
plt.xlabel('K')
plt.ylabel('MSE')
plt.grid(True)