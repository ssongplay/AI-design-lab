# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 18:59:33 2022

@author: syj47
"""

# 실습 07
# 실습 06에서 각 홀드아웃의 결과로 생성된 선형 기저함수 모델을 
# 각각의 훈련데이터, 검증데이터와 함께 그래프에 표시하라.
# 결과물: 코드, 그래프


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

raw_data = pd.read_csv('lin_regression_data_03.csv', names=['months', 'height'])
x = raw_data['months'].values
y = raw_data['height'].values

k = 5  #k=5인 교차검증
K = 9  # K = 9인 선형 기저함수 모델

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
    
    ###################그래프############################
    x_gp = np.linspace(min(x), max(x), 100)
    y_hat_gp = find_y_hat(x_train, x_gp, w)
    
    plt.subplot(3, 2, i+1)
    plt.scatter(x_train, y_train, label='S')
    plt.scatter(x_valid, y_valid, label='T')
    plt.plot(x_gp, y_hat_gp, 'r', label=r'$\hat y$')
    plt.legend()
    plt.xlabel('months')
    plt.ylabel('height')
    plt.grid(True)
    plt.title('Babies\' Height (CV{})'.format(i))