# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 14:35:51 2022

@author: syj47
"""

# 실습 06
# 실습 05에서 만든 다섯 개의 데이터 집합을 이용해 5겹 교차검증을 구현하려고한다. 
# 모델은 K=9인 선형 기저함수 모델이다. 이를 위해 5개의 홀드아웃 검증을
# 설계하고 각 홀드아웃의 결과물(매개변수, 일반화 오차)을 구하라.
# 결과물: 코드, 매개변수, 일반화 오차


import numpy as np
import pandas as pd

raw_data = pd.read_csv('lin_regression_data_03.csv', names=['months', 'height'])
x = raw_data['months'].values
y = raw_data['height'].values

k = 5  #k=5인 교차검증
K = 9  # K = 9인 선형 기저함수 모델

MSE_train_list = []
MSE_valid_list = []

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

    y_hat_valid = find_y_hat(x_train, x_valid, w)
    MSE_valid = np.square(y_hat_valid-y_valid).mean()
    MSE_valid_list.append(MSE_valid)
    
    print('\n< Hold-Out {} >\nw = {}\nGeneralization Error = {}\n'.format(i+1, w, MSE_valid))
    print('----------------------------------------------------------')
