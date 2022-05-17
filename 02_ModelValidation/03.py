# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 14:35:43 2022

@author: syj47
"""

# 실습 03
# 실습 02에서 만든 훈련 집합을 적용해 K=6, 7, 8, 9, 10, 11, 12, 13일 때의 
# 가우스 함수를 이용한 선형 기저함수 모델의 최적해를 구하라. (K는 기저함수의 개수를 의미함)
# 결과물: 코드, 최적해


import numpy as np
import pandas as pd


raw_data = pd.read_csv('lin_regression_data_03.csv', names=['months', 'height'])

train = raw_data[:20]
valid = raw_data[20:]

x_train = train['months'].values
y_train = train['height'].values
x_valid = valid['months'].values
y_valid = valid['height'].values

def optimize(x, y, K):

    phi = np.ones((len(x), K))
    sigma = (max(x)-min(x)) / (K-1)
    
    for k in range(K):
        mu = min(x) + (max(x)-min(x))*k/(K-1)
        gau_k = np.exp(-0.5*(x-mu)**2 / sigma**2)
        phi[:,k] = gau_k
    phi = np.c_[ phi, np.ones((len(x),)) ]
    
    phi_T = phi.transpose()
    w = np.linalg.inv(phi_T@phi)@phi_T@y
    print('K={}\nw={}\n'.format(K, w))
    
# 선형기저함수의 개수 K
total_K = [6, 7, 8, 9, 10, 11, 12, 13]
for i in total_K:
    optimize(x_train, y_train, i)