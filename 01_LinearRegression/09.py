# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 03:47:00 2022

@author: syj47
"""

# 실습 09
# 주어진 데이터에 대해 K개의 가우스 함수를 이용한 선형 기저함수 회귀모델의 최적 매개변수(해석해)를 자동 계산하는 프로그램을 작성하고, 
# K가 3, 5, 8일 때의 매개변수를 구하라.
# 결과물: 코드, 매개변수

import numpy as np
import pandas as pd

raw_data = pd.read_csv('lin_regression_data_01.csv', names=['months', 'height'])

x = raw_data['months'].values
y = raw_data['height'].values

# 가우시안 기저 함수 생성하는 함수
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

def optimize(x, y, K):
    phi_K = phi(x, K)
    phi_K_T = phi_K.transpose()
    w = np.linalg.inv(phi_K_T.dot(phi_K)).dot(phi_K_T).dot(y)
    return w

print('\nK = 3\nw = ', optimize(x, y, 3))
print('\nK = 5\nw = ', optimize(x, y, 5))
print('\nK = 8\nw = ', optimize(x, y, 8))