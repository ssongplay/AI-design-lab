# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 16:26:44 2022

@author: syj47
"""

# 실습 04
# 해석해로 구한 선형모델의 평균제곱오차를 구하라.
# 결과물: 코드, 평균제곱오차

import numpy as np
import pandas as pd

raw_data = pd.read_csv('lin_regression_data_01.csv', names=['months', 'height'])
x = raw_data['months'].values
y = raw_data['height'].values

X = np.c_[x, np.ones(x.shape)]
w = np.linalg.inv(np.transpose(X)@X)@np.transpose(X)@y
y_hat = w[0]*x + w[1]

MSE = ((y_hat-y)**2).mean()

print('MSE = ', MSE)