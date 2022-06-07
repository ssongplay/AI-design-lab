# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 15:16:09 2022

@author: syj47
"""

# 실습 02
# 제공된 데이터를 모두 이용하여 최적 선형회귀를 위한 해석해를 구하라.
# 결과물: 코드, 최적해


import pandas as pd
import numpy as np

raw_data = pd.read_csv('lin_regression_data_01.csv', names=['months', 'height'])
x = raw_data['months'].values
y = raw_data['height'].values

X = np.c_[x, np.ones(x.shape)]

w = np.linalg.inv(np.transpose(X)@X)@np.transpose(X)@y
print('w0 = {}, \nw1 = {}'.format(w[0], w[1]))