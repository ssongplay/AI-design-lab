# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 16:26:44 2022

@author: syj47
"""

# 실습 04
# 해석해로 구한 선형모델의 평균제곱오차를 구하라.
# 결과물: 코드, 평균제곱오차


import pandas as pd
from sympy import symbols, Eq, solve

raw_data = pd.read_csv('lin_regression_data_01.csv', names=['months', 'height'])
x = raw_data['months'].values
y = raw_data['height'].values

w0, w1 = symbols('w0 w1')
eqn_w0 = Eq((x*((w0*x + w1) -y)).mean(), 0)
eqn_w1 = Eq(((w0*x + w1) -y).mean(), 0)
answer = solve([eqn_w0, eqn_w1])
w0 = answer[w0]
w1 = answer[w1]
y_hat = w0*x + w1

MSE = ((y_hat-y)**2).mean()

print('MSE = ', MSE)