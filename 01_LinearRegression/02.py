# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 15:16:09 2022

@author: syj47
"""

# 실습 02
# 제공된 데이터를 모두 이용하여 최적 선형회귀를 위한 해석해를 구하라.
# 결과물: 코드, 최적해


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
print('w0 = {}, \nw1 = {}'.format(w0, w1))