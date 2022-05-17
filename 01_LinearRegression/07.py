# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 21:55:17 2022

@author: syj47
"""

# 실습 07
# 경사하강법의 반복 회수에 따른 평균제곱오차, 매개변수의 값을 그래프로 표시하라. 
# 즉, 교재 7쪽의 그림 3, 4와 같은 그래프를 구하라.
# 결과물: 그래프

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

raw_data = pd.read_csv('lin_regression_data_01.csv', names=['months', 'height'])

x = raw_data['months'].values
y = raw_data['height'].values

w0 = np.random.uniform(low=-1.0, high=1.0)
w1 = np.random.uniform(low=-1.0, high=1.0)
lr = 0.015
epochs = 6000

w0_list = []
w1_list = []
errors = []

# 경사하강법 시작
for i in range(epochs):
    y_hat = w0*x + w1
    MSE = ((y_hat-y)**2).mean()
    
    if MSE < 0.0001:
        break

    w0 = w0 - lr * (x*(y_hat-y)).mean()
    w1 = w1 - lr * (y_hat-y).mean()
    
    w0_list.append(w0)
    w1_list.append(w1)
    errors.append(MSE)

plt.subplot(2, 1, 1)    
plt.plot(w0_list, 'b', w1_list, 'g')
plt.legend([r'$w_0$', r'$w_1$'])
plt.ylabel(r'$w_0, w_1$')
plt.grid(True)
plt.title(r'$\alpha = 0.015$')

plt.subplot(2, 1, 2)    
plt.plot(errors, 'r')
plt.legend([r'MSE'])
plt.xlabel('Step')
plt.ylabel('MSE')
plt.grid(True)
plt.show()