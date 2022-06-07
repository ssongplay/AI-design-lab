# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 22:22:25 2022

@author: syj47
"""

# 실습 08
# 훈련 데이터, 해석해를 이용해 구한 회귀모델, 경사하강법을 이용해 구한 회귀모델을 하나의 그래프에 표시하라.
# 필수요소: x축, y축 이름, grid, legend
# 결과물: 코드, 그래프

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

########### 훈련 데이터 ###########
raw_data = pd.read_csv('lin_regression_data_01.csv', names=['months', 'height'])

x = raw_data['months'].values
y = raw_data['height'].values

########### 해석해 회귀모델 ###########
X = np.c_[x, np.ones(x.shape)]
w = np.linalg.inv(np.transpose(X)@X)@np.transpose(X)@y
y_hat = w[0]*x + w[1]

########### GDM 회귀모델 ###########
def GDM(x, y, lr, epochs):
    
    w0 = np.random.uniform(low=-1.0, high=1.0)
    w1 = np.random.uniform(low=-1.0, high=1.0)

    for i in range(epochs):
        y_hat = w0*x + w1
        MSE = ((y_hat-y)**2).mean()
        
        if MSE < 0.0001:
            break

        w0 = w0 - lr * (x*(y_hat-y)).mean()
        w1 = w1 - lr * (y_hat-y).mean()
    
    y_hat_GDM = y_hat
    return y_hat_GDM

y_hat_GDM = GDM(x, y, lr=0.015,epochs=6000)

########### 그래프 ###########
plt.plot(x, y, 'b.', x, y_hat, 'r-^', x, y_hat_GDM, 'g')
plt.legend(['data', 'optimal', 'GDM'])
plt.xlabel('months')
plt.ylabel('height')
plt.grid(True)
plt.title('Babies\' Height')
plt.show()