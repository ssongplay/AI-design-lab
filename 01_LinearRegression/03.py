# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 16:21:13 2022

@author: syj47
"""

# 실습 03
# 해석해로 구한 선형모델과 데이터를 한 그래프에 표시하라.
# 필수요소: x축, y축 이름, grid, legend
# 결과물: 그래프

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

raw_data = pd.read_csv('lin_regression_data_01.csv', names=['months', 'height'])
x = raw_data['months'].values
y = raw_data['height'].values

X = np.c_[x, np.ones(x.shape)]
w = np.linalg.inv(np.transpose(X)@X)@np.transpose(X)@y
y_hat = w[0]*x + w[1]

plt.plot(x, y, 'b.', x, y_hat, 'r')
plt.legend(['y', 'y_hat'])
plt.xlabel('months')
plt.ylabel('height')
plt.grid(True)
plt.title('Babies\' Height')
plt.show()