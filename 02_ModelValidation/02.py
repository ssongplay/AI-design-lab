# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 14:26:54 2022

@author: syj47
"""

# 실습 02
# 전체 데이터 중 첫 20개(1번~20번)를 훈련 집합(S)으로 후반 5개(21번~25번)를
# 테스트 집합(T)으로 나누고 각 집합의 데이터를 그래프로 나타내어라. (주의: 데이터의 순서를 바꾸지 말 것)
# 결과물: 코드, 그래프


import pandas as pd
import matplotlib.pyplot as plt

raw_data = pd.read_csv('lin_regression_data_03.csv', names=['months', 'height'])

train = raw_data[:20]
valid = raw_data[20:]

x_train = train['months'].values
y_train = train['height'].values
x_valid = valid['months'].values
y_valid = valid['height'].values

plt.scatter(x_train, y_train)
plt.scatter(x_valid, y_valid)
plt.legend(['training set (S)', 'validation set (T)'])
plt.xlabel('months')
plt.ylabel('height')
plt.grid(True)
plt.title('Babies\' Height')
plt.show()