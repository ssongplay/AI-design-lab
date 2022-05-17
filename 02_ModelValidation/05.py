# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 14:09:55 2022

@author: syj47
"""

# 실습 05
# 전체 데이터를 차례로 5등분하여 5개의 부분집합으로 나누고, 
# 각 집합의 데이터를 x축은 나이, y축은 키를 나타내는 2차원 평면에 
# 서로 다른 모양의 마커로 표시하라. (k=5인 교차검증을 위한 준비 작업)
# 결과물: 코드, 그래프 (범례와 함께)


import pandas as pd
import matplotlib.pyplot as plt

raw_data = pd.read_csv('lin_regression_data_03.csv', names=['months', 'height'])
x = raw_data['months'].values
y = raw_data['height'].values

k = 5  #k=5인 교차검증

for i in range(k):
    x_train = []
    y_train = []
    x_valid = []
    y_valid = []
    for j in range(25):
        if j in range(i*k, (i+1)*k):
            x_valid.append(x[j])
            y_valid.append(y[j])
        else:
            x_train.append(x[j])
            y_train.append(y[j])

    plt.scatter(x_valid, y_valid)    
    
plt.legend([r'$D_0$', r'$D_1$', r'$D_2$', r'$D_3$', r'$D_4$'])
plt.xlabel('months')
plt.ylabel('height')
plt.grid(True)
plt.title('Babies\' Height')
plt.show()