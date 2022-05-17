# -*- coding: utf-8 -*-
"""
Created on Tue May  3 14:01:07 2022

@author: syj47
"""

# 실습 01
# 제공된 데이터 파일을 불러들여 x, y로 구성된 2차원 평면에 각 데이터의 위치를 표시하라. 
# 이때 클래스 0은 빨간색 ’o’ 마커를, 클래스 1의 데이터는 파란색 ‘x’마커를 이용해 표시하라.
# 필수요소: x축, y축 이름, grid, legend
# 결과물: 코드, 그래프

import pandas as pd
import matplotlib.pyplot as plt

raw_data = pd.read_csv('logistic_regression_data.csv', names=['num', 'x', 'y', 'label'])
x = raw_data['x'].values
y = raw_data['y'].values
label = raw_data['label'].values

x_0 = []
x_1 = []
y_0 = []
y_1 = []

for i in range(len(x)):
    if label[i]==0:
        x_0.append(x[i])
        y_0.append(y[i])
    else:
        x_1.append(x[i])
        y_1.append(y[i])
    
plt.plot(x_0, y_0, 'ro', label='0')
plt.plot(x_1, y_1, 'bx', label='1')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.title('class')
plt.show()