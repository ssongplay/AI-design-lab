# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 14:59:26 2022

@author: syj47
"""

# 실습 01 
# 제공된 데이터 파일을 불러들여 x축은 나이, y축은 키를 나타내는 2차원 평면에 각 데이터의 위치를 점으로 표시하라.
# 필수요소: x축, y축 이름, grid, legend
# 결과물: 코드, 그래프


import pandas as pd
import matplotlib.pyplot as plt

raw_data = pd.read_csv('lin_regression_data_01.csv', names=['months', 'height'])
x = raw_data['months'].values
y = raw_data['height'].values

plt.plot(x, y, 'b.')
plt.legend(['data'])
plt.xlabel('months')
plt.ylabel('height')
plt.grid(True)
plt.title('Babies\' Height')
plt.show()