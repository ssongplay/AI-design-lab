# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 21:13:51 2022

@author: syj47
"""

# 실습 06
# 실습 05에서 작성한 경사하강법 프로그램을 이용해 최적 매개변수를 구하라. 단, 학습률, 초기값, 반복 회수는 임의로 정하여 사용하라.
# 결과물: 학습률, 초기값, 반복 회수, 최종 평균제곱오차, 최적 매개변수


import numpy as np
import pandas as pd

raw_data = pd.read_csv('lin_regression_data_01.csv', names=['months', 'height'])
x = raw_data['months'].values
y = raw_data['height'].values


def GDM(x, y, lr, epochs):
    
    w0 = np.random.uniform(low=-1.0, high=1.0)
    w1 = np.random.uniform(low=-1.0, high=1.0)
    init_w0 = w0
    init_w1 = w1
    
    for i in range(epochs):
        y_hat = w0*x + w1
        MSE = ((y_hat-y)**2).mean()
        
        if MSE < 0.0001:
            break

        w0 = w0 - lr * (x*(y_hat-y)).mean()
        w1 = w1 - lr * (y_hat-y).mean()
       
        if i%100 == 0:
            print("epoch : {0:2}  w0 = {1:.5f}, w1 = {2:.5f} MSE = {3:.5f}".format(i, w0, w1, MSE))
        
    print("----" * 15)
    print("epoch : {0:2}  w0 = {1:.5f}, w1 = {2:.5f} MSE = {3:.5f}\n".format(i, w0, w1, MSE))
    print('learning rate : {} \ninitial w0 : {}, initial w1 : {} \nepochs : {} \nfinal MSE : {} \noptimal w0 : {}, optimal w1 : {}'.format(lr, init_w0, init_w1, epochs, MSE, w0, w1))

GDM(x, y, lr=0.015,epochs=6000)