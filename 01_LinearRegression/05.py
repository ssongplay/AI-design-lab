# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 17:55:52 2022

@author: syj47
"""

# 실습 05
# 실험적으로 최적 매개변수를 찾기 위한 경사하강법 알고리즘을 프로그램으로 작성하라. 
# 단, 경사하강법 외부 함수 사용 금지.
# 결과물: 코드


import numpy as np

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
