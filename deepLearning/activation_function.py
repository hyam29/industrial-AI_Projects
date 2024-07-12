# -*- coding: utf-8 -*-

import numpy as np

# 계단 함수 구현
'''
def step_function(x):
    if x>0:
        return 1
    else:
        return 0

print(step_function(3.0))
'''
# 위의 함수는 실수만 받을 수 있음 (넘파이 배열을 인수로 넣을 수 없음 따라서 아래대로 구현)
def step_function(x):
    y = x > 0
    return y.astype(int)

print(step_function(np.array([1.0, 2.0])))

x = np.array([-0.1, 1.0, 2.0])
print(x)


y = x > 0
print(y)

y = y.astype(int)
print(y)

import matplotlib.pylab as plt

def step_function(x):
    return np.array(x > 0, dtype=int)

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()


# 시그모이드 함수 구현

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x = np.array([-1.0, 1.0, 2.0])
print(sigmoid(x))

t = np.array([1.0, 2.0, 3.0])
print(1.0 + t)
print(1.0 / t)

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()


# ReLU 함수 구현

def relu(x):
    return np.maximum(0, x)

x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()