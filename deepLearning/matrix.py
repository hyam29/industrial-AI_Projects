# -*- coding: utf-8 -*-
# 행렬의 곱

import numpy as np

A = np.array([[1,2], [3,4]])
print(A.shape)

B = np.array([[5,6], [7,8]])
print(B.shape)

print(np.dot(A, B))

A = np.array([[1,2,3], [4,5,6]])
print(A.shape)

B = np.array([[1,2], [3,4], [5,6]])
print(B.shape)

print(np.dot(A, B))

'''
C = np.array([1,2], [3,4])
print(C.shape)
# np.dot(A, C) -> 차원의 원소 수 (열 수)가 달라 오류
'''

A = np.array([[1,2], [3,4], [5,6]])
B = np.array([7,8])
print(A.shape)
print(B.shape)
print(np.dot(A, B))
# B를 2행 1열로 인식하여 계산이 가능


