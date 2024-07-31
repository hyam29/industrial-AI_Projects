# -*- coding: utf-8 -*-

"""
def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    temp = x1*w1 + x2*w2
    if temp <= theta:
        return 0
    elif temp > theta:
        return 1

print(AND(0, 0))
print(AND(1, 0))
print(AND(0, 1))
print(AND(1, 1))
"""

import numpy as np

def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    temp = np.sum(w*x) + b
    if temp <= 0:
        return 0
    else:
        return 1

print(AND(0, 0))
print(AND(1, 0))
print(AND(0, 1))
print(AND(1, 1))
print()

def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    temp = np.sum(w*x) + b
    if temp <= 0:
        return 0
    else:
        return 1

print(NAND(0, 0))
print(NAND(1, 0))
print(NAND(0, 1))
print(NAND(1, 1))
print()

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    temp = np.sum(w*x) + b
    if temp <= 0:
        return 0
    else:
        return 1

print(OR(0, 0))
print(OR(1, 0))
print(OR(0, 1))
print(OR(1, 1))
print()

def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

print(XOR(0, 0))
print(XOR(1, 0))
print(XOR(0, 1))
print(XOR(1, 1))
print()

# 반가산기
def half_adder(x1, x2):
    s = XOR(x1, x2) # sum
    c = AND(x1, x2) # carry
    return s, c

print(half_adder(0, 0))
print(half_adder(1, 0))
print(half_adder(0, 1))
print(half_adder(1, 1))
print()

# 전가산기
def full_adder(x1, x2, carry):
    s1, c1 = half_adder(x1, x2)
    s2, c2 = half_adder(s1, carry)
    return s2, OR(c1, c2)
print(full_adder(0, 0, 0))  # (0, 0)
print(full_adder(1, 0, 0))  # (1, 0)
print(full_adder(0, 1, 0))  # (1, 0)
print(full_adder(1, 1, 0))  # (0, 1)
